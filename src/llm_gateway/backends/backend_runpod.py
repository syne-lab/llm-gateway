import threading
import time
from collections.abc import AsyncGenerator
from hashlib import blake2b
from typing import Any, Literal, overload

import openai
import redis
import runpod

from llm_gateway.states import state_manager
from llm_gateway.types.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)
from llm_gateway.utils import async_retry

from .abc import LLMAbstractBaseClass


class LifecycleManager:
    """Manages RunPod lifecycle using Redis for coordination across workers"""

    def __init__(self, token: str, hf_token: str | None = None):
        self.token = token
        self.vllm_token = blake2b(token.encode(), digest_size=32).hexdigest()
        self.hf_token = hf_token
        self.namespace = "runpod"
        self._expiry_thread: threading.Thread | None = None
        self._stop_expiry = threading.Event()
        self._active_pods: dict[str, str] = {}  # model -> pod_id mapping
        self.redis_manager = state_manager

        # Start expiry listener thread
        self._start_expiry_listener()

    def _start_expiry_listener(self):
        """Start background thread to listen for key expiration events"""
        if self._expiry_thread is None or not self._expiry_thread.is_alive():
            self._stop_expiry.clear()
            self._expiry_thread = threading.Thread(
                target=self._listen_for_expirations, daemon=True
            )
            self._expiry_thread.start()

    def _listen_for_expirations(self):
        """Listen for Redis key expiration events and cleanup pods"""
        try:
            pubsub = self.redis_manager.subscribe_to_expirations(self.namespace)

            while not self._stop_expiry.is_set():
                try:
                    # Get message with timeout to allow periodic checks for stop signal
                    message = pubsub.get_message(timeout=1.0)

                    if message and message["type"] == "pmessage":
                        expired_key = message["data"]

                        # Check if this is a runpod key that expired
                        if expired_key.startswith(f"{self.namespace}:"):
                            model = expired_key.split(":", 1)[1]
                            pod_id = self._active_pods.get(model)

                            if pod_id:
                                print(
                                    f"Pod state expired for model {model}, cleaning up pod {pod_id}"
                                )
                                self._cleanup_expired_pod(model, pod_id)

                except redis.ConnectionError as e:
                    print(f"Redis connection error in expiry listener: {e}")
                    time.sleep(5)  # Wait before retrying
                except Exception as e:
                    print(f"Error in expiry listener: {e}")

        except Exception as e:
            print(f"Fatal error in expiry listener: {e}")
        finally:
            try:
                pubsub.close()
            except Exception:
                pass

    def _cleanup_expired_pod(self, model: str, pod_id: str):
        """Cleanup a pod that has expired"""
        try:
            runpod.api_key = self.token
            runpod.terminate_pod(pod_id)
            print(f"Successfully terminated expired pod {pod_id} for model {model}")
        except Exception as e:
            print(f"Error terminating expired pod {pod_id}: {e}")
        finally:
            # Remove from active pods tracking
            self._active_pods.pop(model, None)

    def get_or_create_pod(self, model: str) -> tuple[str, str]:
        """Get existing pod or create new one for the model"""
        with self.redis_manager.distributed_lock(self.namespace, model):
            # Check if pod already exists
            pod_state = self.redis_manager.get_state(self.namespace, model)

            if pod_state:
                pod_id = pod_state["pod_id"]

                # Verify pod is still running
                if self._is_pod_running(pod_id):
                    # Extend expiry since we're using it
                    self.redis_manager.extend_expiry(self.namespace, model)
                    # Update active pods tracking
                    self._active_pods[model] = pod_id
                    return pod_id, self._get_pod_base_url(pod_id)
                else:
                    # Pod is dead, clean up Redis state
                    self.redis_manager.delete_state(self.namespace, model)
                    dead_pod_id = self._active_pods.pop(model, None)
                    if dead_pod_id:
                        runpod.terminate_pod(dead_pod_id)

            # Create new pod
            pod_id = self._create_pod(model)
            base_url = self._get_pod_base_url(pod_id)

            # Store in Redis with 10-minute expiry
            state = {"pod_id": pod_id, "model": model, "created_at": int(time.time())}
            self.redis_manager.set_state(
                self.namespace, model, state, expiry_seconds=600
            )

            # Track active pod for cleanup
            self._active_pods[model] = pod_id

            return pod_id, base_url

    def cleanup_pod(self, model: str):
        """Manually cleanup a specific pod"""
        with self.redis_manager.distributed_lock(self.namespace, model, timeout=10):
            pod_state = self.redis_manager.get_state(self.namespace, model)
            if pod_state:
                pod_id = pod_state["pod_id"]
                try:
                    runpod.api_key = self.token
                    runpod.terminate_pod(pod_id)
                    print(f"Terminated pod {pod_id} for model {model}")
                except Exception as e:
                    print(f"Error terminating pod {pod_id}: {e}")
                finally:
                    self.redis_manager.delete_state(self.namespace, model)
                    self._active_pods.pop(model, None)

    def cleanup_all_pods(self):
        """Emergency cleanup of all tracked pods"""
        runpod.api_key = self.token

        # Cleanup from active tracking
        for pod_id in self._active_pods.values():
            try:
                runpod.terminate_pod(pod_id)
                print(f"Emergency cleanup: terminated pod {pod_id}")
            except Exception as e:
                print(f"Error during emergency cleanup of pod {pod_id}: {e}")

        # Clear active pods tracking
        self._active_pods.clear()

        # Also cleanup any remaining Redis state
        all_keys = self.redis_manager.get_all_keys(self.namespace)
        for key in all_keys:
            self.redis_manager.delete_state(self.namespace, key)

    def _create_pod(self, model: str) -> str:
        """Create a new RunPod instance"""
        runpod.api_key = self.token

        pod = runpod.create_pod(
            name=f"vllm-{model.replace('/', '-')}",
            image_name="vllm/vllm-openai:latest",
            docker_args=f"--model {model}",
            env={
                "VLLM_API_KEY": self.vllm_token,
                "HUGGING_FACE_HUB_TOKEN": self.hf_token if self.hf_token else "",
            },
            ports="8000/http",
        )

        print(f"Created new pod {pod['id']} for model {model}")
        return pod["id"]

    def _is_pod_running(self, pod_id: str) -> bool:
        """Check if pod is still running"""
        try:
            runpod.api_key = self.token
            pod_info = runpod.get_pod(pod_id)
            return pod_info and pod_info.get("desiredStatus") == "RUNNING"
        except Exception:
            return False

    def _get_pod_base_url(self, pod_id: str) -> str:
        """Get the base URL for the pod"""
        # This would need to be implemented based on RunPod's API
        # For now, return a placeholder
        return f"https://{pod_id}-8000.proxy.runpod.net/v1"

    def shutdown(self):
        """Shutdown the pod manager and cleanup all pods"""
        self._stop_expiry.set()
        if self._expiry_thread:
            self._expiry_thread.join(timeout=5)

        self.cleanup_all_pods()


class RunPodBackend(LLMAbstractBaseClass):
    """
    RunPod backend implementation for LLMAbstractBaseClass.
    This class implements the methods defined in the abstract base class for RunPod's API.
    Uses Redis for coordinating pod lifecycle across multiple workers.
    """

    namespace = "runpod"
    token_name = "RUNPOD_API_KEY"
    hf_token_name = "HUGGING_FACE_HUB_TOKEN"

    def __init__(self, auth_tokens: dict[str, str]):
        super().__init__(auth_tokens)

        # Initialize pod manager with token
        token = auth_tokens.get(self.token_name)
        if not token:
            raise ValueError(f"Authentication token '{self.token_name}' is required.")

        self.pod_manager = LifecycleManager(token)

    @overload
    async def create_chat_completion(
        self,
        req: ChatCompletionRequest,
        stream: Literal[False],
    ) -> ChatCompletionResponse: ...
    @overload
    async def create_chat_completion(
        self,
        req: ChatCompletionRequest,
        stream: Literal[True],
    ) -> AsyncGenerator[ChatCompletionStreamResponse, Any]:
        """
        Handle chat completions.
        This method should be implemented by subclasses to handle chat completion requests.
        """
        ...

    @async_retry(
        errors=(openai.RateLimitError,),
    )
    async def create_chat_completion(
        self,
        req: ChatCompletionRequest,
        stream: Literal[False] | None | Literal[True],
    ):
        # Get or create pod for this model using Redis coordination
        _, base_url = self.pod_manager.get_or_create_pod(req.model)

        auth_token = self.auth_tokens.get(self.token_name)
        if not auth_token:
            raise ValueError(f"Authentication token '{self.token_name}' is required.")

        client = openai.AsyncOpenAI(api_key=auth_token, base_url=base_url)
        params = req.model_dump(exclude_unset=True)
        params.pop("stream", None)
        if stream:

            async def stream_response():
                async for chunk in await client.chat.completions.create(
                    **params,
                    stream=True,
                ):
                    yield chunk

            return stream_response()
        return await client.chat.completions.create(
            **params,
            stream=False,
        )

    def cleanup(self):
        """Cleanup all pods and shutdown the pod manager"""
        self.pod_manager.shutdown()
