import functools
import os
import signal
import traceback
from contextlib import asynccontextmanager
from hashlib import blake2b

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette import EventSourceResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from llm_gateway.backends.abc import LLMAbstractBaseClass
from llm_gateway.backends.backend_anthropic import AnthropicBackend
from llm_gateway.backends.backend_gemini import GeminiBackend
from llm_gateway.backends.backend_openai import OpenAIBackend

# from llm_gateway.backends.backend_runpod import RunPodBackend
from llm_gateway.types.openai.chat_completion.protocol import ChatCompletionRequest

registry: dict[str, LLMAbstractBaseClass] = {}
tokens: dict[str, str] = {}


@functools.cache
def get_expected_token_hash():
    return (
        blake2b(os.getenv("LLM_GATEWAY_TOKEN", "").encode()).hexdigest()
        if os.getenv("LLM_GATEWAY_TOKEN")
        else None
    )


# We will handle a missing token ourselves
get_bearer_token = HTTPBearer(auto_error=False)


async def get_token(
    auth: HTTPAuthorizationCredentials | None = Depends(get_bearer_token),
) -> str:
    expected_token_hash = get_expected_token_hash()
    if expected_token_hash is None:
        return ""

    if auth is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )
    token = auth.credentials
    if blake2b(token.encode()).hexdigest() != expected_token_hash:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    return token


def token_from_env(env_var: str):
    if tokens.get(env_var) is None:
        env = os.getenv(env_var)
        if env is not None:
            tokens[env_var] = env


def model_parse(name: str) -> tuple[LLMAbstractBaseClass, str]:
    """
    Retrieve the backend instance by name.
    """
    name = name.lower()
    index = name.find(":")
    if index != -1:
        backend = name[:index]
        name = name[index + 1 :]
    else:
        raise HTTPException(
            detail="Backend name must be in the format 'namespace:model'.",
            status_code=400,
        )
    if backend not in registry:
        raise HTTPException(
            detail=f"Backend '{backend}' is not supported.",
            status_code=404,
        )
    return registry[backend], name


router = APIRouter()

@router.get("/")
async def root():
    return {"message": "LLM Gateway API"}

routerv1 = APIRouter(prefix="/v1", tags=["v1"])


@routerv1.post("/chat/completions")
async def create_chat_completion(
    body: ChatCompletionRequest, token: str = Depends(get_token)
):
    """
    Handle chat completions.
    """
    try:
        if not body.model:
            raise HTTPException(status_code=400, detail="Model name is required.")
        [backend, name] = model_parse(body.model)
        body.model = name
        if not body.logprobs:
            body.top_logprobs = None
        if body.max_tokens is not None and body.max_completion_tokens is None:
            print("max_tokens is deprecated, using max_completion_tokens instead")
            body.max_completion_tokens = body.max_tokens
            body.model_fields_set.discard("max_tokens")
        if body.stream:

            async def event_generator():
                async for chunk in await backend.create_chat_completion(
                    req=body,
                    stream=True,
                ):
                    yield {"data": chunk.model_dump_json(exclude_unset=True)}

            return EventSourceResponse(event_generator())
        return (await backend.create_chat_completion(
            req=body,
            stream=False,
        )).model_dump(exclude_unset=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def cleanup():
    print("Cleaning up backend ...")
    for namespace, backend in registry.items():
        try:
            backend.cleanup()
        except Exception as e:
            print(f"Error cleaning up for {namespace}: {e}")
    print("Cleanup completed")


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal for graceful shutdown"""
    print(f"Received signal {signum}, initiating shutdown...")
    cleanup()
    import sys

    sys.exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Register SIGTERM handler for graceful pod cleanup
    signal.signal(signal.SIGTERM, handle_sigterm)

    if get_expected_token_hash() is None:
        print("Warning: LLM_GATEWAY_TOKEN is not set, anyone can access the API")

    yield

    # Cleanup: Ensure pods are cleaned up on shutdown
    cleanup()


def create_app() -> FastAPI:
    for env_var in [
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGING_FACE_HUB_TOKEN",
        "RUNPOD_API_KEY",
    ]:
        token_from_env(env_var)

    for backend in (
        OpenAIBackend,
        GeminiBackend,
        AnthropicBackend,
        # RunPodBackend,
    ):
        registry[backend.namespace] = backend(auth_tokens=tokens)

    app = FastAPI(title="LLM Gateway", version="1.0.0", lifespan=lifespan)

    app.include_router(router)
    app.include_router(routerv1)

    app.add_middleware(
        CORSMiddleware, # type: ignore
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc: StarletteHTTPException):
        print(f"HTTP error: {repr(exc)} {traceback.format_exc()}")
        return JSONResponse(
            content={"type": "error", "message": str(exc.detail)},
            status_code=exc.status_code,
        )

    return app
