from typing import Dict, Iterator, List, Literal, Optional, Union
import anthropic
import time

from llm_gateway.llmabc import LLMAbstractBaseClass
from llm_gateway.llms import Message
from llm_gateway.utils import retry_with_exponential_backoff


class AnthropicLLM(LLMAbstractBaseClass):
    def __init__(self, model, tokens: Optional[Dict[str, str]]) -> None:
        super().__init__(model, tokens)
        token = tokens.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(
            api_key=token,
            timeout=600,
            max_retries=0,
        )

    @retry_with_exponential_backoff
    def completion_with_backoff(
        self,
        *,
        messages: List[Message],
        model: Union[
            str,
            Literal[
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-2.1'",
                "claude-2.0",
                "claude-instant-1.2",
            ],
        ],
        system: Optional[str],
        max_tokens: int,
        stop: Union[Optional[str], List[str]],
        stream: Optional[Literal[True, False]],
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
    ):
        time.sleep(0.05)
        return self._client.messages.create(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            stop_sequences=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def raw_inference(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ) -> str:
        raise NotImplementedError

    def raw_stream_inference(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ) -> Iterator[str]:
        raise NotImplementedError

    def inference(
        self,
        prompt: List[Message],
        suffix: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: int | None = None,
        echo: bool = False,
        stop: str | List[str] | None = ...,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ) -> Message:
        system, messages = self.extract(prompt)
        temperature = temperature / 2
        comp = self.completion_with_backoff(
            messages=messages,
            model=self.model.value,
            system=system,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=False,
        )
        assert comp.content[0].type == "text"
        return {
            "role": comp.role,
            "content": comp.content[0].text,
        }

    def stream_inference(
        self,
        prompt: List[Message],
        suffix: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
    ):
        system, messages = self.extract(prompt)
        temperature = temperature / 2
        completion_role = ""
        for chunk in self.completion_with_backoff(
            messages=messages,
            model=self.model.value,
            system=system,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=True,
        ):
            if chunk.type == "message_start":
                completion_role = chunk.message.role
            if chunk.type == "content_block_delta":
                assert completion_role != ""
                assert chunk.delta.type == "text_delta"
                yield {
                    "role": completion_role,
                    "content": chunk.delta.text,
                }

    def extract(self, prompt):
        system = "\n".join([message["content"] for message in prompt if message["role"] == "system"])
        messages = [message for message in prompt if message["role"] != "system"]
        return system, messages
