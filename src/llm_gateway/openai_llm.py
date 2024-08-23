from typing import Dict, Iterator, List, Literal, Optional, Union
import openai
import time

from llm_gateway.llmabc import LLMAbstractBaseClass
from llm_gateway.llms import Message
from llm_gateway.utils import retry_with_exponential_backoff


class OpenAILLM(LLMAbstractBaseClass):
    def __init__(self, model, tokens: Optional[Dict[str, str]]) -> None:
        super().__init__(model, tokens)
        token = tokens.get("OPENAI_API_KEY")
        self._client = openai.OpenAI(
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
                "gpt-4o",
                "gpt-4o-2024-05-13",
                "gpt-4-turbo",
                "gpt-4-turbo-2024-04-09",
                "gpt-4-0125-preview",
                "gpt-4-turbo-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-4-32k-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo-0125",
                "gpt-3.5-turbo-16k-0613",
            ],
        ],
        frequency_penalty: Optional[float],
        max_tokens: Optional[int],
        presence_penalty: Optional[float],
        stop: Union[Optional[str], List[str]],
        stream: Optional[Literal[True, False]],
        temperature: Optional[float],
        top_p: Optional[float],
    ):
        time.sleep(0.05)
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
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
        comp = self.completion_with_backoff(
            messages=prompt,
            model=self.model.value,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return {
            "role": comp.choices[0].message.role,
            "content": comp.choices[0].message.content,
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
        for chunk in self.completion_with_backoff(
            messages=prompt,
            model=self.model.value,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        ):
            yield {
                "role": chunk.choices[0].delta.role,
                "content": chunk.choices[0].delta.content,
            }
