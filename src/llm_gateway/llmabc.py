from __future__ import annotations
import abc
from typing import Dict, List, Optional, Union, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_gateway.llms import LLMs, Message

from llm_gateway.session import LLMSession


class LLMAbstractBaseClass(abc.ABC):
    """An LLM instance. Use create_session() to create a session for multiple times of inference."""

    def __init__(self, model: LLMs, tokens: Optional[Dict[str, str]] = None):
        self.model = model
        self.tokens = tokens

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def inference(
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
    ) -> Message:
        pass

    @abc.abstractmethod
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
    ) -> Iterator[Message]:
        pass

    def create_session(self) -> LLMSession:
        return LLMSession(self)
