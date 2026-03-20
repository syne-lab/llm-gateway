import abc
from collections.abc import AsyncGenerator
from typing import Any, Literal, overload

from llm_gateway.types.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class LLMAbstractBaseClass(abc.ABC):
    """
    Abstract base class for LLM backends.
    This class defines the interface that all LLM backends must implement.
    """

    namespace: str
    token_name: str

    def __init__(self, auth_tokens: dict[str, str]):
        """
        Initialize the LLM backend with optional authentication tokens.

        :param auth_tokens: A dictionary of authentication tokens for the backend.
        """
        self.auth_tokens = auth_tokens


    @abc.abstractmethod
    def cleanup(self):
        pass

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

    @abc.abstractmethod
    async def create_chat_completion(
        self,
        req: ChatCompletionRequest,
        stream: Literal[False] | None | Literal[True],
    ):
        """
        Handle chat completions.
        This method should be implemented by subclasses to handle chat completion requests.
        """
        pass

    # @abc.abstractmethod
    def models(self):
        """
        List available models.
        This method should be implemented by subclasses to return a list of available models.
        """
        raise NotImplementedError
