from collections.abc import AsyncGenerator
from typing import Any, Literal, overload

import openai

from llm_gateway.types.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)
from llm_gateway.utils import async_retry

from .abc import LLMAbstractBaseClass


class OpenAIBackend(LLMAbstractBaseClass):
    """
    OpenAI backend implementation for LLMAbstractBaseClass.
    This class implements the methods defined in the abstract base class for OpenAI's API.
    """

    namespace = "openai"
    token_name = "OPENAI_API_KEY"
    base_url: str | None = None

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

    @async_retry(
        errors=(openai.RateLimitError,),
    )
    async def create_chat_completion(
        self,
        req: ChatCompletionRequest,
        stream: Literal[False] | None | Literal[True],
    ):
        auth_token = self.auth_tokens.get(self.token_name)
        if not auth_token:
            raise ValueError(f"Authentication token '{self.token_name}' is required.")
        client = openai.AsyncOpenAI(api_key=auth_token, base_url=self.base_url)
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

    def models(self):
        raise NotImplementedError(
            "The 'models' method is not implemented for OpenAI backend."
        )
