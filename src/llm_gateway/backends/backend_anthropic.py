from collections.abc import AsyncGenerator, Iterable
from typing import Any, Literal, cast, overload

import anthropic

from llm_gateway.types.openai.chat_completion.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from llm_gateway.utils import async_retry

from .abc import LLMAbstractBaseClass


class AnthropicBackend(LLMAbstractBaseClass):
    namespace = "anthropic"
    token_name = "ANTHROPIC_API_KEY"

    def cleanup(self):
        pass

    def transform_messages(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> tuple[str | None, list[dict[str, str]]]:
        system = None
        contents = []
        for message in messages:
            if message["role"] in ["system", "developer"]:
                assert system is None, "Only one system message is allowed."
                system = cast(str, message["content"])
                continue
            contents.append(
                {
                    "role": message["role"]
                    if message["role"] == "user"
                    else "assistant",
                    "content": message["content"],
                }
            )
        return system, contents

    def transform_chunk(
        self,
        chunk_index: int,
        chunk: anthropic.types.RawContentBlockDeltaEvent,
        model: str,
    ) -> ChatCompletionStreamResponse:
        choices = []
        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(
                role="assistant" if chunk_index == 0 else None,
                content=chunk.delta.text,
            ),
            logprobs=None,
            finish_reason=None,
        )
        choices.append(choice)

        return ChatCompletionStreamResponse(
            model=model,
            choices=choices,
        )

    def transform_response(
        self,
        response: anthropic.types.Message,
    ):
        choices = []
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content=response.content[0].text,
            ),
            logprobs=None,
            finish_reason="stop",
        )
        choices.append(choice)
        return ChatCompletionResponse(
            id=response.id,
            model=response.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
        )

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
        errors=(anthropic.RateLimitError,),
    )
    async def create_chat_completion(
        self,
        req: ChatCompletionRequest,
        stream: Literal[False] | None | Literal[True],
    ):
        auth_token = self.auth_tokens.get(self.token_name)
        if not auth_token:
            raise ValueError(f"Authentication token '{self.token_name}' is required.")

        if not req.max_completion_tokens:
            req.max_completion_tokens = 4096
        if req.temperature:
            req.temperature = req.temperature / 2.0

        client = anthropic.AsyncAnthropic(
            api_key=auth_token,
        )

        system, contents = self.transform_messages(req.messages)

        if stream:

            async def stream_response():
                chunk_index = 0
                real_model = req.model
                async for chunk in await client.messages.create(
                    model=req.model,
                    messages=contents,
                    system=system if system else anthropic.NOT_GIVEN,
                    max_tokens=req.max_completion_tokens,
                    stop_sequences=req.stop if req.stop else anthropic.NOT_GIVEN,
                    temperature=req.temperature if req.temperature else anthropic.NOT_GIVEN,
                    top_p=req.top_p if req.top_p else anthropic.NOT_GIVEN,
                    stream=True,
                ):
                    if chunk.type == "message_start":
                        real_model = cast(
                            anthropic.types.RawMessageStartEvent, chunk
                        ).message.model
                        continue
                    if chunk.type != "content_block_delta":
                        continue
                    yield self.transform_chunk(
                        chunk_index=chunk_index,
                        chunk=chunk,
                        model=real_model,
                    )
                    chunk_index += 1

            return stream_response()

        return self.transform_response(
            response=await client.messages.create(
                model=req.model,
                messages=contents,
                system=system if system else anthropic.NOT_GIVEN,
                max_tokens=req.max_completion_tokens,
                stop_sequences=req.stop if req.stop else anthropic.NOT_GIVEN,
                temperature=req.temperature if req.temperature else anthropic.NOT_GIVEN,
                top_p=req.top_p if req.top_p else anthropic.NOT_GIVEN,
                stream=False,
            )
        )
