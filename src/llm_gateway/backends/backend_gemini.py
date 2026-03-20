from collections.abc import AsyncGenerator, Iterable
from typing import Any, Literal, cast, overload

from google import genai
from google.genai import types

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

from .abc import LLMAbstractBaseClass


class GeminiBackend(LLMAbstractBaseClass):
    namespace = "gemini"
    token_name = "GOOGLE_API_KEY"

    def cleanup(self):
        pass

    def transform_messages(
        self,
        messages: Iterable[ChatCompletionMessageParam],
    ) -> tuple[str | None, types.ContentListUnion]:
        system = None
        contents: list[types.Content] = []
        for message in messages:
            if message["role"] in ["system", "developer"]:
                assert system is None, "Only one system message is allowed."
                system = cast(str, message["content"])
                continue
            contents.append(
                types.Content(
                    role=message["role"] if message["role"] == "user" else "model",
                    parts=[
                        types.Part(
                            text=message["content"],
                        )
                    ],
                )
            )
        return system, contents

    def transform_chunk(
        self,
        chunk_index: int,
        chunk: types.GenerateContentResponse,
    ) -> ChatCompletionStreamResponse:
        choices = []
        if chunk.model_version is None:
            chunk.model_version = ""
        if chunk.candidates is None:
            return ChatCompletionStreamResponse(
                model=chunk.model_version,
                choices=choices,
            )
        for i, candidate in enumerate(chunk.candidates):
            choice = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(
                    role="assistant" if chunk_index == 0 else None,
                    content=candidate.content.parts[0].text,
                ),
            )
            if candidate.finish_reason == 1:
                choice.finish_reason = "stop"
            elif candidate.finish_reason == 2:
                choice.finish_reason = "length"
            choices.append(choice)
        return ChatCompletionStreamResponse(
            model=chunk.model_version,
            choices=choices,
        )

    def transform_response(
        self,
        response: types.GenerateContentResponse,
    ) -> ChatCompletionResponse:
        choices = []
        if response.model_version is None:
            response.model_version = ""
        usage = UsageInfo()
        if response.usage_metadata is not None:
            if response.usage_metadata.prompt_token_count is not None:
                usage.prompt_tokens = response.usage_metadata.prompt_token_count
            if response.usage_metadata.candidates_token_count is not None:
                usage.completion_tokens = response.usage_metadata.candidates_token_count
            if response.usage_metadata.total_token_count is not None:
                usage.total_tokens = response.usage_metadata.total_token_count
        if response.candidates is None:
            return ChatCompletionResponse(
                model=response.model_version,
                choices=choices,
                usage=usage,
            )
        for i, candidate in enumerate(response.candidates):
            choice = ChatCompletionResponseChoice(
                index=i,
                message=ChatMessage(
                    role="assistant",
                    content=candidate.content.parts[0].text,
                ),
                logprobs=None,
                finish_reason="stop",
            )
            if candidate.finish_reason == 1:
                choice.finish_reason = "stop"
            elif candidate.finish_reason == 2:
                choice.finish_reason = "length"
            choices.append(choice)
        return ChatCompletionResponse(
            model=response.model_version,
            choices=choices,
            usage=usage,
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

    async def create_chat_completion(
        self,
        req: ChatCompletionRequest,
        stream: Literal[False] | None | Literal[True],
    ):
        auth_token = self.auth_tokens.get(self.token_name)
        if not auth_token:
            raise ValueError(f"Authentication token '{self.token_name}' is required.")

        client = genai.Client(
            api_key=auth_token,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=10,
                    initial_delay=1,
                    exp_base=2,
                    jitter=1,
                    max_delay=60,
                )
            ),
        )

        system, contents = self.transform_messages(req.messages)

        generation_config = types.GenerateContentConfig(
            system_instruction=system,
            stop_sequences=req.stop
            if (isinstance(req.stop, list) or req.stop is None)
            else [req.stop],
            max_output_tokens=req.max_completion_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            presence_penalty=req.presence_penalty,
            frequency_penalty=req.frequency_penalty,
        )

        if stream:

            async def stream_response():
                chunk_index = 0
                async for chunk in await client.aio.models.generate_content_stream(
                    model=cast(str, req.model),
                    contents=contents,
                    config=generation_config,
                ):
                    yield self.transform_chunk(
                        chunk_index=chunk_index,
                        chunk=chunk,
                    )
                    chunk_index += 1

            return stream_response()

        return self.transform_response(
            response=await client.aio.models.generate_content(
                model=cast(str, req.model),
                contents=contents,
                config=generation_config,
            )
        )
