# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# AUTO-GENERATED synthetic stub by types_gen.py

# AUTO-GENERATED synthetic stub — re-exports the public types that downstream
# protocol files import from vllm.entrypoints.chat_utils.
from __future__ import annotations

import uuid as _uuid
from typing import Any, Literal, Required, TypedDict

# Single unambiguous import so static analysers resolve ChatCompletionMessageParam
# as a direct member of this module.
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
)

__all__ = [
    "ChatCompletionContentPartImageParam",
    "ChatCompletionContentPartParam",
    "ChatCompletionContentPartTextParam",
    "ChatCompletionMessageParam",
    "ChatTemplateContentFormatOption",
    "ConversationMessage",
    "make_tool_call_id",
    "random_tool_call_id",
]


# vLLM extends the openai type with a few extra roles/fields; we keep the
# openai type as the public alias — compatible with all openai clients.
type ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]


class ConversationMessage(TypedDict, total=False):
    role: Required[str]
    content: str | list[dict[str, Any]] | None
    name: str | None
    tool_call_id: str | None


def random_tool_call_id() -> str:
    return f"call_{_uuid.uuid4().hex}"


def make_tool_call_id() -> str:
    return f"call_{_uuid.uuid4().hex}"
