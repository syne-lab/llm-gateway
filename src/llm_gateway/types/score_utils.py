# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# AUTO-GENERATED synthetic stub by types_gen.py

# AUTO-GENERATED synthetic stub for score_utils public types.
from __future__ import annotations

from typing import Any, TypedDict

__all__ = [
    "ScoreContentPartParam",
    "ScoreMultiModalParam",
]


class ScoreContentPartParam(TypedDict, total=False):
    type: str
    text: str


type ScoreMultiModalParam = ScoreContentPartParam | dict[str, Any]
