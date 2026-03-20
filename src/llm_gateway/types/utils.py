# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# AUTO-GENERATED synthetic stub by types_gen.py

# AUTO-GENERATED synthetic stub — only the helpers used by protocol files.
from __future__ import annotations

import uuid as _uuid
from typing import Any

__all__ = [
    "random_tool_call_id",
    "random_uuid",
    "resolve_obj_by_qualname",
]


def random_uuid() -> str:
    return _uuid.uuid4().hex


def random_tool_call_id() -> str:
    return f"call_{random_uuid()}"


def resolve_obj_by_qualname(qualname: str) -> Any:
    parts = qualname.rsplit(".", 1)
    if len(parts) == 1:
        import builtins

        return getattr(builtins, parts[0])
    import importlib as _il

    mod = _il.import_module(parts[0])
    return getattr(mod, parts[1])
