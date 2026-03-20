"""
types_gen.py — vLLM protocol → standalone lightweight type files

Strategy
--------
Multi-pass pipeline:

  1. Fetch source files starting from ENTRY_POINTS, following transitive vllm
     deps under FOLLOW_PREFIXES (BFS).
  2. For each fetched module, run VllmCleaner which:
       - Drops method bodies, logger, torch, vllm.config, vllm.sampling_params.
       - Rewrites followed vllm.* imports → target package namespace.
       - Drops unresolvable imports; collects their names for stub injection.
       - Handles if TYPE_CHECKING / bare if blocks safely.
  3. SYNTHETIC_OVERRIDES: modules too complex to AST-clean (e.g. chat_utils)
     are replaced with hand-crafted minimal re-exports of only the public
     names the protocol files actually need.
  4. Inject inline stubs for any vllm runtime types (SamplingParams etc.)
     that appear in field annotations but have no real dependency.
  5. Emit one file per module; write __init__.py for all directories.

Usage
-----
    python types_gen.py [--commit <sha>] [--out-dir src/llm_gateway/types]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# DEFAULT_COMMIT = "refs/heads/main"
DEFAULT_COMMIT = "0674d1fee76102e05db6ebb415952e51a6cf81a8"
BASE_URL_TPL = "https://raw.githubusercontent.com/vllm-project/vllm/{commit}/"

ENTRY_POINTS = [
    "vllm/entrypoints/openai/chat_completion/protocol.py",
    "vllm/entrypoints/openai/responses/protocol.py",
    "vllm/entrypoints/anthropic/protocol.py",
    "vllm/entrypoints/openai/engine/protocol.py",
    "vllm/entrypoints/chat_utils.py",
    "vllm/entrypoints/score_utils.py",
    # Note: vllm.utils is handled by SYNTHETIC_OVERRIDES, not as a file entry
]

# Follow transitive deps under these vllm prefixes
FOLLOW_PREFIXES = (
    "vllm.entrypoints.openai.engine",
    "vllm.entrypoints.openai.chat_completion",
    "vllm.entrypoints.openai.responses",
    "vllm.entrypoints.anthropic",
    "vllm.logprobs",
)

# Drop these vllm modules entirely (collect their names for stub injection)
DROP_MODULES = {
    "vllm.config",
    "vllm.config.utils",
    "vllm.sampling_params",
    "vllm.exceptions",
    "vllm.logger",
    "vllm.sequence",
    "vllm.pooling_params",
    "vllm.renderers",
    "vllm.logprobs",
}

# Modules we override with a synthetic source instead of AST-cleaning the real one.
# Key = vllm module name. Value = callable(pkg) -> source string.
# These are modules that are too large / too entangled to clean automatically.
def _chat_utils_synthetic(pkg: str) -> str:
    return '''\
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
'''


def _score_utils_synthetic(pkg: str) -> str:
    return '''\
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
'''


def _utils_synthetic(pkg: str) -> str:
    return '''\
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
'''


SYNTHETIC_OVERRIDES: dict[str, Callable] = {
    "vllm.entrypoints.chat_utils":  _chat_utils_synthetic,
    "vllm.entrypoints.score_utils": _score_utils_synthetic,
    "vllm.utils":                   _utils_synthetic,
}

# Inline stubs injected when a name is referenced in annotations but the
# module it came from was dropped.
# Use type aliases to Any for pydantic compatibility
DROP_NAME_STUBS: dict[str, str] = {
    "VLLMValidationError":      "class VLLMValidationError(ValueError): ...",
    "ModelConfig":              "type ModelConfig = Any",
    "SamplingParams":           "type SamplingParams = Any",
    "BeamSearchParams":         "type BeamSearchParams = Any",
    "RepetitionDetectionParams":"type RepetitionDetectionParams = Any",
    "RequestOutputKind":        "type RequestOutputKind = Any",
    "StructuredOutputsParams":  "type StructuredOutputsParams = Any",
    "ChatParams":               "type ChatParams = Any",
    "TokenizeParams":           "type TokenizeParams = Any",
    "replace":                  "def replace(obj, **kw): ...",
    "merge_kwargs":             "def merge_kwargs(*dicts): ...",
    "MultiModalSharedField":    "type MultiModalSharedField = Any",
    "LazyLoader": (
        "class LazyLoader:\n"
        "    def __init__(self, *a, **kw): pass\n"
        "    def __getattr__(self, name): raise ImportError(name)"
    ),
}

_INT64_STUB = """\
_INT64_MIN: int = -(2**63)
_INT64_MAX: int = 2**63 - 1
"""

_OPENAI_BASE_MODEL_STUB = """\
class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    field_names: ClassVar[Optional[set[str]]] = None
"""

_RANDOM_UUID_STUB = """\
import uuid as _uuid

def random_uuid() -> str:
    return _uuid.uuid4().hex

def random_tool_call_id() -> str:
    return f"call_{random_uuid()}"
"""

# ---------------------------------------------------------------------------
# Module path helpers
# ---------------------------------------------------------------------------

def _vllm_mod_to_output(vllm_module: str, pkg: str) -> str:
    """vllm.entrypoints.openai.engine.protocol → pkg.openai.engine.protocol"""
    m = vllm_module
    if m.endswith(".__init__"):
        m = m[: -len(".__init__")]
    for prefix in ("vllm.entrypoints.", "vllm."):
        if m.startswith(prefix):
            m = m[len(prefix):]
            break
    return f"{pkg}.{m}"


def path_to_mod(path: str) -> str:
    m = path.replace("/", ".").removesuffix(".py")
    if m.endswith(".__init__"):
        m = m[: -len(".__init__")]
    return m


def mod_to_path(mod: str) -> str:
    return mod.replace(".", "/") + ".py"


def output_path_for(vllm_path: str, out_dir: str) -> str:
    rel = vllm_path
    for strip in ("vllm/entrypoints/", "vllm/"):
        if rel.startswith(strip):
            rel = rel[len(strip):]
            break
    # Keep __init__.py so packages with submodules stay packages.
    return os.path.join(out_dir, rel)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch_text(url: str) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": "types_gen/2.0"})
        with urlopen(req, timeout=20) as r:
            return r.read().decode()
    except HTTPError as e:
        if e.code == 404:
            return None
        raise

# ---------------------------------------------------------------------------
# Dependency collection pass
# ---------------------------------------------------------------------------

def collect_vllm_imports(source: str) -> set[str]:
    """Return all vllm.* module names imported from *source*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("vllm"):
                mods.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("vllm"):
                    mods.add(alias.name)
    return mods

# ---------------------------------------------------------------------------
# AST Utilities
# ---------------------------------------------------------------------------

class Parentage(ast.NodeTransformer):
    """Add parent references to all AST nodes.

    This allows traversing up the tree from any node, which is useful for
    context-aware transformations.
    """

    def __init__(self) -> None:
        self._parent_stack: list[ast.AST] = []

    def visit(self, node: ast.AST) -> ast.AST:
        """Set parent attribute and traverse."""
        # Set parent for this node (None for the root module)
        node.parent = self._parent_stack[-1] if self._parent_stack else None  # type: ignore[attr-defined]
        # Push this node as the current parent
        self._parent_stack.append(node)
        try:
            result = super().visit(node)
            # Handle the case where super() returns a different node
            if result is not None and result is not node:
                result.parent = node.parent  # type: ignore[attr-defined]
            return result if result is not None else node
        finally:
            self._parent_stack.pop()


def add_parent_refs(tree: ast.Module) -> ast.Module:
    """Add parent references to all nodes in the tree."""
    Parentage().visit(tree)
    return tree


def get_parent_chain(node: ast.AST) -> list[ast.AST]:
    """Get the chain of parents from the node to the root."""
    chain = []
    current = node
    while hasattr(current, "parent") and current.parent is not None:
        current = current.parent
        chain.append(current)
    return chain


def is_inside_class(node: ast.AST) -> bool:
    """Check if a node is inside a class definition."""
    for parent in get_parent_chain(node):
        if isinstance(parent, ast.ClassDef):
            return True
    return False


def is_inside_try_import(node: ast.AST) -> bool:
    """Check if a node is inside a try block that contains imports."""
    for parent in get_parent_chain(node):
        if isinstance(parent, ast.Try):
            # Check if the try block contains any import statements
            for stmt in parent.body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    return True
    return False


def is_inside_type_checking(node: ast.AST) -> bool:
    """Check if a node is inside an `if TYPE_CHECKING:` block."""
    for parent in get_parent_chain(node):
        if isinstance(parent, ast.If):
            test = parent.test
            if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
               (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"):
                return True
    return False


# ---------------------------------------------------------------------------
# The AST cleaner
# ---------------------------------------------------------------------------

class VllmCleaner(ast.NodeTransformer):
    """
    Strips a vllm protocol module down to type-only content:
      - Keeps: ClassDef (fields only), TypeAlias, top-level Assign constants,
               AnnAssign, imports from non-vllm packages.
      - Drops: all method bodies, logger, torch, vllm internal imports.
      - Rewrites: followed vllm.* imports → target package namespace.
      - Drops: if TYPE_CHECKING blocks; collapses empty if/else branches.
    """

    KEEP_FUNCTIONS = frozenset({
        "random_uuid", "random_tool_call_id", "make_tool_call_id",
        "serialize_message", "serialize_messages",
    })

    def __init__(self, output_pkg: str, this_vllm_mod: str, followed_mods: set[str]):
        self.output_pkg = output_pkg
        self.this_vllm_mod = this_vllm_mod
        self.followed_mods = followed_mods
        self._stubs_needed: set[str] = set()

    # ------------------------------------------------------------------ module

    def visit_Module(self, node: ast.Module) -> ast.Module:
        new_body = []
        for stmt in node.body:
            result = self.visit(stmt)
            if result is None:
                continue
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
        node.body = new_body or [ast.Pass()]  # type: ignore[assignment]
        return node

    # ------------------------------------------------------------------ imports

    def visit_Import(self, node: ast.Import) -> ast.stmt | None:
        kept = [a for a in node.names
                if not a.name.startswith("vllm") and a.name != "torch"]
        if not kept:
            return None
        # Drop packages we can't import in this env but keep known-good ones
        safe = []
        for alias in kept:
            try:
                importlib.import_module(alias.name)
                safe.append(alias)
            except ImportError:
                pass
        if not safe:
            return None
        node.names = safe
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.stmt | None:
        mod = node.module or ""

        if not mod.startswith("vllm"):
            # Non-vllm import: keep if importable or from a known-safe namespace
            return self._keep_external_import(node)

        # ---- vllm import ----
        names = [a.name for a in node.names]

        # Is this a module we follow (i.e. we generate an output file for it)?
        if mod in self.followed_mods:
            node.module = _vllm_mod_to_output(mod, self.output_pkg)
            return node

        # Is it a synthetic override module?
        if mod in SYNTHETIC_OVERRIDES:
            node.module = _vllm_mod_to_output(mod, self.output_pkg)
            return node

        # Drop module — collect names for potential stub injection
        self._stubs_needed.update(names)
        return None

    def _keep_external_import(self, node: ast.ImportFrom) -> ast.stmt | None:
        mod = node.module or ""
        always_keep = {
            "typing", "typing_extensions", "pydantic", "openai",
            "enum", "abc", "dataclasses", "collections", "functools",
            "json", "time", "uuid", "re", "os", "sys", "io",
            "openai.types", "openai_harmony",
            "__future__",
        }
        if any(mod == k or mod.startswith(k + ".") for k in always_keep):
            return node
        try:
            importlib.import_module(mod)
            return node
        except ImportError:
            return None

    # ------------------------------------------------------------------ statements

    def visit_Expr(self, node: ast.Expr) -> ast.stmt | None:
        return node  # preserve docstrings

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | None:
        if isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("logger", "_LONG_INFO"):
                return None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt | None:
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.stmt | None:
        if node.name in self.KEEP_FUNCTIONS:
            return node
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.stmt | None:
        if node.name in self.KEEP_FUNCTIONS:
            return node
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # drop all methods
            elif isinstance(stmt, ast.AnnAssign):
                new_body.append(stmt)
            elif isinstance(stmt, ast.Assign):
                new_body.append(stmt)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                new_body.append(stmt)  # docstring
            elif isinstance(stmt, ast.ClassDef):
                new_body.append(self.visit(stmt))
            elif isinstance(stmt, ast.Pass):
                new_body.append(stmt)
        node.body = new_body or [ast.Pass()]  # type: ignore[assignment]
        return node

    def visit_If(self, node: ast.If) -> ast.stmt | None:
        # Drop if TYPE_CHECKING: entirely - use parent tracking for case nested blocks
        if is_inside_type_checking(node):
            return None
        # Also check direct TYPE_CHECKING test
        test = node.test
        if (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or \
           (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"):
            return None
        # Visit branches; drop if both become empty
        new_body = [s for s in map(self.visit, node.body) if s is not None]
        new_orelse = [s for s in map(self.visit, node.orelse) if s is not None]
        if not new_body and not new_orelse:
            return None
        node.body = new_body or [ast.Pass()]  # type: ignore[assignment]
        node.orelse = new_orelse
        return node


def _collect_used_names(tree: ast.Module) -> set[str]:
    """Collect all names that are used in the AST (for import pruning).

    Uses parent tracking to skip function/method bodies since we only care
    about names used in type annotations and class definitions.
    """
    used: set[str] = set()

    class NameCollector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            used.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # Collect the root name for attribute access like Foo.bar.baz
            if isinstance(node.value, ast.Name):
                used.add(node.value.id)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            # Handle subscripts like list[Foo], dict[str, Any]
            if isinstance(node.value, ast.Name):
                used.add(node.value.id)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # Handle calls like Field(...), BaseModel(...)
            if isinstance(node.func, ast.Name):
                used.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    used.add(node.func.value.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Only collect function name and decorators, skip body for kept functions
            used.add(node.name)
            for dec in node.decorator_list:
                self.visit(dec)
            # Visit annotations but not body
            for arg in node.args.args:
                if arg.annotation:
                    self.visit(arg.annotation)
            if node.returns:
                self.visit(node.returns)
            # Skip body - we don't need to track names inside function bodies

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            # Same as FunctionDef
            used.add(node.name)
            for dec in node.decorator_list:
                self.visit(dec)
            for arg in node.args.args:
                if arg.annotation:
                    self.visit(arg.annotation)
            if node.returns:
                self.visit(node.returns)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            used.add(node.name)
            for dec in node.decorator_list:
                self.visit(dec)
            for base in node.bases:
                self.visit(base)
            for kw in node.keywords:
                self.visit(kw.value)
            # Continue into class body for field annotations
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name):
                used.add(node.target.id)
            self.visit(node.annotation)
            if node.value:
                self.visit(node.value)

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    used.add(target.id)
                else:
                    self.visit(target)
            # Always visit the value to collect used names like ConfigDict
            self.visit(node.value)

    collector = NameCollector()
    collector.visit(tree)
    return used


def _prune_unused_imports(tree: ast.Module) -> ast.Module:
    """Remove imports that aren't used in the code."""
    used_names = _collect_used_names(tree)

    new_body: list[ast.stmt] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            # Keep only used imports
            kept = [a for a in stmt.names if a.asname in used_names or a.name.split(".")[0] in used_names]
            if kept:
                stmt.names = kept
                new_body.append(stmt)
        elif isinstance(stmt, ast.ImportFrom):
            # Keep only used imports from
            kept = [a for a in stmt.names if a.asname in used_names or a.name in used_names]
            if kept:
                stmt.names = kept
                new_body.append(stmt)
        else:
            new_body.append(stmt)

    tree.body = new_body
    return tree


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class GeneratedModule:
    vllm_path: str
    vllm_mod: str
    output_path: str
    source: str = ""
    transformed: str = ""


def build_graph(entry_paths: list[str], base_url: str) -> dict[str, GeneratedModule]:
    """BFS over vllm source — returns map of vllm_path → GeneratedModule."""
    queue = list(entry_paths)
    visited: set[str] = set()
    graph: dict[str, GeneratedModule] = {}

    # Pre-seed synthetic overrides so the graph knows about them
    for syn_mod in SYNTHETIC_OVERRIDES:
        syn_path = mod_to_path(syn_mod)
        visited.add(syn_path)
        vmod = path_to_mod(syn_path)
        graph[syn_path] = GeneratedModule(
            vllm_path=syn_path, vllm_mod=vmod, output_path="", source="<synthetic>"
        )

    while queue:
        vpath = queue.pop(0)
        if vpath in visited:
            continue
        visited.add(vpath)

        url = base_url + vpath
        print(f"  Fetching {url} ...", end=" ", flush=True)
        source = fetch_text(url)
        if source is None:
            print("NOT FOUND — skipping")
            continue
        print(f"{len(source)} bytes")

        vmod = path_to_mod(vpath)
        graph[vpath] = GeneratedModule(vllm_path=vpath, vllm_mod=vmod,
                                       source=source, output_path="")

        for dep_mod in collect_vllm_imports(source):
            # Follow only explicit FOLLOW_PREFIXES
            if any(dep_mod == p or dep_mod.startswith(p + ".")
                   for p in FOLLOW_PREFIXES):
                dep_path = mod_to_path(dep_mod)
                if dep_path not in visited:
                    queue.append(dep_path)
            # Synthetic modules are already seeded — no need to queue

    return graph


def _split_imports_and_body(code: str) -> tuple[str, str]:
    """Split code into import section and body section.

    Returns (imports_section, body_section) where imports_section contains
    all import statements, docstrings, and comments at the top, and body_section
    contains the rest.
    """
    lines = code.split("\n")
    import_end_idx = 0
    in_try_block = False
    try_block_depth = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track try/except blocks (often used for conditional imports)
        if stripped.startswith("try:"):
            in_try_block = True
            try_block_depth = 1
            import_end_idx = i + 1
            continue
        elif in_try_block:
            if stripped.startswith(("except", "else:", "finally:")):
                try_block_depth -= 1
                if try_block_depth <= 0:
                    in_try_block = False
                import_end_idx = i + 1
                continue
            elif stripped and not stripped.startswith(("import ", "from ", "#", "@")):
                # Non-import statement inside try block means imports are done
                break

        # Skip empty lines, comments, and docstrings at the top
        if not stripped or stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            import_end_idx = i + 1
            continue

        # Check if this is an import statement
        if stripped.startswith(("import ", "from ", "import\t")) or (
            stripped.startswith("from ") and " import " in stripped
        ):
            import_end_idx = i + 1
            continue

        # If we hit a non-import statement, we're done with imports
        break

    imports_section = "\n".join(lines[:import_end_idx])
    body_section = "\n".join(lines[import_end_idx:])

    return imports_section, body_section


def transform_graph(graph: dict[str, GeneratedModule], out_dir: str, pkg: str) -> None:
    followed_mods = {gm.vllm_mod for gm in graph.values()}

    for vpath, gm in graph.items():
        gm.output_path = output_path_for(vpath, out_dir)

        # ---- Synthetic override: emit hand-crafted source directly ----
        if gm.source == "<synthetic>":
            fn = SYNTHETIC_OVERRIDES[gm.vllm_mod]
            gm.transformed = (
                "# SPDX-License-Identifier: Apache-2.0\n"
                "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
                "# AUTO-GENERATED synthetic stub by types_gen.py\n\n"
                + fn(pkg)
            )
            continue

        # ---- AST clean ----
        try:
            tree = ast.parse(gm.source)
        except SyntaxError as e:
            print(f"  WARN: syntax error in {vpath}: {e}")
            continue

        # Add parent references for context-aware transformations
        add_parent_refs(tree)

        cleaner = VllmCleaner(output_pkg=pkg, this_vllm_mod=gm.vllm_mod,
                              followed_mods=followed_mods)
        new_tree = cleaner.visit(tree)
        ast.fix_missing_locations(new_tree)

        # Prune unused imports to avoid ruff removing them later
        new_tree = _prune_unused_imports(new_tree)
        ast.fix_missing_locations(new_tree)

        code = ast.unparse(new_tree)

        # ---- Post-process: inject helpers / stubs AFTER imports ----

        # Collect all stubs that need to be injected
        stubs_to_inject: list[str] = []

        # Stubs for dropped vllm names — check against *defined and imported* names in AST
        try:
            _defined: set[str] = set()
            for n in ast.walk(new_tree):
                if isinstance(n, ast.ClassDef):
                    _defined.add(n.name)
                elif isinstance(n, ast.FunctionDef):
                    _defined.add(n.name)
                elif isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            _defined.add(t.id)
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    _defined.add(n.target.id)
                elif isinstance(n, ast.ImportFrom):
                    _defined.update(a.asname or a.name for a in n.names)
        except Exception:
            _defined = set()

        # Collect stub lines for dropped vllm names
        for name in sorted(cleaner._stubs_needed):
            if name in DROP_NAME_STUBS and name not in _defined:
                stubs_to_inject.append(DROP_NAME_STUBS[name])

        # Add INT64 constants if needed
        if "_INT64_MIN" in gm.source or "_INT64_MAX" in gm.source:
            if "_INT64_MIN" not in _defined:
                stubs_to_inject.append(_INT64_STUB.strip())

        # Add OpenAIBaseModel stub if needed
        needs_openai_base_model = False
        if "OpenAIBaseModel" in code and "class OpenAIBaseModel" not in code:
            if f"{pkg}.openai.engine.protocol" not in code:
                stubs_to_inject.append(_OPENAI_BASE_MODEL_STUB.strip())
                needs_openai_base_model = True

        # Add random_uuid stub if needed
        if "random_uuid" in code and "random_uuid" not in _defined:
            stubs_to_inject.append(_RANDOM_UUID_STUB.strip())

        # Split code into imports and body
        imports_section, body_section = _split_imports_and_body(code)

        # Add required imports for OpenAIBaseModel stub if needed
        if needs_openai_base_model:
            # Add ConfigDict to pydantic imports if not present
            if "ConfigDict" not in imports_section:
                # Just add a new import line - simpler and more reliable
                imports_section = imports_section.rstrip() + "\nfrom pydantic import ConfigDict\n"

            # Add ClassVar, Optional to typing imports if not present
            if "ClassVar" not in imports_section:
                imports_section = imports_section.rstrip() + "\nfrom typing import ClassVar\n"
            if "Optional" not in imports_section:
                imports_section = imports_section.rstrip() + "\nfrom typing import Optional\n"

        # Build the final code with stubs inserted AFTER imports
        parts = [imports_section]

        if stubs_to_inject:
            parts.append("")  # Empty line before stubs
            parts.append("\n\n".join(stubs_to_inject))

        if body_section.strip():
            parts.append("")  # Empty line before body
            parts.append(body_section)

        code = "\n".join(parts)

        # Fix any _LONG_INFO torch leftovers
        code = re.sub(
            r"_LONG_INFO\s*=\s*[^\n]+",
            (
                "class _LongInfo:\n"
                "    min: int = _INT64_MIN\n"
                "    max: int = _INT64_MAX\n"
                "_LONG_INFO = _LongInfo()"
            ),
            code,
        )

        gm.transformed = (
            "# SPDX-License-Identifier: Apache-2.0\n"
            "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
            "# AUTO-GENERATED by types_gen.py — do not edit manually\n\n"
            + code + "\n"
        )


def write_outputs(graph: dict[str, GeneratedModule], out_dir: str) -> None:
    for gm in graph.values():
        if not gm.transformed:
            continue
        os.makedirs(os.path.dirname(gm.output_path), exist_ok=True)
        with open(gm.output_path, "w") as f:
            f.write(gm.transformed)
        print(f"  Wrote {gm.output_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--out-dir", default="src/llm_gateway/types")
    parser.add_argument("--pkg", default="llm_gateway.types")
    parser.add_argument("--extra-file", action="append", default=[],
                        dest="extra_files", metavar="VLLM_PATH")
    args = parser.parse_args()

    base_url = BASE_URL_TPL.format(commit=args.commit)
    all_entries = ENTRY_POINTS + args.extra_files

    print(f"\n=== Fetching vllm sources from {base_url} ===\n")
    graph = build_graph(all_entries, base_url)

    print(f"\n=== Transforming {len(graph)} modules ===\n")
    transform_graph(graph, args.out_dir, args.pkg)

    print("\n=== Writing output files ===\n")
    write_outputs(graph, args.out_dir)

    print(
        f"\nDone. Run:\n"
        f"  ruff check --fix {args.out_dir}\n"
        f"  ruff format {args.out_dir}\n"
    )


if __name__ == "__main__":
    main()
