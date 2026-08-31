"""Static audit of pinned EndoFM checkpoint-loading behavior.

This parses source without importing torch or deserializing a checkpoint. It
records why the official fine-tuning script cannot by itself establish a safe,
strict, CPU reference load.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointLoadAuditResult:
    eval_uses_torch_load: bool
    eval_loads_on_cpu: bool
    eval_filters_backbone_prefix: bool
    eval_uses_non_strict_state_dict: bool
    eval_calls_cuda_unconditionally: bool
    eval_wraps_ddp: bool
    helper_uses_non_strict_state_dict: bool
    helper_has_bare_checkpoint_fallback: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def audit_checkpoint_loading(eval_path: str | Path, helper_path: str | Path) -> CheckpointLoadAuditResult:
    eval_tree = ast.parse(Path(eval_path).read_text(encoding="utf-8"), filename=str(eval_path))
    helper_tree = ast.parse(Path(helper_path).read_text(encoding="utf-8"), filename=str(helper_path))
    return CheckpointLoadAuditResult(
        eval_uses_torch_load=_has_call(eval_tree, ("torch", "load")),
        eval_loads_on_cpu=_call_has_keyword_value(eval_tree, ("torch", "load"), "map_location", "cpu"),
        eval_filters_backbone_prefix=_has_startswith_literal(eval_tree, "backbone."),
        eval_uses_non_strict_state_dict=_has_non_strict_load(eval_tree),
        eval_calls_cuda_unconditionally=_has_zero_arg_method_call(eval_tree, "cuda"),
        eval_wraps_ddp=_has_call(eval_tree, ("nn", "parallel", "DistributedDataParallel")),
        helper_uses_non_strict_state_dict=_has_non_strict_load(helper_tree),
        helper_has_bare_checkpoint_fallback=_function_has_bare_except(helper_tree, "load_pretrained"),
    )


def _attribute_parts(node: ast.AST) -> tuple[str, ...]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _has_call(tree: ast.AST, parts: tuple[str, ...]) -> bool:
    return any(
        isinstance(node, ast.Call) and _attribute_parts(node.func) == parts
        for node in ast.walk(tree)
    )


def _call_has_keyword_value(
    tree: ast.AST, parts: tuple[str, ...], keyword_name: str, expected: str
) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _attribute_parts(node.func) != parts:
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == keyword_name
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == expected
            ):
                return True
    return False


def _has_startswith_literal(tree: ast.AST, prefix: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "startswith"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == prefix
        for node in ast.walk(tree)
    )


def _has_non_strict_load(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "load_state_dict":
            continue
        if any(
            keyword.arg == "strict"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        ):
            return True
    return False


def _has_zero_arg_method_call(tree: ast.AST, method: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and not node.args
        and not node.keywords
        for node in ast.walk(tree)
    )


def _function_has_bare_except(tree: ast.Module, function_name: str) -> bool:
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(functions) != 1:
        raise ValueError(f"expected exactly one {function_name}, found {len(functions)}")
    return any(isinstance(node, ast.ExceptHandler) and node.type is None for node in ast.walk(functions[0]))
