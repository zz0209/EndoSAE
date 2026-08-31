"""Static audit of the pinned EndoFM TimeSformer source.

The audit deliberately parses Python source without importing EndoFM or torch.
It protects early G1 work from silent upstream/source drift, but it is not a
runtime or checkpoint validation.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPECTED_DEFAULTS = {
    "img_size": 224,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "num_frames": 8,
    "attention_type": "divided_space_time",
}


@dataclass(frozen=True)
class SourceAuditResult:
    source_path: str
    defaults: dict[str, Any]
    defaults_match: bool
    has_forward_features: bool
    has_get_intermediate_layers: bool
    intermediate_returns_single_final_tensor: bool
    has_get_last_selfattention: bool
    has_prepare_tokens: bool
    last_attention_calls_missing_prepare_tokens: bool
    attention_accepts_mask: bool
    block_accepts_mask: bool
    forward_features_accepts_mask: bool
    patch_embed_expects_bcthw: bool
    patch_tokens_are_hwt_ordered: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_timesformer_source(source_path: str | Path) -> SourceAuditResult:
    """Audit architecture defaults and known helper-method contracts."""
    path = Path(source_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    vision = _find_class(tree, "VisionTransformer")
    methods = {
        node.name: node
        for node in vision.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    init = methods.get("__init__")
    if init is None:
        raise ValueError("VisionTransformer.__init__ not found")
    defaults = _function_defaults(init)

    intermediate = methods.get("get_intermediate_layers")
    attention = methods.get("get_last_selfattention")
    return SourceAuditResult(
        source_path=str(path),
        defaults={key: defaults.get(key) for key in EXPECTED_DEFAULTS},
        defaults_match=all(defaults.get(k) == v for k, v in EXPECTED_DEFAULTS.items()),
        has_forward_features="forward_features" in methods,
        has_get_intermediate_layers=intermediate is not None,
        intermediate_returns_single_final_tensor=(
            intermediate is not None and _is_single_final_tensor_helper(intermediate)
        ),
        has_get_last_selfattention=attention is not None,
        has_prepare_tokens="prepare_tokens" in methods,
        last_attention_calls_missing_prepare_tokens=(
            attention is not None
            and _calls_attribute(attention, "self", "prepare_tokens")
            and "prepare_tokens" not in methods
        ),
        attention_accepts_mask=_accepts_mask(_find_method(_find_class(tree, "Attention"), "forward")),
        block_accepts_mask=_accepts_mask(_find_method(_find_class(tree, "Block"), "forward")),
        forward_features_accepts_mask=_accepts_mask(methods["forward_features"]),
        patch_embed_expects_bcthw=_contains_assignment_pattern(
            _find_method(_find_class(tree, "PatchEmbed"), "forward"),
            ("B", "C", "T", "H", "W"),
        ),
        patch_tokens_are_hwt_ordered=(
            _contains_string_literal(methods["forward_features"], "(b n) t m -> b (n t) m")
            and _contains_string_literal(
                _find_method(_find_class(tree, "Block"), "forward"),
                "(b h w) t m -> b (h w t) m",
            )
        ),
    )


def _find_class(tree: ast.Module, class_name: str) -> ast.ClassDef:
    matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {class_name}, found {len(matches)}")
    return matches[0]


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {class_node.name}.{name}, found {len(matches)}")
    return matches[0]


def _accepts_mask(function: ast.FunctionDef) -> bool:
    names = {argument.arg.lower() for argument in [*function.args.posonlyargs, *function.args.args]}
    names.update(argument.arg.lower() for argument in function.args.kwonlyargs)
    return any("mask" in name or "padding" in name for name in names)


def _contains_assignment_pattern(function: ast.FunctionDef, names: tuple[str, ...]) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Tuple):
            continue
        observed = tuple(
            element.id if isinstance(element, ast.Name) else "" for element in node.targets[0].elts
        )
        if observed == names:
            return True
    return False


def _contains_string_literal(function: ast.FunctionDef, value: str) -> bool:
    return any(
        isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == value
        for node in ast.walk(function)
    )


def _function_defaults(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    positional = [*function.args.posonlyargs, *function.args.args]
    padded = [None] * (len(positional) - len(function.args.defaults)) + list(
        function.args.defaults
    )
    values: dict[str, Any] = {}
    for argument, default in zip(positional, padded):
        if default is not None:
            try:
                values[argument.arg] = ast.literal_eval(default)
            except (ValueError, TypeError):
                values[argument.arg] = None
    return values


def _is_single_final_tensor_helper(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    calls_forward_all = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "forward_features"
        and any(
            keyword.arg == "get_all"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
        for node in ast.walk(function)
    )
    returns_single_x = any(
        isinstance(node, ast.Return)
        and isinstance(node.value, (ast.List, ast.Tuple))
        and len(node.value.elts) == 1
        and isinstance(node.value.elts[0], ast.Name)
        and node.value.elts[0].id == "x"
        for node in ast.walk(function)
    )
    return calls_forward_all and returns_single_x


def _calls_attribute(
    function: ast.FunctionDef | ast.AsyncFunctionDef, owner: str, attribute: str
) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
        and node.func.attr == attribute
        for node in ast.walk(function)
    )
