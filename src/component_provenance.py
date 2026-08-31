"""Validation for the pinned EndoFM component and license boundary registry."""

from __future__ import annotations

from typing import Any, Mapping


class ProvenanceError(ValueError):
    """Raised when component provenance would overstate license clearance."""


REQUIRED_COMPONENT_IDS = {
    "endofm-top-level", "dino", "timesformer", "svt", "transunet",
    "stft", "endofm-checkpoint",
}


def validate_component_registry(registry: Mapping[str, Any]) -> None:
    if not isinstance(registry, Mapping):
        raise ProvenanceError("registry must be a mapping")
    if registry.get("schema_version") != "endosae.component-provenance.v0":
        raise ProvenanceError("unsupported schema_version")
    components = registry.get("components")
    if not isinstance(components, list) or not components:
        raise ProvenanceError("components must be non-empty")
    by_id = {}
    for component in components:
        if not isinstance(component, Mapping):
            raise ProvenanceError("component must be a mapping")
        required = {
            "component_id", "upstream_url", "license_id", "evidence",
            "embedded_paths", "upstream_version_in_endofm",
            "restriction_class", "redistribution_status",
        }
        missing = sorted(required.difference(component))
        if missing:
            raise ProvenanceError(f"missing component fields: {', '.join(missing)}")
        component_id = component["component_id"]
        if not isinstance(component_id, str) or not component_id or component_id in by_id:
            raise ProvenanceError("component_id must be non-empty and unique")
        by_id[component_id] = component
        if component["restriction_class"] in {"noncommercial-or-mixed", "unknown"}:
            if component["redistribution_status"] != "not-cleared":
                raise ProvenanceError(
                    f"{component_id} restriction cannot be marked redistribution-cleared"
                )
    missing_ids = sorted(REQUIRED_COMPONENT_IDS.difference(by_id))
    if missing_ids:
        raise ProvenanceError(f"missing required components: {', '.join(missing_ids)}")
    if registry.get("status") != "audit-incomplete-not-cleared-for-redistribution":
        raise ProvenanceError("registry must remain not-cleared while restrictions are open")
