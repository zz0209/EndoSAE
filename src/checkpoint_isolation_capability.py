"""Validate that unavailable isolation cannot silently admit checkpoint loading."""

from __future__ import annotations

from typing import Any, Mapping


class CheckpointIsolationCapabilityError(ValueError):
    pass


def validate_checkpoint_isolation_capability(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != "endosae.checkpoint-isolation-capability.v0":
        raise CheckpointIsolationCapabilityError("unsupported schema")
    if record.get("status") != "blocked-no-usable-isolated-runtime":
        raise CheckpointIsolationCapabilityError("current capability status must remain blocked")
    capabilities = record.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise CheckpointIsolationCapabilityError("capabilities missing")
    usable = any(
        capabilities.get(key) is True
        for key in (
            "docker_command_present", "podman_command_present",
            "windows_sandbox_executable_present", "wsl_distribution_enumeration_succeeded",
        )
    )
    if usable:
        raise CheckpointIsolationCapabilityError("record says blocked despite a nominally usable runtime")
    boundary = record.get("required_boundary")
    if not isinstance(boundary, Mapping) or set(boundary.values()) != {True}:
        raise CheckpointIsolationCapabilityError("required isolation boundary weakened")
    if record.get("static_pickle_audit_available") is not True:
        raise CheckpointIsolationCapabilityError("static audit availability drift")
    if record.get("static_pickle_audit_is_deserialization") is not False:
        raise CheckpointIsolationCapabilityError("static audit cannot be called deserialization")
    if record.get("host_deserialization_allowed") is not False or record.get("checkpoint_deserialization_allowed") is not False:
        raise CheckpointIsolationCapabilityError("unavailable isolation cannot admit loading")
    options = record.get("next_admissible_options")
    if not isinstance(options, list) or len(options) < 4:
        raise CheckpointIsolationCapabilityError("safe alternatives are incomplete")
