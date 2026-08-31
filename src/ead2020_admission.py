"""Conservative admission checks for EAD2020 data use."""

from __future__ import annotations

from typing import Any, Mapping


class EAD2020AdmissionError(ValueError):
    """Raised when an EAD admission record exceeds verified rights or grouping."""


def validate_ead2020_admission(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise EAD2020AdmissionError("record must be a mapping")
    if record.get("schema_version") != "endosae.ead2020-admission.v0":
        raise EAD2020AdmissionError("unsupported schema_version")

    dataset = record.get("dataset")
    if not isinstance(dataset, Mapping):
        raise EAD2020AdmissionError("dataset section is required")
    expected = {
        "doi": "10.17632/c7fjbxcgj9.4",
        "version": 4,
        "latest_public_version": 4,
        "license_spdx": "CC-BY-NC-3.0",
        "noncommercial_only": True,
        "data_archive_downloaded": False,
    }
    for field, value in expected.items():
        if dataset.get(field) != value:
            raise EAD2020AdmissionError(f"dataset {field} drift")

    lineage = record.get("version_lineage")
    if not isinstance(lineage, list) or [item.get("version") for item in lineage] != [3, 4]:
        raise EAD2020AdmissionError("version lineage must retain ordered v3 and v4")
    expected_files = {
        3: (501745761, "8300dc96da12aa9df5a7c716cc8147635d31f8552c0882f2c6c4f38e45a0cf0a"),
        4: (504211398, "100885d49c2d047fad7b6a5102512824d1b21c6c3feac8134175665485dc0e5f"),
    }
    for item in lineage:
        if not isinstance(item, Mapping):
            raise EAD2020AdmissionError("version lineage item must be a mapping")
        if (item.get("size_bytes"), item.get("sha256")) != expected_files[item["version"]]:
            raise EAD2020AdmissionError("version file metadata drift")

    metadata = record.get("public_metadata_boundary")
    if not isinstance(metadata, Mapping):
        raise EAD2020AdmissionError("public_metadata_boundary is required")
    if metadata.get("root_file_count_per_version") != 1 or metadata.get("root_asset_type") != "single_zip_archive":
        raise EAD2020AdmissionError("public root asset structure drift")
    for field in (
        "standalone_readme_present",
        "standalone_split_manifest_present",
        "standalone_patient_or_source_video_map_present",
        "archive_members_inspected",
        "snapshot_endpoint_version_pinning_reliable",
    ):
        if metadata.get(field) is not False:
            raise EAD2020AdmissionError(f"unverified public metadata boundary: {field}")

    evidence = record.get("primary_evidence")
    if not isinstance(evidence, list):
        raise EAD2020AdmissionError("primary_evidence must be a list")
    source_ids = {item.get("source_id") for item in evidence if isinstance(item, Mapping)}
    required = {"ead-datacite-v4", "ead-mendeley-v4", "ead-paper", "ead-official-code-repository"}
    if source_ids != required:
        raise EAD2020AdmissionError("primary evidence set drift")
    repo = next(item for item in evidence if item.get("source_id") == "ead-official-code-repository")
    if repo.get("root_license_detected") is not False:
        raise EAD2020AdmissionError("code repository license must remain distinct from dataset license")

    grouping = record.get("grouping_boundary")
    if not isinstance(grouping, Mapping):
        raise EAD2020AdmissionError("grouping_boundary is required")
    for field in (
        "public_patient_id_available",
        "public_source_video_id_available",
        "custom_patient_or_video_grouped_resampling_allowed",
        "continuous_sequence_identity_available",
        "temporal_transport_claims_allowed",
    ):
        if grouping.get(field) is not False:
            raise EAD2020AdmissionError(f"unverified grouping must block {field}")
    if grouping.get("official_split_only") is not True:
        raise EAD2020AdmissionError("EAD must remain official-split-only")

    admission = record.get("admission")
    if not isinstance(admission, Mapping):
        raise EAD2020AdmissionError("admission section is required")
    if not admission.get("license_terms_accepted_by_user_or_institution", False):
        for field in (
            "download_authorized",
            "formal_experiment_use_authorized",
            "derived_asset_release_authorized",
            "g0_ead_pass",
        ):
            if admission.get(field) is not False:
                raise EAD2020AdmissionError(f"unaccepted terms must block {field}")
    if not grouping.get("public_patient_id_available") or not grouping.get("public_source_video_id_available"):
        if admission.get("custom_split_authorized") is not False:
            raise EAD2020AdmissionError("missing grouping IDs must block custom_split_authorized")
    if not grouping.get("continuous_sequence_identity_available"):
        if admission.get("temporal_claim_authorized") is not False:
            raise EAD2020AdmissionError("missing sequence identity must block temporal_claim_authorized")
