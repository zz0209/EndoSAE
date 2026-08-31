import copy
import json
import unittest
from pathlib import Path

from src.checkpoint_rights_evidence import (
    CheckpointRightsError,
    validate_checkpoint_rights_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = PROJECT_ROOT / "configs" / "checkpoint_rights_evidence_v0.json"


class CheckpointRightsEvidenceTests(unittest.TestCase):
    def record(self):
        return json.loads(RECORD_PATH.read_text(encoding="utf-8"))

    def test_current_user_authorized_internal_use_record_is_valid(self):
        validate_checkpoint_rights_evidence(self.record(), PROJECT_ROOT)

    def test_public_link_cannot_be_treated_as_permission(self):
        record = self.record()
        record["inference_boundary"]["public_download_link_is_permission"] = True
        with self.assertRaisesRegex(CheckpointRightsError, "public_download_link"):
            validate_checkpoint_rights_evidence(record, PROJECT_ROOT)

    def test_internal_use_authorization_cannot_expand_to_deserialization(self):
        record = self.record()
        record["admission"]["deserialization_authorized"] = True
        with self.assertRaisesRegex(CheckpointRightsError, "deserialization_authorized"):
            validate_checkpoint_rights_evidence(record, PROJECT_ROOT)

    def test_internal_use_authorization_cannot_expand_to_redistribution(self):
        record = self.record()
        record["admission"]["derived_asset_release_authorized"] = True
        with self.assertRaisesRegex(CheckpointRightsError, "derived_asset_release_authorized"):
            validate_checkpoint_rights_evidence(record, PROJECT_ROOT)

    def test_authorization_boundary_cannot_claim_upstream_scope(self):
        record = self.record()
        record["project_authorization"][
            "does_not_establish_upstream_weight_license_scope"
        ] = False
        with self.assertRaisesRegex(CheckpointRightsError, "upstream_weight_license"):
            validate_checkpoint_rights_evidence(record, PROJECT_ROOT)

    def test_local_evidence_tampering_is_detected(self):
        record = self.record()
        record["official_evidence"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(CheckpointRightsError, "hash mismatch"):
            validate_checkpoint_rights_evidence(record, PROJECT_ROOT)

    def test_downloaded_checkpoint_hash_is_locked(self):
        record = self.record()
        record["checkpoint"]["local_sha256"] = "0" * 64
        with self.assertRaisesRegex(CheckpointRightsError, "checkpoint hash mismatch"):
            validate_checkpoint_rights_evidence(record, PROJECT_ROOT)

    def test_clarification_scope_cannot_drop_derived_assets(self):
        record = self.record()
        record["required_clarifications"] = [
            q for q in record["required_clarifications"] if "derived activations" not in q
        ]
        with self.assertRaisesRegex(CheckpointRightsError, "derived activations"):
            validate_checkpoint_rights_evidence(record, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
