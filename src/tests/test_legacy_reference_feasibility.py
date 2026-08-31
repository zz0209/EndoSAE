import copy
import json
import unittest
from pathlib import Path

from src.legacy_reference_feasibility import (
    LegacyReferenceFeasibilityError,
    validate_legacy_reference_feasibility,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "legacy_reference_feasibility_v0.json"


class LegacyReferenceFeasibilityTests(unittest.TestCase):
    def record(self):
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_current_audit_records_lr0_without_admitting_g1(self):
        record = self.record()
        validate_legacy_reference_feasibility(record, ROOT)
        self.assertEqual(record["stages"][0]["status"], "installed-reference-preprocessing-parity-failed")
        self.assertFalse(record["formal_activation_cache_allowed"])

    def test_environment_hash_is_recomputed(self):
        record = self.record()
        record["official_environment"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(LegacyReferenceFeasibilityError, "hash mismatch"):
            validate_legacy_reference_feasibility(record, ROOT)

    def test_lr0_cannot_expand_to_decoder_stack(self):
        record = self.record()
        record["stages"][0]["required_packages"].append("av==10.0.0")
        with self.assertRaisesRegex(LegacyReferenceFeasibilityError, "minimal tensor-only"):
            validate_legacy_reference_feasibility(record, ROOT)

    def test_unexecuted_stage_cannot_claim_g1(self):
        record = self.record()
        record["stages"][0]["g1_evidence_allowed"] = True
        with self.assertRaisesRegex(LegacyReferenceFeasibilityError, "cannot admit"):
            validate_legacy_reference_feasibility(record, ROOT)

    def test_authorization_cannot_expand_to_external_environment(self):
        record = self.record()
        record["authorization_packet"]["external_project_environment_mutation_allowed"] = True
        with self.assertRaisesRegex(LegacyReferenceFeasibilityError, "external environments"):
            validate_legacy_reference_feasibility(record, ROOT)


if __name__ == "__main__":
    unittest.main()
