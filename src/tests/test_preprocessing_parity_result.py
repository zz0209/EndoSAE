import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.preprocessing_parity_result import (
    PreprocessingParityResultError,
    compare_preprocessing_reports,
    validate_preprocessing_parity_result,
)


ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "configs" / "probes" / "preprocessing_parity_incomplete_20260831.json"
CANDIDATE = ROOT / "configs" / "probes" / "synthetic_preprocessing_external_probe_20260831.json"


class PreprocessingParityResultTests(unittest.TestCase):
    def record(self):
        return json.loads(RECORD.read_text(encoding="utf-8"))

    def candidate(self):
        return json.loads(CANDIDATE.read_text(encoding="utf-8"))

    def test_current_raw_comparison_fails_frozen_tolerance(self):
        record = self.record()
        validate_preprocessing_parity_result(record, ROOT)
        self.assertEqual(record["status"], "fail")
        self.assertEqual(record["first_divergent_stage"], "resize_tensor")
        self.assertFalse(record["formal_preprocessing_admission_allowed"])

    def test_raw_diff_record_cannot_claim_pass(self):
        record = self.record()
        record["parity_passed"] = True
        with self.assertRaisesRegex(PreprocessingParityResultError, "does not match"):
            validate_preprocessing_parity_result(record, ROOT)

    def test_artifact_hash_is_recomputed(self):
        record = self.record()
        record["candidate_report"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(PreprocessingParityResultError, "hash mismatch"):
            validate_preprocessing_parity_result(record, ROOT)

    def test_raw_diff_hash_is_recomputed(self):
        record = self.record()
        record["raw_diff_report"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(PreprocessingParityResultError, "hash mismatch"):
            validate_preprocessing_parity_result(record, ROOT)

    def test_exact_reference_report_can_pass_comparison(self):
        candidate = self.candidate()
        reference = copy.deepcopy(candidate)
        reference["status"] = "legacy-reference"
        result = compare_preprocessing_reports(candidate, reference)
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["parity_passed"])

    def test_tensor_hash_mismatch_requires_raw_diff(self):
        candidate = self.candidate()
        reference = copy.deepcopy(candidate)
        reference["status"] = "legacy-reference"
        reference["resize"]["tensor_fingerprint"] = "0" * 64
        result = compare_preprocessing_reports(candidate, reference)
        self.assertEqual(result["status"], "needs-raw-diff")
        self.assertFalse(result["parity_passed"])

    def test_geometry_mismatch_is_failure(self):
        candidate = self.candidate()
        reference = copy.deepcopy(candidate)
        reference["status"] = "legacy-reference"
        reference["views"][1]["crop_x"] += 1
        result = compare_preprocessing_reports(candidate, reference)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["first_divergent_stage"], "crop_geometry")


if __name__ == "__main__":
    unittest.main()
