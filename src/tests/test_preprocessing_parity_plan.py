import json
import hashlib
import unittest
from pathlib import Path

from src.preprocessing_parity_plan import PreprocessingParityPlanError, validate_preprocessing_parity_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs" / "preprocessing_parity_plan_v0.json"


class PreprocessingParityPlanTests(unittest.TestCase):
    def plan(self):
        return json.loads(PLAN.read_text(encoding="utf-8"))

    def test_current_plan_is_complete_failed_and_not_admitted(self):
        plan = self.plan()
        validate_preprocessing_parity_plan(plan)
        self.assertEqual(plan["status"], "complete")
        self.assertTrue(plan["reference_result_present"])
        self.assertFalse(plan["parity_passed"])
        self.assertFalse(plan["formal_preprocessing_admission_allowed"])
        spec = ROOT / "configs" / "synthetic_preprocessing_clip_v0.json"
        self.assertEqual(hashlib.sha256(spec.read_bytes()).hexdigest(), plan["fixture_spec_sha256"])

    def test_tolerance_cannot_be_relaxed(self):
        plan = self.plan()
        plan["metrics"]["resized_and_crop_tensors"]["fallback_max_abs_diff"] = 1e-4
        with self.assertRaisesRegex(PreprocessingParityPlanError, "frozen ceiling"):
            validate_preprocessing_parity_plan(plan)

    def test_candidate_only_cannot_pass_parity(self):
        plan = self.plan()
        plan["reference_result_present"] = False
        plan["parity_passed"] = True
        with self.assertRaisesRegex(PreprocessingParityPlanError, "both results"):
            validate_preprocessing_parity_plan(plan)


if __name__ == "__main__":
    unittest.main()
