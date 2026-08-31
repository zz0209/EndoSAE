import json
import unittest
from pathlib import Path

from src.model_port_parity_plan import ModelPortParityPlanError, validate_model_port_parity_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "configs" / "model_port_parity_plan_v0.json"


class ModelPortParityPlanTests(unittest.TestCase):
    def plan(self):
        return json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    def test_current_plan_is_valid_and_blocked(self):
        validate_model_port_parity_plan(self.plan(), ROOT)

    def test_posthoc_tolerance_relaxation_is_rejected(self):
        plan = self.plan()
        plan["frozen_numeric_criteria"]["max_abs_error"] = 1e-3
        with self.assertRaises(ModelPortParityPlanError):
            validate_model_port_parity_plan(plan, ROOT)

    def test_missing_middle_stage_is_rejected(self):
        plan = self.plan()
        plan["stages"].pop(2)
        with self.assertRaises(ModelPortParityPlanError):
            validate_model_port_parity_plan(plan, ROOT)

    def test_plan_cannot_authorize_deserialization(self):
        plan = self.plan()
        plan["checkpoint_deserialization_allowed"] = True
        with self.assertRaises(ModelPortParityPlanError):
            validate_model_port_parity_plan(plan, ROOT)


if __name__ == "__main__":
    unittest.main()
