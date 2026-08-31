import copy
import json
import unittest
from pathlib import Path

from src.environment_plan import EnvironmentPlanError, validate_environment_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs" / "endofm_environment_plan_v0.json"


class EnvironmentPlanTests(unittest.TestCase):
    def plan(self):
        return json.loads(PLAN.read_text(encoding="utf-8"))

    def test_current_plan_preserves_blocked_boundary(self):
        plan = self.plan()
        validate_environment_plan(plan)
        self.assertFalse(plan["formal_activation_cache_allowed"])

    def test_reference_pins_cannot_drift_silently(self):
        plan = self.plan()
        plan["reference_environment"]["packages"]["torch"] = "2.8.0"
        with self.assertRaisesRegex(EnvironmentPlanError, "official environment"):
            validate_environment_plan(plan)

    def test_external_probe_cannot_load_checkpoint(self):
        plan = self.plan()
        plan["modern_environment"]["checkpoint_loading_allowed"] = True
        with self.assertRaisesRegex(EnvironmentPlanError, "passed reference"):
            validate_environment_plan(plan)

    def test_formal_cache_requires_complete_parity(self):
        plan = self.plan()
        plan["formal_activation_cache_allowed"] = True
        with self.assertRaisesRegex(EnvironmentPlanError, "complete two-environment"):
            validate_environment_plan(plan)


if __name__ == "__main__":
    unittest.main()
