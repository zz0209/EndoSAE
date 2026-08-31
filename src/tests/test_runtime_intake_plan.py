import json
import unittest
from pathlib import Path

from src.runtime_intake_plan import RuntimeIntakePlanError, validate_runtime_intake_plan


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs" / "endofm_runtime_intake_plan_v0.json"


class RuntimeIntakePlanTests(unittest.TestCase):
    def plan(self):
        return json.loads(PLAN.read_text(encoding="utf-8"))

    def test_current_plan_preserves_stop_boundary(self):
        validate_runtime_intake_plan(self.plan())

    def test_blocked_unresolved_rights_cannot_authorize_download(self):
        plan = self.plan()
        plan["checkpoint"]["rights_status"] = "blocked-unresolved"
        plan["checkpoint"]["download_authorized"] = True
        with self.assertRaisesRegex(RuntimeIntakePlanError, "block download"):
            validate_runtime_intake_plan(plan)

    def test_internal_use_authorization_cannot_admit_deserialization(self):
        plan = self.plan()
        plan["checkpoint"]["deserialization_authorized"] = True
        with self.assertRaisesRegex(RuntimeIntakePlanError, "cannot admit deserialization"):
            validate_runtime_intake_plan(plan)

    def test_no_isolation_cannot_enable_host_deserialization(self):
        plan = self.plan()
        plan["security_boundary"]["host_deserialization_allowed"] = True
        with self.assertRaisesRegex(RuntimeIntakePlanError, "block host"):
            validate_runtime_intake_plan(plan)

    def test_rights_evidence_hash_cannot_drift(self):
        plan = self.plan()
        plan["checkpoint"]["rights_evidence_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeIntakePlanError, "rights evidence hash mismatch"):
            validate_runtime_intake_plan(plan)


if __name__ == "__main__":
    unittest.main()
