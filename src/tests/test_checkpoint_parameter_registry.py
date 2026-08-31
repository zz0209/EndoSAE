import json
import unittest
from pathlib import Path

from checkpoint_parameter_registry import (
    PROJECT_ROOT,
    CheckpointParameterRegistryError,
    scan_parameter_names,
    validate_registry_record,
)


class CheckpointParameterRegistryTests(unittest.TestCase):
    def setUp(self):
        self.path = PROJECT_ROOT / "configs/probes/checkpoint_parameter_name_registry_static_20260831.json"
        self.record = json.loads(self.path.read_text(encoding="utf-8"))

    def test_frozen_record_and_live_rescan(self):
        validate_registry_record(self.record)
        self.assertEqual(scan_parameter_names()["direct_count"], 255)

    def test_rejects_loaded_schema_claim(self):
        changed = dict(self.record)
        changed["loaded_schema_claimed"] = True
        with self.assertRaises(CheckpointParameterRegistryError):
            validate_registry_record(changed, rescan=False)

    def test_rejects_count_drift(self):
        changed = json.loads(json.dumps(self.record))
        changed["observed"]["direct_count"] = 254
        with self.assertRaises(CheckpointParameterRegistryError):
            validate_registry_record(changed, rescan=False)


if __name__ == "__main__":
    unittest.main()

