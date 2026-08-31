import json
import unittest
from pathlib import Path

from src.checkpoint_isolation_capability import (
    CheckpointIsolationCapabilityError,
    validate_checkpoint_isolation_capability,
)


ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "configs" / "checkpoint_isolation_capability_v0.json"


class CheckpointIsolationCapabilityTests(unittest.TestCase):
    def record(self):
        return json.loads(RECORD.read_text(encoding="utf-8"))

    def test_current_record_blocks_loading(self):
        validate_checkpoint_isolation_capability(self.record())

    def test_static_audit_cannot_be_promoted_to_deserialization(self):
        record = self.record()
        record["static_pickle_audit_is_deserialization"] = True
        with self.assertRaises(CheckpointIsolationCapabilityError):
            validate_checkpoint_isolation_capability(record)

    def test_host_loading_cannot_be_enabled(self):
        record = self.record()
        record["host_deserialization_allowed"] = True
        with self.assertRaises(CheckpointIsolationCapabilityError):
            validate_checkpoint_isolation_capability(record)

    def test_boundary_cannot_drop_network_isolation(self):
        record = self.record()
        record["required_boundary"]["network_disabled"] = False
        with self.assertRaises(CheckpointIsolationCapabilityError):
            validate_checkpoint_isolation_capability(record)


if __name__ == "__main__":
    unittest.main()
