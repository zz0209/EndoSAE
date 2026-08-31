import unittest

from src.checkpoint_inventory import CheckpointInventoryError, validate_checkpoint_inventory


class CheckpointInventoryTests(unittest.TestCase):
    def record(self):
        return {
            "schema_version": "endosae.checkpoint-inventory.v0",
            "status": "pass",
            "checkpoint_sha256": "a" * 64,
            "preflight_report_sha256": "b" * 64,
            "isolated_execution": True,
            "network_disabled": True,
            "sensitive_mounts_absent": True,
            "loader_runtime": "torch-test",
            "loader_api": "torch.load(weights_only=True)",
            "weights_only_requested": True,
            "weights_only_succeeded": True,
            "top_level_keys": ["teacher"],
            "selected_state_dict_path": "teacher",
            "prefix_filter": "backbone.",
            "tensors": [
                {"name": "backbone.cls_token", "shape": [1, 1, 3], "dtype": "float32", "numel": 3}
            ],
            "missing_keys": ["head.weight"],
            "unexpected_keys": [],
            "allowlist": {
                "status": "not-frozen",
                "frozen_before_behavior_testing": False,
                "missing_keys": [],
                "unexpected_keys": [],
            },
            "behavior_testing_allowed": False,
        }

    def test_pass_inventory_can_remain_blocked_before_allowlist(self):
        validate_checkpoint_inventory(self.record())

    def test_pass_requires_isolation(self):
        record = self.record()
        record["network_disabled"] = False
        with self.assertRaisesRegex(CheckpointInventoryError, "isolated successful"):
            validate_checkpoint_inventory(record)

    def test_tensor_numel_must_match_shape(self):
        record = self.record()
        record["tensors"][0]["numel"] = 4
        with self.assertRaisesRegex(CheckpointInventoryError, "numel"):
            validate_checkpoint_inventory(record)

    def test_behavior_requires_prefrozen_exact_allowlist(self):
        record = self.record()
        record["behavior_testing_allowed"] = True
        with self.assertRaisesRegex(CheckpointInventoryError, "pre-frozen"):
            validate_checkpoint_inventory(record)

        record["allowlist"] = {
            "status": "frozen",
            "frozen_before_behavior_testing": True,
            "missing_keys": ["head.weight"],
            "unexpected_keys": [],
        }
        validate_checkpoint_inventory(record)

    def test_broad_allowlist_cannot_hide_unobserved_keys(self):
        record = self.record()
        record["behavior_testing_allowed"] = True
        record["allowlist"] = {
            "status": "frozen",
            "frozen_before_behavior_testing": True,
            "missing_keys": ["head.weight", "arbitrary.extra"],
            "unexpected_keys": [],
        }
        with self.assertRaisesRegex(CheckpointInventoryError, "missing keys disagree"):
            validate_checkpoint_inventory(record)


if __name__ == "__main__":
    unittest.main()
