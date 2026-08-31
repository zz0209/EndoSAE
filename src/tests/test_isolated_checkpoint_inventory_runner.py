import unittest

import torch

from isolated_checkpoint_inventory_runner import (
    CHECKPOINT_SHA256,
    EXPECTED_TOP_LEVEL,
    POLICY_SHA256,
    IsolatedInventoryError,
    inventory_loaded_object,
    validate_attestation,
)


class IsolatedCheckpointInventoryRunnerTests(unittest.TestCase):
    def attestation(self):
        return {
            "schema_version": "endosae.first-load-isolation-attestation.v0",
            "backend": "windows-sandbox",
            "network_enabled": False,
            "sensitive_mounts_present": False,
            "ephemeral_workspace": True,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "safe_globals_policy_sha256": POLICY_SHA256,
            "loader_environment_sha256": "a" * 64,
            "attestation_id": "fixture",
        }

    def test_valid_attestation(self):
        validate_attestation(self.attestation())

    def test_network_or_sensitive_mount_is_rejected(self):
        for field in ("network_enabled", "sensitive_mounts_present"):
            record = self.attestation()
            record[field] = True
            with self.assertRaises(IsolatedInventoryError):
                validate_attestation(record)

    def test_metadata_only_inventory_and_mapping(self):
        student = {"backbone.weight": torch.zeros(2, 3)}
        teacher = {"module.backbone.weight": torch.ones(2, 3)}
        root = dict.fromkeys(EXPECTED_TOP_LEVEL)
        root.update({"student": student, "teacher": teacher, "optimizer": {}})
        result = inventory_loaded_object(root, torch)
        self.assertEqual(result["recursive_tensor_count"], 2)
        self.assertEqual(result["recursive_tensor_numel"], 12)
        self.assertTrue(result["student_teacher_mapping_summary"]["canonical_sets_equal"])
        self.assertNotIn("value", result["tensor_records"][0])

    def test_top_level_mismatch_stops(self):
        with self.assertRaises(IsolatedInventoryError):
            inventory_loaded_object({"student": {}}, torch)


if __name__ == "__main__":
    unittest.main()
