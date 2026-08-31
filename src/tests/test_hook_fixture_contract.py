import unittest

from src.hook_fixture_contract import (
    HookFixtureError,
    validate_hook_fixture,
    validate_hook_fixture_v1,
)


class HookFixtureContractTests(unittest.TestCase):
    def valid_record(self):
        return {
            "schema_version": "endosae.hook-fixture.v0",
            "status": "pass",
            "fixture_id": "deterministic-clip-001-block5",
            "model_commit": "206427ebfb77a937ef0cd60370331bcedd74e5a2",
            "checkpoint_sha256": "a" * 64,
            "preprocessing_id": "official-polypdiag-eval-v0",
            "input_asset_sha256": "b" * 64,
            "seed": 0,
            "device": "cpu",
            "dtype": "float32",
            "eval_mode": True,
            "inference_mode": True,
            "hook_target": "model.blocks.5",
            "hook_kind": "read-only-forward-output",
            "hook_call_count": 1,
            "hook_removed_after_run": True,
            "baseline_output_shape": [1, 2],
            "hooked_output_shape": [1, 2],
            "activation_shape": [1, 1569, 768],
            "baseline_output_sha256": "c" * 64,
            "hooked_output_sha256": "c" * 64,
            "input_before_sha256": "d" * 64,
            "input_after_sha256": "d" * 64,
            "atol": 0.0,
            "rtol": 0.0,
            "max_abs_diff": 0.0,
            "allclose_passed": True,
        }

    def test_valid_pass(self):
        validate_hook_fixture(self.valid_record())

    def test_pass_rejects_mutated_input(self):
        record = self.valid_record()
        record["input_after_sha256"] = "e" * 64
        with self.assertRaisesRegex(HookFixtureError, "unchanged input"):
            validate_hook_fixture(record)

    def test_pass_rejects_missing_or_leaked_hook(self):
        for field, value in (("hook_call_count", 0), ("hook_removed_after_run", False)):
            with self.subTest(field=field):
                record = self.valid_record()
                record[field] = value
                with self.assertRaises(HookFixtureError):
                    validate_hook_fixture(record)

    def test_pass_rejects_output_change(self):
        record = self.valid_record()
        record["max_abs_diff"] = 1e-4
        with self.assertRaisesRegex(HookFixtureError, "exceeds"):
            validate_hook_fixture(record)

    def test_failed_fixture_may_preserve_diagnostics(self):
        record = self.valid_record()
        record["status"] = "fail"
        record["allclose_passed"] = False
        record["max_abs_diff"] = 1.0
        validate_hook_fixture(record)

    def valid_v1_record(self):
        record = self.valid_record()
        record.update({
            "schema_version": "endosae.hook-fixture.v1",
            "post_removal_output_shape": [1, 2],
            "post_removal_output_sha256": "c" * 64,
            "post_removal_allclose_passed": True,
            "batch_single_checked": True,
            "batch_single_allclose_passed": True,
            "batch_single_max_abs_diff": 0.0,
            "frame_order_probe_checked": True,
            "reversed_input_sha256": "e" * 64,
            "reversed_output_sha256": "f" * 64,
            "frame_order_output_changed": True,
            "frame_order_max_abs_diff": 0.1,
        })
        return record

    def test_valid_v1_pass(self):
        validate_hook_fixture_v1(self.valid_v1_record())

    def test_v1_pass_requires_post_removal_recovery(self):
        record = self.valid_v1_record()
        record["post_removal_allclose_passed"] = False
        with self.assertRaisesRegex(HookFixtureError, "post-removal"):
            validate_hook_fixture_v1(record)

    def test_v1_pass_requires_batch_equivalence(self):
        record = self.valid_v1_record()
        record["batch_single_allclose_passed"] = False
        with self.assertRaisesRegex(HookFixtureError, "batch-vs-single"):
            validate_hook_fixture_v1(record)

    def test_v1_pass_requires_frame_order_sensitivity(self):
        record = self.valid_v1_record()
        record["frame_order_output_changed"] = False
        with self.assertRaisesRegex(HookFixtureError, "frame-order"):
            validate_hook_fixture_v1(record)


if __name__ == "__main__":
    unittest.main()
