import json
import unittest
from pathlib import Path

from src.checkpoint_safe_globals_policy import (
    CheckpointSafeGlobalsPolicyError,
    validate_checkpoint_safe_globals_policy,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "configs" / "checkpoint_safe_globals_policy_v0.json"


class CheckpointSafeGlobalsPolicyTests(unittest.TestCase):
    def policy(self):
        return json.loads(POLICY.read_text(encoding="utf-8"))

    def test_current_policy_is_frozen_and_blocked(self):
        validate_checkpoint_safe_globals_policy(self.policy())

    def test_extra_global_is_rejected(self):
        policy = self.policy()
        policy["exact_allowlist"].append({"symbol": "builtins.eval", "reason": "bad"})
        with self.assertRaises(CheckpointSafeGlobalsPolicyError):
            validate_checkpoint_safe_globals_policy(policy)

    def test_unsafe_fallback_is_rejected(self):
        policy = self.policy()
        policy["failure_policy"]["fallback_weights_only_false"] = True
        with self.assertRaises(CheckpointSafeGlobalsPolicyError):
            validate_checkpoint_safe_globals_policy(policy)

    def test_host_deserialization_is_rejected(self):
        policy = self.policy()
        policy["deserialization_allowed_on_current_host"] = True
        with self.assertRaises(CheckpointSafeGlobalsPolicyError):
            validate_checkpoint_safe_globals_policy(policy)


if __name__ == "__main__":
    unittest.main()
