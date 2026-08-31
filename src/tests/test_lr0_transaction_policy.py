import copy
import json
import unittest
from pathlib import Path

from src.lr0_transaction_policy import LR0TransactionPolicyError, validate_lr0_transaction_policy


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "lr0_transaction_policy_v0.json"


class LR0TransactionPolicyTests(unittest.TestCase):
    def record(self):
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_current_policy_admits_locked_install_but_blocks_checkpoint(self):
        record = self.record()
        validate_lr0_transaction_policy(record, ROOT)
        self.assertTrue(record["dry_run_allowed"])
        self.assertTrue(record["install_allowed"])
        self.assertFalse(record["checkpoint_allowed"])

    def test_artifact_lock_tampering_is_rejected(self):
        record = self.record()
        record["artifact_lock"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(LR0TransactionPolicyError, "hash mismatch"):
            validate_lr0_transaction_policy(record, ROOT)

    def test_user_condarc_cannot_leak_into_solve(self):
        record = self.record()
        record["transaction"]["user_condarc_read"] = True
        with self.assertRaisesRegex(LR0TransactionPolicyError, "user config"):
            validate_lr0_transaction_policy(record, ROOT)

    def test_unlocated_conda_build_cannot_replace_wheel(self):
        record = self.record()
        record["official_pytorch_conda_alternative"]["replacement_authorized"] = True
        with self.assertRaisesRegex(LR0TransactionPolicyError, "cannot replace"):
            validate_lr0_transaction_policy(record, ROOT)

    def test_lr0_install_cannot_pre_authorize_checkpoint(self):
        record = self.record()
        record["checkpoint_allowed"] = True
        with self.assertRaisesRegex(LR0TransactionPolicyError, "cannot admit checkpoint"):
            validate_lr0_transaction_policy(record, ROOT)


if __name__ == "__main__":
    unittest.main()
