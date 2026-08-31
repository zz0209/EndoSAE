import copy
import json
import unittest
from pathlib import Path

from src.lr0_artifact_lock import LR0ArtifactLockError, validate_lr0_artifact_lock


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "lr0_artifact_lock_v0.json"


class LR0ArtifactLockTests(unittest.TestCase):
    def record(self):
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_current_lock_is_complete_but_install_blocked(self):
        record = self.record()
        validate_lr0_artifact_lock(record)
        self.assertTrue(record["lock_complete"])
        self.assertFalse(record["install_allowed"])

    def test_mutable_latest_bootstrap_is_rejected(self):
        record = self.record()
        record["artifacts"][0]["url"] = "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64"
        with self.assertRaisesRegex(LR0ArtifactLockError, "versioned release"):
            validate_lr0_artifact_lock(record)

    def test_hash_locked_artifact_requires_digest(self):
        record = self.record()
        record["artifacts"][0]["sha256"] = None
        with self.assertRaisesRegex(LR0ArtifactLockError, "require"):
            validate_lr0_artifact_lock(record)

    def test_complete_lock_cannot_drop_downloaded_hash(self):
        record = self.record()
        record["artifacts"][3]["sha256"] = None
        with self.assertRaisesRegex(LR0ArtifactLockError, "require"):
            validate_lr0_artifact_lock(record)

    def test_complete_lock_requires_dry_run_admission(self):
        record = self.record()
        record["dry_run_allowed"] = False
        with self.assertRaisesRegex(LR0ArtifactLockError, "must admit"):
            validate_lr0_artifact_lock(record)


if __name__ == "__main__":
    unittest.main()
