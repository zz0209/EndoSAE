import copy
import json
import unittest
from pathlib import Path

from src.modern_environment_lock import ModernEnvironmentLockError, validate_modern_environment_lock


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "configs" / "modern_environment_lock_v0.json"


class ModernEnvironmentLockTests(unittest.TestCase):
    def lock(self):
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def test_current_lock_passes(self):
        validate_modern_environment_lock(self.lock(), ROOT)

    def test_mixed_build_is_rejected(self):
        lock = self.lock()
        lock["runtime_probe"]["torch_torchvision_builds_coherent"] = False
        with self.assertRaises(ModernEnvironmentLockError):
            validate_modern_environment_lock(lock, ROOT)

    def test_failed_lr0_parity_cannot_be_rewritten(self):
        lock = self.lock()
        lock["preprocessing_probe"]["matches_lr0_under_frozen_tolerance"] = True
        with self.assertRaises(ModernEnvironmentLockError):
            validate_modern_environment_lock(lock, ROOT)

    def test_checkpoint_cannot_be_enabled(self):
        lock = self.lock()
        lock["checkpoint_loading_allowed"] = True
        with self.assertRaises(ModernEnvironmentLockError):
            validate_modern_environment_lock(lock, ROOT)


if __name__ == "__main__":
    unittest.main()
