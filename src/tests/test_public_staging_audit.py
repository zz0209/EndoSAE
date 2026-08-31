import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_public_staging.py"
SPEC = importlib.util.spec_from_file_location("audit_public_staging", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PublicStagingAuditTests(unittest.TestCase):
    def test_forbidden_roots_and_markdown(self):
        errors = MODULE.audit([Path("docs/plan.txt"), Path("README.md")])
        self.assertEqual(len(errors), 2)

    def test_regular_safe_source(self):
        with tempfile.TemporaryDirectory(dir=MODULE.ROOT) as directory:
            target = Path(directory) / "safe.py"
            target.write_text("print('safe')\n", encoding="utf-8")
            relative = target.relative_to(MODULE.ROOT)
            self.assertEqual(MODULE.audit([relative]), [])

    def test_private_key_pattern(self):
        with tempfile.TemporaryDirectory(dir=MODULE.ROOT) as directory:
            target = Path(directory) / "secret.txt"
            target.write_text("-----BEGIN " + "PRIVATE KEY-----\n", encoding="utf-8")
            relative = target.relative_to(MODULE.ROOT)
            self.assertTrue(MODULE.audit([relative]))


if __name__ == "__main__":
    unittest.main()
