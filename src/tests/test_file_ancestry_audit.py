import json
import unittest
from pathlib import Path

from src.file_ancestry_audit import FileAncestryError, validate_file_ancestry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "endofm_file_ancestry_v0.json"


class FileAncestryAuditTests(unittest.TestCase):
    def registry(self):
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_pinned_comparisons_reproduce(self):
        validate_file_ancestry(self.registry(), PROJECT_ROOT)

    def test_tampered_similarity_is_rejected(self):
        registry = self.registry()
        registry["comparisons"][0]["line_similarity"] = 1.0
        with self.assertRaisesRegex(FileAncestryError, "similarity"):
            validate_file_ancestry(registry, PROJECT_ROOT)

    def test_path_escape_is_rejected(self):
        registry = self.registry()
        registry["comparisons"][0]["upstream_path"] = "../outside.py"
        with self.assertRaisesRegex(FileAncestryError, "escapes"):
            validate_file_ancestry(registry, PROJECT_ROOT)

    def test_incomplete_history_is_rejected(self):
        registry = self.registry()
        registry["upstream_history"]["svt_history_complete"] = False
        with self.assertRaisesRegex(FileAncestryError, "complete SVT history"):
            validate_file_ancestry(registry, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
