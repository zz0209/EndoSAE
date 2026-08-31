import json
import tempfile
import unittest
from pathlib import Path

from src.preprocessing_raw_diff import PreprocessingRawDiffError, compare_raw_assets


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "probes" / "preprocessing_raw_assets_20260831.json"


class PreprocessingRawDiffTests(unittest.TestCase):
    def test_observed_assets_are_hash_verified_and_compared(self):
        result = compare_raw_assets(MANIFEST, ROOT)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["formal_preprocessing_admission_allowed"])
        self.assertEqual([x["stage"] for x in result["stages"]], [
            "resize", "crop_0", "crop_1", "crop_2"
        ])

    def test_manifest_hash_tampering_is_rejected(self):
        record = json.loads(MANIFEST.read_text(encoding="utf-8"))
        record["stages"][0]["reference"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(PreprocessingRawDiffError, "hash mismatch"):
                compare_raw_assets(path, ROOT)


if __name__ == "__main__":
    unittest.main()
