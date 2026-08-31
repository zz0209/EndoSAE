import json
import tempfile
import unittest
from pathlib import Path

from src.reference_tensor_exchange import (
    ReferenceTensorExchangeError,
    validate_reference_tensor_exchange,
)


ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = ROOT / "configs" / "probes" / "reference_tensor_exchange_20260831.json"


class ReferenceTensorExchangeTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))

    def test_locked_project_asset_passes(self):
        path = validate_reference_tensor_exchange(self.record, ROOT)
        self.assertTrue(path.is_file())

    def test_g1_claim_is_rejected(self):
        self.record["g1_admission"] = True
        with self.assertRaises(ReferenceTensorExchangeError):
            validate_reference_tensor_exchange(self.record, ROOT, verify_asset=False)

    def test_path_escape_is_rejected(self):
        self.record["tensor"]["relative_path"] = "../outside.bin"
        with self.assertRaises(ReferenceTensorExchangeError):
            validate_reference_tensor_exchange(self.record, ROOT, verify_asset=False)

    def test_shape_byte_mismatch_is_rejected(self):
        self.record["tensor"]["shape"] = [3, 8, 1, 1]
        with self.assertRaises(ReferenceTensorExchangeError):
            validate_reference_tensor_exchange(self.record, ROOT, verify_asset=False)

    def test_hash_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            asset = Path(directory) / "tensor.bin"
            asset.write_bytes(b"\x00" * 24)
            self.record["tensor"].update(
                relative_path=asset.relative_to(ROOT).as_posix(),
                bytes=24,
                shape=[3, 2, 1, 1],
            )
            with self.assertRaises(ReferenceTensorExchangeError):
                validate_reference_tensor_exchange(self.record, ROOT)


if __name__ == "__main__":
    unittest.main()
