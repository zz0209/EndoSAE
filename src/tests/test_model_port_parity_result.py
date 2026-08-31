import json
import unittest
from pathlib import Path

from src.model_port_parity_result import ModelPortParityResultError, validate_model_port_parity_result


ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "configs" / "probes" / "model_port_parity_result_incomplete_20260831.json"


class ModelPortParityResultTests(unittest.TestCase):
    def record(self):
        return json.loads(RECORD.read_text(encoding="utf-8"))

    def test_current_record_is_valid_and_incomplete(self):
        validate_model_port_parity_result(self.record())

    def test_missing_requirement_cannot_be_hidden(self):
        record = self.record()
        record["missing_requirements"].pop()
        with self.assertRaises(ModelPortParityResultError):
            validate_model_port_parity_result(record)

    def test_incomplete_result_cannot_pass(self):
        record = self.record()
        record["model_port_parity_pass"] = True
        with self.assertRaises(ModelPortParityResultError):
            validate_model_port_parity_result(record)

    def test_g1_cannot_be_admitted(self):
        record = self.record()
        record["g1_admission"] = True
        with self.assertRaises(ModelPortParityResultError):
            validate_model_port_parity_result(record)


if __name__ == "__main__":
    unittest.main()
