import json
import unittest
from pathlib import Path

from src.synthetic_clip_fixture import SyntheticClipError, generate_synthetic_clip
from src.tensor_fingerprint import fingerprint_tensor_payload


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs" / "synthetic_preprocessing_clip_v0.json"


class SyntheticClipFixtureTests(unittest.TestCase):
    def spec(self):
        return json.loads(SPEC.read_text(encoding="utf-8"))

    def test_frozen_generator_reproduces_expected_bytes(self):
        spec = self.spec()
        payload = generate_synthetic_clip(spec)
        self.assertEqual(len(payload), 8 * 120 * 200 * 3)
        self.assertEqual(
            fingerprint_tensor_payload(payload, dtype="uint8", shape=spec["shape"], layout=spec["layout"]),
            spec["expected_tensor_fingerprint"],
        )

    def test_anchor_drift_is_rejected(self):
        spec = self.spec()
        spec["anchor_expectations"][1]["value"] += 1
        with self.assertRaisesRegex(SyntheticClipError, "anchor value"):
            generate_synthetic_clip(spec)


if __name__ == "__main__":
    unittest.main()
