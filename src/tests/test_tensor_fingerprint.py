import struct
import unittest

from src.tensor_fingerprint import (
    TensorFingerprintError,
    fingerprint_tensor_payload,
    summarize_tensor_payload,
)


class TensorFingerprintTests(unittest.TestCase):
    def test_metadata_is_domain_separated_from_identical_bytes(self):
        payload = bytes(range(8))
        a = fingerprint_tensor_payload(payload, dtype="uint8", shape=[2, 4], layout="T,C")
        b = fingerprint_tensor_payload(payload, dtype="uint8", shape=[4, 2], layout="T,C")
        c = fingerprint_tensor_payload(payload, dtype="uint8", shape=[2, 4], layout="C,T")
        self.assertEqual(len({a, b, c}), 3)

    def test_float32_little_endian_summary_is_deterministic(self):
        payload = struct.pack("<4f", -1.0, 0.0, 1.0, 2.0)
        summary = summarize_tensor_payload(payload, dtype="float32-le", shape=[1, 4], layout="C,T")
        self.assertEqual(summary["min"], -1.0)
        self.assertEqual(summary["max"], 2.0)
        self.assertEqual(summary["mean"], 0.5)
        self.assertAlmostEqual(summary["std"], 1.118033988749895)

    def test_payload_length_must_match_shape(self):
        with self.assertRaisesRegex(TensorFingerprintError, "payload length"):
            fingerprint_tensor_payload(b"123", dtype="uint8", shape=[2, 2], layout="H,W")

    def test_nonfinite_values_are_rejected_from_summary(self):
        payload = struct.pack("<2f", 0.0, float("nan"))
        with self.assertRaisesRegex(TensorFingerprintError, "finite"):
            summarize_tensor_payload(payload, dtype="float32-le", shape=[2], layout="T")

    def test_mutable_or_ambiguous_dtype_is_rejected(self):
        with self.assertRaisesRegex(TensorFingerprintError, "immutable"):
            fingerprint_tensor_payload(bytearray([1]), dtype="uint8", shape=[1], layout="T")
        with self.assertRaisesRegex(TensorFingerprintError, "unsupported"):
            fingerprint_tensor_payload(b"\0\0\0\0", dtype="float32", shape=[1], layout="T")


if __name__ == "__main__":
    unittest.main()
