import unittest

from src.c3vdv2_contracts import (
    C3VDv2ContractError,
    decode_flow_uint16,
    parse_video_id,
    validate_clean_debris_pair,
    validate_frame_inventory,
)


class C3VDv2ContractsTests(unittest.TestCase):
    def test_parse_and_pair(self):
        identity = parse_video_id("c1_cecum_t1_v2")
        self.assertEqual(identity.physical_unit_id, "c1_cecum_t1")
        self.assertEqual(
            validate_clean_debris_pair("c1_cecum_t1_v2", "c1_cecum_t1_v3"),
            "c1_cecum_t1_v2v3",
        )

    def test_rejects_mismatched_pair(self):
        with self.assertRaises(C3VDv2ContractError):
            validate_clean_debris_pair("c1_cecum_t1_v2", "c1_rectum_t1_v3")

    def test_flow_decode_endpoints_and_center(self):
        self.assertEqual(decode_flow_uint16(0), -20.0)
        self.assertEqual(decode_flow_uint16(65535), 20.0)
        self.assertAlmostEqual(decode_flow_uint16(32768), 20.0 / 65535, places=10)

    def test_flow_decode_rejects_invalid_values(self):
        for value in (-1, 65536, 1.5, True):
            with self.subTest(value=value), self.assertRaises(C3VDv2ContractError):
                decode_flow_uint16(value)

    def test_frame_inventory_alignment(self):
        modalities = {
            name: [10, 11, 12]
            for name in ("rgb", "depth", "normals", "optical_flow", "occlusion", "diffuse")
        }
        self.assertEqual(validate_frame_inventory(modalities), (10, 11, 12))

    def test_frame_inventory_rejects_mismatch(self):
        modalities = {
            name: [10, 11, 12]
            for name in ("rgb", "depth", "normals", "optical_flow", "occlusion", "diffuse")
        }
        modalities["optical_flow"] = [10, 12]
        with self.assertRaises(C3VDv2ContractError):
            validate_frame_inventory(modalities)


if __name__ == "__main__":
    unittest.main()
