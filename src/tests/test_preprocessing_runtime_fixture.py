import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.preprocessing_runtime_fixture import (
    PreprocessingFixtureError,
    evaluation_view_plan,
    validate_preprocessing_runtime_fixture,
)


H = "a" * 64
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs" / "endofm_reference_input_contract_v0.json"
SAMPLING = ROOT / "configs" / "examples" / "temporal_coordinate_sampling_example_v0.json"
INVALID_TEMPLATE = ROOT / "configs" / "examples" / "preprocessing_runtime_fixture_invalid_template_v0.json"


class PreprocessingRuntimeFixtureTests(unittest.TestCase):
    def valid_record(self, protocol="official-faithful"):
        construction, runtime = {
            "official-faithful": (1, 3),
            "corrected-center": (1, 1),
            "corrected-three-crop": (3, 3),
        }[protocol]
        views = []
        for item in evaluation_view_plan(construction, runtime):
            idx = item["spatial_sample_index"]
            views.append({**item, "resized_height": 224, "resized_width": 300,
                          "crop_x": (0, 38, 76)[idx], "crop_y": 0,
                          "output_tensor_sha256": H})
        return {
            "schema_version": "endosae.preprocessing-runtime-fixture.v0", "status": "pass",
            "fixture_id": f"synthetic-{protocol}",
            "reference_contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
            "sampling_index_asset_sha256": hashlib.sha256(SAMPLING.read_bytes()).hexdigest(),
            "clip_id": "coordinate-template-not-run", "protocol": protocol,
            "construction_num_spatial_crops": construction,
            "runtime_num_spatial_crops": runtime, "num_ensemble_views": 1,
            "decoder_backend": "pyav", "decoded_layout": "T,H,W,C",
            "decoded_dtype": "uint8", "decoded_channel_order": "RGB",
            "decoded_tensor_sha256": H, "normalized_layout": "C,T,H,W",
            "normalized_dtype": "float32", "normalized_shape": [3, 8, 224, 224],
            "normalized_tensor_sha256": H,
            "normalization_summary": {"min": -2.0, "max": 2.4, "mean": 0.1, "std": 0.8},
            "views": views,
        }

    def test_three_protocol_state_machines_are_distinct(self):
        self.assertEqual(evaluation_view_plan(1, 3), [{"stored_view_index": 0, "temporal_sample_index": 0, "spatial_sample_index": 0}])
        self.assertEqual(evaluation_view_plan(1, 1)[0]["spatial_sample_index"], 1)
        self.assertEqual([v["spatial_sample_index"] for v in evaluation_view_plan(3, 3)], [0, 1, 2])

    def test_all_protocol_records_pass(self):
        for protocol in ("official-faithful", "corrected-center", "corrected-three-crop"):
            validate_preprocessing_runtime_fixture(self.valid_record(protocol), CONTRACT, SAMPLING)

    def test_protocol_state_mismatch_is_rejected(self):
        record = self.valid_record()
        record["runtime_num_spatial_crops"] = 1
        with self.assertRaisesRegex(PreprocessingFixtureError, "protocol disagrees"):
            validate_preprocessing_runtime_fixture(record, CONTRACT, SAMPLING)

    def test_wrong_crop_coordinates_are_rejected(self):
        record = self.valid_record("corrected-center")
        record["views"][0]["crop_x"] = 0
        with self.assertRaisesRegex(PreprocessingFixtureError, "crop coordinates"):
            validate_preprocessing_runtime_fixture(record, CONTRACT, SAMPLING)

    def test_tampered_linked_contract_is_rejected(self):
        record = self.valid_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(CONTRACT.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(PreprocessingFixtureError, "contract file hash"):
                validate_preprocessing_runtime_fixture(record, path, SAMPLING)

    def test_sampling_clip_link_is_required(self):
        record = copy.deepcopy(self.valid_record())
        record["clip_id"] = "missing"
        with self.assertRaisesRegex(PreprocessingFixtureError, "clip_id missing"):
            validate_preprocessing_runtime_fixture(record, CONTRACT, SAMPLING)

    def test_invalid_template_is_linked_but_not_runtime_evidence(self):
        payload = json.loads(INVALID_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "invalid")
        validate_preprocessing_runtime_fixture(payload, CONTRACT, SAMPLING)


if __name__ == "__main__":
    unittest.main()
