import hashlib
import json
import unittest
from pathlib import Path

from src.temporal_coordinate_fixture import (
    TemporalCoordinateFixtureError,
    validate_temporal_coordinate_fixture,
)


class TemporalCoordinateFixtureTests(unittest.TestCase):
    sampling_hash = "c" * 64

    def sampling_asset(self, duplicate=False):
        actual = [0, 4, 8] if not duplicate else [0, 4, 4]
        return {
            "schema_version": "endosae.sampling-indices.v0",
            "records": [{
                "clip_id": "coordinate-clip",
                "manifest_asset_sha256": "a" * 64,
                "source_video_id": "synthetic-frame-code",
                "source_frame_count": 9 if not duplicate else 5,
                "source_fps": 30.0,
                "target_fps": 30.0,
                "num_frames": 3,
                "sampling_rate": 4,
                "requested_frame_indices": [0, 4, 8],
                "actual_frame_indices": actual,
                "was_clamped": duplicate,
                "has_duplicate_indices": duplicate,
            }],
        }

    def valid_fixture(self):
        original_hashes = ["1" * 64, "2" * 64, "3" * 64]
        permutation = [2, 0, 1]
        return {
            "schema_version": "endosae.temporal-coordinate-fixture.v0",
            "status": "pass",
            "fixture_id": "patch-embed-frame-code-permutation",
            "model_commit": "206427ebfb77a937ef0cd60370331bcedd74e5a2",
            "checkpoint_sha256": "b" * 64,
            "preprocessing_id": "frame-code-v0",
            "sampling_index_asset_sha256": "c" * 64,
            "clip_id": "coordinate-clip",
            "device": "cpu",
            "dtype": "float32",
            "hook_target": "patch_embed",
            "input_layout": "B,C,T,H,W",
            "image_height": 8,
            "image_width": 8,
            "patch_size": 4,
            "hidden_size": 5,
            "original_input_sha256": "d" * 64,
            "permuted_input_sha256": "e" * 64,
            "original_actual_frame_indices": [0, 4, 8],
            "permutation": permutation,
            "permuted_actual_frame_indices": [8, 0, 4],
            "activation_shape": [3, 4, 5],
            "original_frame_fingerprints": original_hashes,
            "permuted_frame_fingerprints": [original_hashes[i] for i in permutation],
            "token_landmarks": [
                {"h": 0, "w": 0, "t": 0, "token_index": 1, "decoded_frame_index": 0},
                {"h": 0, "w": 0, "t": 1, "token_index": 2, "decoded_frame_index": 4},
                {"h": 0, "w": 0, "t": 2, "token_index": 3, "decoded_frame_index": 8},
            ],
        }

    def test_valid_pass_links_sampling_permutation_and_landmarks(self):
        validate_temporal_coordinate_fixture(
            self.valid_fixture(), self.sampling_asset(), self.sampling_hash
        )

    def test_pass_rejects_duplicate_decoded_frames(self):
        fixture = self.valid_fixture()
        fixture["original_actual_frame_indices"] = [0, 4, 4]
        fixture["permuted_actual_frame_indices"] = [4, 0, 4]
        fixture["token_landmarks"][-1]["decoded_frame_index"] = 4
        with self.assertRaisesRegex(TemporalCoordinateFixtureError, "duplicate-free"):
            validate_temporal_coordinate_fixture(
                fixture, self.sampling_asset(duplicate=True), self.sampling_hash
            )

    def test_permuted_fingerprints_must_follow_exact_permutation(self):
        fixture = self.valid_fixture()
        fixture["permuted_frame_fingerprints"] = fixture["original_frame_fingerprints"]
        with self.assertRaisesRegex(TemporalCoordinateFixtureError, "exact permutation"):
            validate_temporal_coordinate_fixture(fixture, self.sampling_asset(), self.sampling_hash)

    def test_landmark_must_match_token_layout(self):
        fixture = self.valid_fixture()
        fixture["token_landmarks"][1]["token_index"] = 3
        with self.assertRaisesRegex(TemporalCoordinateFixtureError, "H-W-T layout"):
            validate_temporal_coordinate_fixture(fixture, self.sampling_asset(), self.sampling_hash)

    def test_fixture_indices_must_match_sampling_provenance(self):
        fixture = self.valid_fixture()
        fixture["original_actual_frame_indices"] = [0, 3, 8]
        with self.assertRaisesRegex(TemporalCoordinateFixtureError, "sampling provenance"):
            validate_temporal_coordinate_fixture(fixture, self.sampling_asset(), self.sampling_hash)

    def test_sampling_asset_hash_must_match_record(self):
        with self.assertRaisesRegex(TemporalCoordinateFixtureError, "hash does not match"):
            validate_temporal_coordinate_fixture(
                self.valid_fixture(), self.sampling_asset(), "f" * 64
            )

    def test_invalid_template_is_hash_linked_but_not_a_pass_claim(self):
        root = Path(__file__).resolve().parents[2]
        sampling_path = root / "configs" / "examples" / "temporal_coordinate_sampling_example_v0.json"
        fixture_path = root / "configs" / "examples" / "temporal_coordinate_fixture_invalid_template_v0.json"
        sampling_bytes = sampling_path.read_bytes()
        sampling = json.loads(sampling_bytes)
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        observed_hash = hashlib.sha256(sampling_bytes).hexdigest()
        validate_temporal_coordinate_fixture(fixture, sampling, observed_hash)
        self.assertEqual(fixture["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
