import unittest

from src.endofm_contracts import (
    ContractError,
    DEFAULT_ENDOFM_LAYOUT,
    TokenLayout,
    validate_cache_metadata,
    validate_sampling_index_asset,
)


class TokenLayoutTests(unittest.TestCase):
    def test_default_shape_matches_pinned_source_inference(self):
        self.assertEqual(DEFAULT_ENDOFM_LAYOUT.grid_height, 14)
        self.assertEqual(DEFAULT_ENDOFM_LAYOUT.grid_width, 14)
        self.assertEqual(DEFAULT_ENDOFM_LAYOUT.patch_tokens, 1568)
        self.assertEqual(DEFAULT_ENDOFM_LAYOUT.token_count, 1569)
        self.assertEqual(
            DEFAULT_ENDOFM_LAYOUT.activation_shape_per_clip,
            (1569, 768),
        )

    def test_spatial_major_time_minor_landmarks(self):
        layout = DEFAULT_ENDOFM_LAYOUT
        self.assertEqual(layout.patch_token_index(0, 0, 0), 1)
        self.assertEqual(layout.patch_token_index(0, 0, 7), 8)
        self.assertEqual(layout.patch_token_index(0, 1, 0), 9)
        self.assertEqual(layout.patch_token_index(13, 13, 7), 1568)

    def test_index_round_trip_is_bijective(self):
        layout = TokenLayout(3, 8, 12, 4, 5)
        observed = set()
        for h in range(layout.grid_height):
            for w in range(layout.grid_width):
                for t in range(layout.frames):
                    index = layout.patch_token_index(h, w, t)
                    self.assertEqual(layout.patch_coordinates(index), (h, w, t))
                    observed.add(index)
        self.assertEqual(observed, set(range(1, layout.token_count)))

    def test_cls_and_out_of_range_indices_are_rejected(self):
        layout = DEFAULT_ENDOFM_LAYOUT
        with self.assertRaises(ContractError):
            layout.patch_coordinates(0)
        with self.assertRaises(ContractError):
            layout.patch_coordinates(layout.token_count)
        with self.assertRaises(ContractError):
            layout.patch_token_index(0, 0, layout.frames)

    def test_non_divisible_image_is_rejected(self):
        with self.assertRaises(ContractError):
            TokenLayout(8, 225, 224, 16, 768)


class CacheMetadataTests(unittest.TestCase):
    def valid_metadata(self):
        return {
            "schema_version": "endosae.activation-cache.v0",
            "model_repository": "https://github.com/med-air/Endo-FM",
            "model_commit": "206427ebfb77a937ef0cd60370331bcedd74e5a2",
            "checkpoint_sha256": "a" * 64,
            "preprocessing_id": "official-polypdiag-eval-v0",
            "layer": "model.blocks.5",
            "hook_kind": "module-output",
            "dtype": "float32",
            "shape": [2, 1569, 768],
            "token_layout": {
                "order": "spatial-major,time-minor",
                "has_global_cls": True,
                "frames": 8,
                "grid_height": 14,
                "grid_width": 14,
                "hidden_size": 768,
            },
            "sampling": {
                "num_frames": 8,
                "sampling_rate": 32,
                "target_fps": 30.0,
                "index_space": "decoded-frame-index",
                "index_policy": "pinned-endofm-linspace-long",
                "clamp_policy": "clamp-to-last-decoded-frame",
                "decoder_source_sha256": "c" * 64,
                "indices_schema_version": "endosae.sampling-indices.v0",
                "indices_asset_sha256": "d" * 64,
                "clip_count": 2,
                "clamped_clip_count": 1,
                "duplicate_index_clip_count": 1,
                "contains_clamped_clips": True,
                "contains_duplicate_indices": True,
            },
            "source_manifest_sha256": "b" * 64,
            "extractor_commit": "uncommitted-test-fixture",
        }

    def test_valid_metadata(self):
        validate_cache_metadata(self.valid_metadata())

    def test_missing_provenance_is_rejected(self):
        metadata = self.valid_metadata()
        del metadata["source_manifest_sha256"]
        with self.assertRaisesRegex(ContractError, "source_manifest_sha256"):
            validate_cache_metadata(metadata)

    def test_bad_hash_is_rejected(self):
        metadata = self.valid_metadata()
        metadata["checkpoint_sha256"] = "not-a-hash"
        with self.assertRaisesRegex(ContractError, "checkpoint_sha256"):
            validate_cache_metadata(metadata)

    def test_wrong_layout_order_is_rejected(self):
        metadata = self.valid_metadata()
        metadata["token_layout"]["order"] = "time-major,spatial-minor"
        with self.assertRaisesRegex(ContractError, "layout order"):
            validate_cache_metadata(metadata)

    def test_sampling_summary_flags_must_match_counts(self):
        metadata = self.valid_metadata()
        metadata["sampling"]["contains_clamped_clips"] = False
        with self.assertRaisesRegex(ContractError, "contains_clamped_clips"):
            validate_cache_metadata(metadata)


class SamplingIndexAssetTests(unittest.TestCase):
    def valid_asset(self):
        return {
            "schema_version": "endosae.sampling-indices.v0",
            "records": [{
                "clip_id": "clip-001",
                "manifest_asset_sha256": "e" * 64,
                "source_video_id": "c1_transverse1_t1_v2",
                "source_frame_count": 117,
                "source_fps": 30.0,
                "target_fps": 30.0,
                "num_frames": 8,
                "sampling_rate": 32,
                "requested_frame_indices": [0, 36, 72, 109, 145, 182, 218, 255],
                "actual_frame_indices": [0, 36, 72, 109, 116, 116, 116, 116],
                "was_clamped": True,
                "has_duplicate_indices": True,
            }],
        }

    def test_valid_clamped_asset(self):
        validate_sampling_index_asset(self.valid_asset())

    def test_actual_indices_must_match_clamp_policy(self):
        asset = self.valid_asset()
        asset["records"][0]["actual_frame_indices"][-1] = 115
        with self.assertRaisesRegex(ContractError, "clamp policy"):
            validate_sampling_index_asset(asset)

    def test_duplicate_flag_must_match_indices(self):
        asset = self.valid_asset()
        asset["records"][0]["has_duplicate_indices"] = False
        with self.assertRaisesRegex(ContractError, "has_duplicate_indices"):
            validate_sampling_index_asset(asset)

    def test_sampling_parameters_must_be_typed_and_positive(self):
        asset = self.valid_asset()
        asset["records"][0]["source_fps"] = 0
        with self.assertRaisesRegex(ContractError, "source_fps"):
            validate_sampling_index_asset(asset)

        asset = self.valid_asset()
        asset["records"][0]["sampling_rate"] = True
        with self.assertRaisesRegex(ContractError, "sampling_rate"):
            validate_sampling_index_asset(asset)

    def test_flags_must_be_boolean(self):
        asset = self.valid_asset()
        asset["records"][0]["was_clamped"] = 1
        with self.assertRaisesRegex(ContractError, "was_clamped must be boolean"):
            validate_sampling_index_asset(asset)


if __name__ == "__main__":
    unittest.main()
