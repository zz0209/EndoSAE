import copy
import unittest

from src.visibility_episodes import VisibilityEpisodeError, validate_visibility_episode


def records():
    base = {
        "dataset_id": "sunseg",
        "dataset_version": "frozen-v1",
        "source_case_id": "case1",
        "split": "development",
        "temporal": {"start_frame": 0, "end_frame": 299},
        "labels": {"underlying_target_status": "present_supported"},
    }
    result = []
    for char in "a":
        record = copy.deepcopy(base)
        record["asset_sha256"] = char * 64
        result.append(record)
    return result


def episode():
    return {
        "schema_version": "0.1.0",
        "episode_id": "case1-lesion1-episode1",
        "dataset_id": "sunseg",
        "dataset_version": "frozen-v1",
        "source_case_id": "case1",
        "target_id": "lesion1",
        "split": "development",
        "identity_evidence": "continuous_mask_track",
        "segments": [
            {"manifest_asset_sha256": "a" * 64, "role": "pre_visible", "start_frame": 10, "end_frame": 19, "observability": "visible_full", "censoring_mechanisms": [], "annotation_status": "source_ground_truth"},
            {"manifest_asset_sha256": "a" * 64, "role": "censored", "start_frame": 20, "end_frame": 29, "observability": "censored_in_view", "censoring_mechanisms": ["fluid_or_bubble"], "annotation_status": "expert_confirmed"},
            {"manifest_asset_sha256": "a" * 64, "role": "post_visible", "start_frame": 30, "end_frame": 39, "observability": "visible_partial", "censoring_mechanisms": [], "annotation_status": "source_ground_truth"},
        ],
    }


class VisibilityEpisodeTests(unittest.TestCase):
    def test_valid_episode(self):
        validate_visibility_episode(episode(), records())

    def test_roles_cannot_be_reordered(self):
        value = episode()
        value["segments"][0], value["segments"][1] = value["segments"][1], value["segments"][0]
        with self.assertRaises(VisibilityEpisodeError):
            validate_visibility_episode(value, records())

    def test_segments_cannot_overlap(self):
        value = episode()
        value["segments"][1]["start_frame"] = 19
        with self.assertRaises(VisibilityEpisodeError):
            validate_visibility_episode(value, records())

    def test_manifest_case_must_match(self):
        manifest = records()
        manifest[0]["source_case_id"] = "case2"
        with self.assertRaises(VisibilityEpisodeError):
            validate_visibility_episode(episode(), manifest)

    def test_all_anchors_require_present_target(self):
        manifest = records()
        manifest[0]["labels"]["underlying_target_status"] = "unknown"
        with self.assertRaises(VisibilityEpisodeError):
            validate_visibility_episode(episode(), manifest)

    def test_visible_anchor_cannot_have_censoring_mechanism(self):
        value = episode()
        value["segments"][0]["censoring_mechanisms"] = ["blur_or_fast_motion"]
        with self.assertRaises(VisibilityEpisodeError):
            validate_visibility_episode(value, records())

    def test_cross_clip_episode_is_rejected_without_case_timeline(self):
        value = episode()
        value["segments"][2]["manifest_asset_sha256"] = "b" * 64
        manifest = records()
        extra = copy.deepcopy(manifest[0])
        extra["asset_sha256"] = "b" * 64
        manifest.append(extra)
        with self.assertRaises(VisibilityEpisodeError):
            validate_visibility_episode(value, manifest)

    def test_confirmatory_episode_rejects_unknown_mechanism(self):
        value = episode()
        value["segments"][1]["censoring_mechanisms"] = ["unknown"]
        with self.assertRaises(VisibilityEpisodeError):
            validate_visibility_episode(value, records(), confirmatory=True)

    def test_confirmatory_episode_rejects_rule_only_annotation(self):
        value = episode()
        value["segments"][1]["annotation_status"] = "derived_rule"
        with self.assertRaises(VisibilityEpisodeError):
            validate_visibility_episode(value, records(), confirmatory=True)


if __name__ == "__main__":
    unittest.main()
