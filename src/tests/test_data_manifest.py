import copy
import unittest

from src.data_manifest import (
    ManifestValidationError,
    validate_record,
    validate_split_integrity,
)


def valid_record():
    return {
        "schema_version": "0.1.0",
        "dataset_id": "sunseg",
        "dataset_version": "metadata-audit",
        "source_uri": "official-source",
        "rights_status": "verified_restricted",
        "license_or_terms_ref": "official-terms",
        "patient_id": None,
        "procedure_id": None,
        "source_case_id": "case1",
        "phantom_id": None,
        "segment_id": None,
        "trajectory_id": None,
        "grouping_level": "source_case",
        "group_id": "case1",
        "video_id": "case1-video",
        "clip_id": "case1_1",
        "paired_asset_id": None,
        "asset_sha256": "a" * 64,
        "split": "development",
        "temporal": {"ordered": True, "start_frame": 0, "end_frame": 9, "fps": 30.0},
        "labels": {
            "underlying_target_status": "present_supported",
            "observability": "visible_partial",
            "annotation_status": "source_ground_truth",
        },
    }


class ManifestRecordTests(unittest.TestCase):
    def test_valid_record(self):
        validate_record(valid_record(), require_authorized=True)

    def test_reversed_temporal_interval_is_rejected(self):
        record = valid_record()
        record["temporal"]["end_frame"] = -1
        with self.assertRaises(ManifestValidationError):
            validate_record(record)

    def test_group_id_must_match_source_case(self):
        record = valid_record()
        record["group_id"] = "case2"
        with self.assertRaises(ManifestValidationError):
            validate_record(record)

    def test_pending_rights_rejected_for_formal_use(self):
        record = valid_record()
        record["rights_status"] = "pending"
        with self.assertRaises(ManifestValidationError):
            validate_record(record, require_authorized=True)

    def test_absent_requires_not_applicable_observability(self):
        record = valid_record()
        record["labels"]["underlying_target_status"] = "absent_confirmed"
        with self.assertRaises(ManifestValidationError):
            validate_record(record)


class SplitIntegrityTests(unittest.TestCase):
    def test_same_source_case_cannot_cross_splits(self):
        first = valid_record()
        second = copy.deepcopy(first)
        second["clip_id"] = "case1_2"
        second["asset_sha256"] = "b" * 64
        second["split"] = "confirmation"
        with self.assertRaises(ManifestValidationError):
            validate_split_integrity([first, second])

    def test_phantom_pair_cannot_cross_splits(self):
        first = valid_record()
        first.update({
            "dataset_id": "c3vdv2",
            "grouping_level": "phantom_segment",
            "group_id": "c0-cecum-t1",
            "source_case_id": None,
            "phantom_id": "c0",
            "segment_id": "cecum-t1",
            "paired_asset_id": "pair-001",
        })
        second = copy.deepcopy(first)
        second["group_id"] = "c1-cecum-t1"
        second["phantom_id"] = "c1"
        second["asset_sha256"] = "b" * 64
        second["split"] = "test"
        with self.assertRaises(ManifestValidationError):
            validate_split_integrity([first, second])


if __name__ == "__main__":
    unittest.main()
