import copy
import unittest

from src.visibility_annotations import VisibilityAnnotationError, validate_annotation_record


def valid_record():
    return {
        "schema_version": "0.1.0",
        "blind_item_id": "A0-1A2B3C4D",
        "annotator_id": "annotator-01",
        "round_id": "independent",
        "same_target_pre_post": "yes",
        "middle_observability": "censored_in_view",
        "censoring_mechanisms": ["fluid_or_bubble"],
        "boundary_start_frame": 20,
        "boundary_end_frame": 29,
        "confidence": 3,
        "exclude_reason": "none",
        "notes": None,
    }


class VisibilityAnnotationTests(unittest.TestCase):
    def test_valid_record(self):
        validate_annotation_record(valid_record())

    def test_unknown_mechanism_is_exclusive(self):
        record = valid_record()
        record["censoring_mechanisms"] = ["unknown", "debris"]
        with self.assertRaises(VisibilityAnnotationError):
            validate_annotation_record(record)

    def test_censored_record_requires_boundaries(self):
        record = valid_record()
        record["boundary_end_frame"] = None
        with self.assertRaises(VisibilityAnnotationError):
            validate_annotation_record(record)

    def test_visible_middle_is_explicitly_excluded(self):
        record = valid_record()
        record.update({"middle_observability": "visible_partial", "censoring_mechanisms": [], "boundary_start_frame": None, "boundary_end_frame": None, "exclude_reason": "no_true_censoring"})
        validate_annotation_record(record)

    def test_unresolved_identity_requires_exclusion(self):
        record = valid_record()
        record["same_target_pre_post"] = "uncertain"
        with self.assertRaises(VisibilityAnnotationError):
            validate_annotation_record(record)


if __name__ == "__main__":
    unittest.main()
