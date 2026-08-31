import copy
import json
import unittest
from pathlib import Path

from src.ead2020_admission import EAD2020AdmissionError, validate_ead2020_admission


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = PROJECT_ROOT / "configs" / "ead2020_admission_v0.json"


class EAD2020AdmissionTests(unittest.TestCase):
    def record(self):
        return json.loads(RECORD_PATH.read_text(encoding="utf-8"))

    def test_current_record_is_valid(self):
        validate_ead2020_admission(self.record())

    def test_license_drift_is_rejected(self):
        record = self.record()
        record["dataset"]["license_spdx"] = "CC-BY-4.0"
        with self.assertRaisesRegex(EAD2020AdmissionError, "license_spdx"):
            validate_ead2020_admission(record)

    def test_prior_v3_cannot_silently_replace_latest_v4(self):
        record = self.record()
        record["dataset"]["version"] = 3
        record["dataset"]["doi"] = "10.17632/c7fjbxcgj9.3"
        with self.assertRaisesRegex(EAD2020AdmissionError, "doi|version"):
            validate_ead2020_admission(record)

    def test_archive_members_cannot_be_claimed_inspected(self):
        record = self.record()
        record["public_metadata_boundary"]["archive_members_inspected"] = True
        with self.assertRaisesRegex(EAD2020AdmissionError, "archive_members_inspected"):
            validate_ead2020_admission(record)

    def test_code_repository_license_cannot_replace_dataset_license(self):
        record = self.record()
        record["primary_evidence"][-1]["root_license_detected"] = True
        with self.assertRaisesRegex(EAD2020AdmissionError, "distinct"):
            validate_ead2020_admission(record)

    def test_missing_grouping_blocks_custom_split(self):
        record = self.record()
        record["admission"]["custom_split_authorized"] = True
        with self.assertRaisesRegex(EAD2020AdmissionError, "custom_split"):
            validate_ead2020_admission(record)

    def test_unaccepted_terms_block_formal_use(self):
        record = self.record()
        record["admission"]["formal_experiment_use_authorized"] = True
        with self.assertRaisesRegex(EAD2020AdmissionError, "formal_experiment"):
            validate_ead2020_admission(record)

    def test_missing_sequence_identity_blocks_temporal_claim(self):
        record = self.record()
        record["admission"]["temporal_claim_authorized"] = True
        with self.assertRaisesRegex(EAD2020AdmissionError, "temporal_claim"):
            validate_ead2020_admission(record)


if __name__ == "__main__":
    unittest.main()
