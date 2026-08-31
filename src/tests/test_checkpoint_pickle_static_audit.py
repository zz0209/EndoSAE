import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.checkpoint_pickle_static_audit import (
    CheckpointPickleStaticAuditError,
    audit_pickle_member,
    validate_static_audit_report,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "configs" / "probes" / "checkpoint_pickle_static_audit_20260831.json"
CHECKPOINT = ROOT / "artifacts" / "quarantine" / "endofm-main" / "endo_fm.pth"


class CheckpointPickleStaticAuditTests(unittest.TestCase):
    def report(self):
        return json.loads(REPORT.read_text(encoding="utf-8"))

    def test_current_report_is_conservative(self):
        validate_static_audit_report(self.report())

    def test_current_report_recomputes_from_quarantined_bytes(self):
        report = self.report()
        observed = audit_pickle_member(CHECKPOINT)
        for key in (
            "pickle_member", "pickle_member_size_bytes", "pickle_member_sha256",
            "opcode_count", "opcode_counts", "global_symbols",
            "forbidden_dynamic_opcodes_present", "persistent_id_opcode_count",
            "storage_member_count", "complete_stop",
        ):
            self.assertEqual(report[key], observed[key])

    def test_loading_cannot_be_authorized(self):
        report = self.report()
        report["checkpoint_loading_allowed"] = True
        with self.assertRaises(CheckpointPickleStaticAuditError):
            validate_static_audit_report(report)

    def test_unsafe_global_set_cannot_be_hidden(self):
        report = self.report()
        report["torch_static_unsafe_globals"] = []
        with self.assertRaises(CheckpointPickleStaticAuditError):
            validate_static_audit_report(report)

    def test_dependency_free_scanner_does_not_execute_reduce(self):
        marker = ROOT / "artifacts" / "static-audit-must-not-exist.txt"

        class WouldWrite:
            def __reduce__(self):
                return (Path.write_text, (marker, "unsafe"))

        payload = pickle.dumps(WouldWrite(), protocol=2)
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as directory:
            checkpoint = Path(directory) / "synthetic.pth"
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("archive/data.pkl", payload)
            result = audit_pickle_member(checkpoint)
        self.assertGreater(result["opcode_count"], 0)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
