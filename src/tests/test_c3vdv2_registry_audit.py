import tempfile
import unittest
from pathlib import Path

from src.c3vdv2_contracts import C3VDv2ContractError
from src.c3vdv2_registry_audit import audit_registered_summary


HEADER = "Colon,Segment,Phantom Number,Video Number,Video Name,Num_frames,preview\n"


class C3VDv2RegistryAuditTests(unittest.TestCase):
    def write_csv(self, body: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "summary.csv"
        path.write_text(HEADER + body, encoding="utf-8")
        return path

    def test_counts_pairs_and_missing(self):
        path = self.write_csv(
            "c1,cecum,t1,v2,c1_cecum_t1_v2,10,x\n"
            "c1,cecum,t1,v3,c1_cecum_t1_v3,10,y\n"
            "c1,rectum,t1,v1,c1_rectum_t1_v1,5,z\n"
        )
        result = audit_registered_summary(path)
        self.assertEqual((result.videos, result.frames, result.physical_units), (3, 25, 2))
        self.assertEqual(result.matched_v2_v3_pairs, 1)
        self.assertEqual(result.missing_v2_v3, ("c1_rectum_t1",))

    def test_detects_pair_frame_mismatch(self):
        path = self.write_csv(
            "c1,cecum,t1,v2,c1_cecum_t1_v2,10,x\n"
            "c1,cecum,t1,v3,c1_cecum_t1_v3,11,y\n"
        )
        result = audit_registered_summary(path)
        self.assertEqual(result.frame_mismatched_pairs, ("c1_cecum_t1",))

    def test_rejects_disagreeing_columns(self):
        path = self.write_csv("c1,rectum,t1,v2,c1_cecum_t1_v2,10,x\n")
        with self.assertRaises(C3VDv2ContractError):
            audit_registered_summary(path)


if __name__ == "__main__":
    unittest.main()
