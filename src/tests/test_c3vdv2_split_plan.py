import json
import unittest
from pathlib import Path

from src.c3vdv2_contracts import C3VDv2ContractError
from src.c3vdv2_split_plan import (
    PairUnit,
    endofm_deterministic_indices,
    endofm_required_unclamped_frames,
    load_complete_pairs,
    plan_geometry_grouped_split,
    summarize_plan,
)


class C3VDv2SplitPlanTests(unittest.TestCase):
    def test_short_video_clamps_and_repeats_boundary(self):
        indices = endofm_deterministic_indices(117)
        self.assertEqual(indices, (0, 36, 72, 109, 116, 116, 116, 116))
        self.assertLess(len(set(indices)), 8)

    def test_256_frames_are_duplicate_free(self):
        self.assertEqual(endofm_deterministic_indices(256), (0, 36, 72, 109, 145, 182, 218, 255))
        self.assertEqual(endofm_required_unclamped_frames(), 256)

    def test_invalid_sampling_parameters_are_rejected(self):
        with self.assertRaises(C3VDv2ContractError):
            endofm_deterministic_indices(0)

    def test_split_keeps_geometry_groups_together(self):
        pairs = []
        for colon, group_count in (("c1", 8), ("c2", 7)):
            for group_index in range(group_count):
                geometry = f"{colon}_segment{group_index}"
                for take in range(1, 5):
                    pairs.append(PairUnit(f"{geometry}_t{take}", geometry, colon, f"segment{group_index}", 300))
        assignment = plan_geometry_grouped_split(tuple(pairs))
        self.assertEqual(len(assignment), 15)
        self.assertEqual(sorted(assignment.values()).count("confirmation"), 3)
        self.assertEqual(sorted(assignment.values()).count("test"), 3)
        for split in ("development", "confirmation", "test"):
            colons = {pair.colon for pair in pairs if assignment[pair.geometry_group] == split}
            self.assertEqual(colons, {"c1", "c2"})

    def test_pinned_candidate_config_matches_official_registry(self):
        root = Path(__file__).resolve().parents[2]
        pairs = load_complete_pairs(root / "third_party/C3VDv2/docs/assets/data/C3VDv2_data_summary_registered.csv")
        assignment = plan_geometry_grouped_split(pairs)
        config = json.loads((root / "configs/c3vdv2_split_candidate_v0.json").read_text(encoding="utf-8"))
        self.assertEqual(config["assignment"], assignment)
        summary = summarize_plan(pairs, assignment)
        for split in ("development", "confirmation", "test"):
            self.assertEqual(config["all_complete_pairs"][split]["pairs"], summary[split]["pairs"])
            self.assertEqual(config["default_rate_32_sampling"][split]["unclamped_pairs"], summary[split]["default_rate_unclamped_pairs"])


if __name__ == "__main__":
    unittest.main()
