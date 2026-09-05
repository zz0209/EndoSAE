import copy
import unittest

import numpy as np

from src.evaluation.realcolon_task_regularization import FIXED_KEYS, validate_config, validate_scope


class ReusedFoldTests(unittest.TestCase):
    def test_reuse_rejects_a_different_fold_or_labels(self):
        rows = [{"video_id": "a", "class": 0, "split": "train"}, {"video_id": "b", "class": 1, "split": "train"}]
        fit = np.array([True, False])
        scope = {"all_rows": [rows[0], dict(rows[1], split="validation")], "fit_videos": ["a"],
                 "validation_videos": ["b"], "fit_clip_indices": [0]}
        validate_scope(scope, rows, fit)
        with self.assertRaises(ValueError):
            validate_scope(scope, rows, ~fit)
        changed = copy.deepcopy(rows)
        changed[0]["class"] = 1
        with self.assertRaises(ValueError):
            validate_scope(scope, changed, fit)

    def test_only_declared_head_regularization_changes(self):
        reference = {key: "fixture" for key in FIXED_KEYS}
        current = dict(reference, mse_weight_candidates=[.1], joint_head_weight_decay=10.)
        validate_config(current, dict(reference, joint_head_weight_decay=1.))
        with self.assertRaises(ValueError):
            validate_config(dict(current, head_steps=99), reference)
        with self.assertRaises(ValueError):
            validate_config(dict(current, mse_weight_candidates=[1.]), reference)


if __name__ == "__main__":
    unittest.main()
