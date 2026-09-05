import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.evaluation.realcolon_task_supervised import SupportSampler, feature_stats, fit_one, fold_masks
from src.sae.baselines import TopKAutoencoder


class SupervisedTaskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rows = [{"video_id": video, "split": "train", "class": label}
                     for video in ("a", "b", "held") for label in (0, 1) for _ in range(2)]
        self.masks = np.zeros((12, 2, 4), dtype=bool)
        for index, row in enumerate(self.rows):
            if row["class"]:
                self.masks[index, :, index % 2] = True
        self.fit = np.arange(12) < 8

    def test_sampler_is_label_correct_balanced_and_excludes_heldout(self):
        sampler = SupportSampler(self.rows, self.masks, self.fit, "cpu", torch)
        generator = torch.Generator().manual_seed(4)
        indices, labels = sampler.sample(4000, generator)
        self.assertLess(int(indices.max()), 8 * 8)
        np.testing.assert_array_equal(labels.numpy().astype(bool), self.masks.reshape(-1)[indices.numpy()])
        self.assertEqual(float(labels.mean()), .5)
        clips = indices.numpy() // 8
        self.assertEqual(int(np.sum(clips % 4 < 2)), 1000)
        self.assertEqual(sampler.videos, ["a", "b"])
        self.assertTrue(all(np.bincount(clips, minlength=8) > 100))

    def test_fold_and_statistics_do_not_fit_validation(self):
        folds = list(fold_masks(self.rows, [["a"], ["b"], ["held"]]))
        self.assertEqual(len(folds), 3)
        with self.assertRaises(ValueError):
            list(fold_masks(self.rows, [["a"], ["b"]]))
        with self.assertRaises(ValueError):
            list(fold_masks([dict(self.rows[0], split="development")] + self.rows[1:], [["a"], ["b"], ["held"]]))
        x = torch.arange(24, dtype=torch.float32).reshape(12, 2)
        before = feature_stats(x, torch.arange(8), lambda value: value, torch)
        x[8:] = 1000000.
        after = feature_stats(x, torch.arange(8), lambda value: value, torch)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    def test_task_gradient_differs_from_reconstruction_only_and_reference_is_unchanged(self):
        torch.manual_seed(4)
        x = torch.randn(96, 6)
        x[:, 0] = torch.tensor(self.masks.reshape(-1).astype(float) * 2 - 1)
        model = TopKAutoencoder(6, 12, 3)
        initial = copy.deepcopy(model.state_dict())
        fit_indices = torch.arange(64)
        stats = {"topk": feature_stats(x, fit_indices, model.encode_inference, torch)}
        sampler = SupportSampler(self.rows, self.masks, self.fit, "cpu", torch)
        config = {"seed": 4, "head_learning_rate": .05, "joint_head_weight_decay": 1.,
                  "finetune_learning_rate": .01, "head_steps": 20, "head_batch_size": 32, "topk_batch_size": 32}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trained = []
            for name in ("topk_reconstruction_continued", "topk_task"):
                path = root / name
                path.mkdir()
                pred = fit_one(name, {"mse_weight": 1.}, x, fit_indices, sampler,
                               {"topk": model}, stats, config, path, torch)
                self.assertTrue(np.isfinite(pred).all())
                with np.load(path / "dictionary.npz", allow_pickle=False) as state:
                    trained.append(np.array(state["encoder.weight"]))
            self.assertFalse(np.array_equal(*trained))
        self.assertTrue(all(torch.equal(initial[key], value) for key, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
