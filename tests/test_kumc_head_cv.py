import unittest
import numpy as np
import torch
from src.evaluation.kumc_head_cv import FrameSupportSampler


class Sampling(unittest.TestCase):
    def test_holdout_and_unqualified_frames_cannot_be_sampled(self):
        rows = [{'video_id': 'pre/1'}, {'video_id': 'post/1'}, {'video_id': 'pre/2'}]
        masks = np.zeros((3, 8, 196), dtype=bool)
        masks[0, 0, :3] = True
        masks[0, 1] = True
        masks[1, 2, :5] = True
        masks[2, :, :98] = True
        sampler = FrameSupportSampler(rows, masks, np.array([True, True, False]), torch, 'cpu')
        indices, labels = sampler.sample(2048, torch.Generator().manual_seed(123))
        values = indices.numpy()
        self.assertEqual(set(values // 196), {0, 10})
        np.testing.assert_array_equal(masks.reshape(-1)[values], labels.numpy().astype(bool))
        np.testing.assert_array_equal((values // 196)[::2], (values // 196)[1::2])
        self.assertEqual(int(labels.sum()), 1024)
        self.assertEqual(sampler.fit_frame_indices, [0, 10])


if __name__ == '__main__':
    unittest.main()
