import unittest

import numpy as np
import torch

from src.evaluation.realcolon_score_preserving import constrained_basis, probabilities


class ScorePreservingProjectionTest(unittest.TestCase):
    def test_low_variance_task_direction_survives_compression(self):
        # Unsupervised top-3 PCA would drop the sole task-bearing coordinate.
        covariance = torch.diag(torch.tensor([10., 9., 8., 7., 6., 5., 4., 1.], dtype=torch.float64))
        direction = torch.tensor([0., 0., 0., 0., 0., 0., 0., 2.])
        basis = constrained_basis(covariance, direction, 3, torch)
        projection = (basis @ basis.T).numpy()
        expected = np.diag([1., 1., 0., 0., 0., 0., 0., 1.])
        np.testing.assert_allclose(projection, expected, atol=1e-6)
        samples = torch.linspace(-1., 1., 80).reshape(10, 8)
        actual = probabilities(samples, {'basis': basis, 'score_scale': torch.tensor(2.),
                                         'score_intercept': torch.tensor(.3)}, torch)
        expected_scores = 1. / (1. + np.exp(-(samples.numpy()[:, -1] * 2. + .3)))
        np.testing.assert_allclose(actual, expected_scores, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
