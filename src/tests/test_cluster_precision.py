import unittest

from src.cluster_precision import all_successes_clopper_pearson_lower


class ClusterPrecisionTests(unittest.TestCase):
    def test_three_and_fifteen_cluster_bounds(self):
        self.assertAlmostEqual(all_successes_clopper_pearson_lower(3), 0.2924017738)
        self.assertAlmostEqual(all_successes_clopper_pearson_lower(15), 0.7819806391)

    def test_invalid_inputs_are_rejected(self):
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    all_successes_clopper_pearson_lower(value)
        with self.assertRaises(ValueError):
            all_successes_clopper_pearson_lower(3, confidence=1.0)


if __name__ == "__main__":
    unittest.main()
