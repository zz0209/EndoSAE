import unittest
import numpy as np
from src.evaluation.kumc_localization import support_frame, localization_metrics
from src.evaluation.realcolon_task import project_boxes


class Coordinates(unittest.TestCase):
    def mask(self, box):
        return project_boxes(support_frame({'width': 280, 'height': 224,
            'objects': [{'name': 'object', 'raw_box': box}]}))[0]

    def test_asymmetric_xy_and_crop(self):
        self.assertEqual(np.flatnonzero(self.mask([200, 32, 215, 47])).tolist(), [39])

    def test_inclusive_single_pixel_and_full_extent(self):
        self.assertEqual(np.flatnonzero(self.mask([251, 223, 251, 223])).tolist(), [195])
        self.assertEqual(int(self.mask([0, 0, 279, 223]).sum()), 196)

    def test_crop_dropped_is_empty_support(self):
        self.assertFalse(self.mask([0, 0, 5, 223]).any())

    def test_tied_maxima_and_missing_support(self):
        target = np.zeros((1, 8, 196), dtype=bool)
        target[0, 0, :49] = True
        result = localization_metrics(np.ones(target.shape), target, [{'video_id': 'fixture'}])['fixture']
        self.assertEqual(result['eligible_frames'], 1)
        self.assertEqual(result['frame_mean_patch_ap'], .25)
        self.assertEqual(result['pointing_hit_rate'], .25)


if __name__ == '__main__':
    unittest.main()
