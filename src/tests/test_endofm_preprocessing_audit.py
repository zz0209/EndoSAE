import unittest
from pathlib import Path

from src.endofm_preprocessing_audit import audit_preprocessing_source


ROOT = Path(__file__).resolve().parents[2]
ENDO = ROOT / "third_party" / "Endo-FM"


class EndoFMPreprocessingAuditTests(unittest.TestCase):
    def test_pinned_reference_input_path(self):
        result = audit_preprocessing_source(
            ENDO / "eval_finetune.py",
            ENDO / "datasets" / "ucf101.py",
            ENDO / "datasets" / "data_utils.py",
            ENDO / "datasets" / "decoder.py",
            ENDO / "utils" / "defaults.py",
            ENDO / "models" / "configs" / "Kinetics" / "TimeSformer_divST_8x32_224.yaml",
        )
        self.assertEqual(result.num_frames, 8)
        self.assertEqual(result.sampling_rate, 32)
        self.assertEqual(result.test_crop_size, 224)
        self.assertEqual(result.target_fps, 30)
        self.assertEqual(result.decoding_backend, "pyav")
        self.assertEqual(result.mean, [0.45, 0.45, 0.45])
        self.assertEqual(result.std, [0.225, 0.225, 0.225])
        self.assertTrue(result.uint8_scaled_by_255)
        self.assertTrue(result.normalize_before_permute)
        self.assertTrue(result.deterministic_uniform_crop)
        self.assertTrue(result.validation_config_mutation_hazard)
        self.assertEqual(len(result.source_sha256), 6)


if __name__ == "__main__":
    unittest.main()
