import unittest
from pathlib import Path

from src.endofm_source_audit import EXPECTED_DEFAULTS, audit_timesformer_source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMESFORMER_SOURCE = PROJECT_ROOT / "third_party" / "Endo-FM" / "models" / "timesformer.py"


class EndoFMSourceAuditTests(unittest.TestCase):
    def test_pinned_architecture_defaults_match(self):
        result = audit_timesformer_source(TIMESFORMER_SOURCE)
        self.assertTrue(result.defaults_match)
        self.assertEqual(result.defaults, EXPECTED_DEFAULTS)
        self.assertTrue(result.has_forward_features)

    def test_intermediate_helper_is_not_layerwise(self):
        result = audit_timesformer_source(TIMESFORMER_SOURCE)
        self.assertTrue(result.has_get_intermediate_layers)
        self.assertTrue(result.intermediate_returns_single_final_tensor)

    def test_attention_helper_references_missing_method(self):
        result = audit_timesformer_source(TIMESFORMER_SOURCE)
        self.assertTrue(result.has_get_last_selfattention)
        self.assertFalse(result.has_prepare_tokens)
        self.assertTrue(result.last_attention_calls_missing_prepare_tokens)

    def test_forward_api_has_no_padding_mask_and_layout_is_explicit(self):
        result = audit_timesformer_source(TIMESFORMER_SOURCE)
        self.assertFalse(result.attention_accepts_mask)
        self.assertFalse(result.block_accepts_mask)
        self.assertFalse(result.forward_features_accepts_mask)
        self.assertTrue(result.patch_embed_expects_bcthw)
        self.assertTrue(result.patch_tokens_are_hwt_ordered)


if __name__ == "__main__":
    unittest.main()
