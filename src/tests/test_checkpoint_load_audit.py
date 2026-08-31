import unittest
from pathlib import Path

from src.checkpoint_load_audit import audit_checkpoint_loading


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "third_party" / "Endo-FM" / "eval_finetune.py"
HELPERS = ROOT / "third_party" / "Endo-FM" / "models" / "helpers.py"


class CheckpointLoadAuditTests(unittest.TestCase):
    def test_pinned_loader_has_partial_load_and_gpu_assumptions(self):
        result = audit_checkpoint_loading(EVAL, HELPERS)
        self.assertTrue(result.eval_uses_torch_load)
        self.assertTrue(result.eval_loads_on_cpu)
        self.assertTrue(result.eval_filters_backbone_prefix)
        self.assertTrue(result.eval_uses_non_strict_state_dict)
        self.assertTrue(result.eval_calls_cuda_unconditionally)
        self.assertTrue(result.eval_wraps_ddp)
        self.assertTrue(result.helper_uses_non_strict_state_dict)
        self.assertTrue(result.helper_has_bare_checkpoint_fallback)


if __name__ == "__main__":
    unittest.main()
