import unittest
import torch
from src.sae.baselines import TopKAutoencoder
from src.sae.low_rank_topk import LowRankTopK


class LowRankUpdate(unittest.TestCase):
    def test_identity_base_freeze_and_materialization(self):
        torch.manual_seed(71)
        base = TopKAutoencoder(8, 12, 3)
        x = torch.randn(20, 8)
        expected = base.encode_inference(x).detach()
        model = LowRankTopK(base, 2)
        torch.testing.assert_close(model.encode_inference(x), expected, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 80)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        code = model.encode_inference(x)
        loss = (model.decode(code) - x).square().mean() + code.sum() * .001
        loss.backward(); optimizer.step()
        self.assertTrue(model.base_unchanged())
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        merged = TopKAutoencoder(8, 12, 3)
        merged.load_state_dict(model.materialized_state(), strict=True)
        torch.testing.assert_close(merged.encode_inference(x), model.encode_inference(x))
        torch.testing.assert_close(merged.decode(merged.encode_inference(x)), model.decode(model.encode_inference(x)))
        torch.testing.assert_close(model.effective_decoder().norm(dim=0), torch.ones(12))


if __name__ == '__main__':
    unittest.main()
