import unittest
import torch
from src.evaluation.kumc_dictionary_cv import LinearBottleneck, task_loss


class RepresentationControls(unittest.TestCase):
    def test_linear_initialization_preserves_pca_projection(self):
        torch.manual_seed(83)
        basis = torch.linalg.qr(torch.randn(11, 4)).Q
        values = torch.randn(7, 11)
        model = LinearBottleneck(basis)
        codes = model.encode_inference(values)
        torch.testing.assert_close(codes, values @ basis)
        torch.testing.assert_close(model.decode(codes), (values @ basis) @ basis.T)

    def test_task_gradient_switch_keeps_head_learning(self):
        torch.manual_seed(19)
        model = LinearBottleneck(torch.linalg.qr(torch.randn(8, 3)).Q)
        head = torch.nn.Linear(3, 1)
        values, labels = torch.randn(12, 8), torch.arange(12).remainder(2).float()
        for enabled in [False, True]:
            model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
            loss = task_loss(model, head, values, labels, torch.zeros(3), torch.ones(3), enabled)
            loss.backward()
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)
            if enabled:
                self.assertGreater(float(model.encoder.weight.grad.abs().sum()), 0)
            else:
                self.assertIsNone(model.encoder.weight.grad)


if __name__ == '__main__':
    unittest.main()
