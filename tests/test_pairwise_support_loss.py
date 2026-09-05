import unittest
import torch
from src.evaluation.pairwise_support_loss import pairwise_support_loss


class PairwiseLossTests(unittest.TestCase):
    def test_frame_offsets_cancel_and_gradient_is_antisymmetric(self):
        scores=torch.tensor([.4,-.2,-.5,.8],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([1.,0.,1.,0.]);ids=torch.tensor([2,5,201,210])
        loss=pairwise_support_loss(scores,y,ids)
        shifted=pairwise_support_loss(scores+torch.tensor([5.,5.,-3.,-3.]),y,ids)
        self.assertAlmostEqual(float(loss.detach()),float(shifted.detach()),places=12)
        gradient=torch.autograd.grad(loss,scores)[0]
        torch.testing.assert_close(gradient[0::2],-gradient[1::2])
        self.assertTrue(bool((gradient[0::2]<0).all()))

    def test_rejects_wrong_label_order_and_cross_frame_pairs(self):
        s=torch.zeros(2)
        with self.assertRaises(ValueError):pairwise_support_loss(s,torch.tensor([0.,1.]),torch.tensor([0,1]))
        with self.assertRaises(ValueError):pairwise_support_loss(s,torch.tensor([1.,0.]),torch.tensor([0,196]))


if __name__=='__main__':unittest.main()
