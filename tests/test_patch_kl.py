import unittest
import torch
from lib.loss.loss_contrast_mem import ContrastChangeLoss


def loss(student, teacher, mask):
    return ContrastChangeLoss.patch_algin_loss(None, student, (None, teacher), mask)


class PatchKL(unittest.TestCase):
    def test_direction_mask_and_student_gradient(self):
        s=torch.tensor([[[1.2,-0.3,0.4],[0.1,2.0,-1.0]]],requires_grad=True)
        t=torch.tensor([[[-0.5,0.7,1.0],[4.0,-2.0,0.0]]],requires_grad=True)
        actual=loss(s,t,torch.tensor([[[0.,1.]]]))
        expected=torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=s[:,0]),
            torch.distributions.Categorical(logits=t[:,0].detach())).mean()
        torch.testing.assert_close(actual,expected)
        grad=torch.autograd.grad(actual,s,retain_graph=True)[0]
        expected_grad=torch.autograd.grad(expected,s)[0]
        torch.testing.assert_close(grad,expected_grad)
        actual.backward()
        self.assertIsNone(t.grad)
        self.assertGreater(float(s.grad[:,0].abs().sum()),0)
        self.assertEqual(float(s.grad[:,1].abs().sum()),0)

    def test_equal_distributions_and_all_masked(self):
        x=torch.randn(2,4,7,requires_grad=True)
        torch.testing.assert_close(loss(x,x.detach(),torch.zeros(2,2,2)),torch.tensor(0.))
        result=loss(x,torch.randn_like(x),torch.ones(2,2,2))
        result.backward()
        self.assertEqual(result.item(),0.)
        self.assertEqual(x.grad.abs().sum().item(),0.)

    def test_half_precision_extreme_logits_are_finite(self):
        s=torch.tensor([[[1000.,-1000.,0.]]],dtype=torch.float16,requires_grad=True)
        t=-s.detach()
        result=loss(s,t,torch.zeros(1,1,1))
        self.assertTrue(torch.isfinite(result))
        self.assertEqual(result.dtype,torch.float32)
        result.backward()
        self.assertTrue(torch.isfinite(s.grad).all())


if __name__=='__main__': unittest.main()
