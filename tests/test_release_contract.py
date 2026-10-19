import copy
import unittest
import torch
from torch import nn
from lib.models.decoder.UperHead import UperHead
from test import metrics


class ReleaseContract(unittest.TestCase):
    def test_shared_norm_matches_historical_feature_transformation_and_gradient(self):
        # Avoid PPM allocation: exercise the actual pyramid transformation method.
        head = UperHead.__new__(UperHead)
        nn.Module.__init__(head)
        head.feature_normalization = 'shared_chunk_norm1'
        head.norm1 = nn.LayerNorm(4)
        for name in ('up1', 'up2', 'up3', 'up4'):
            setattr(head, name, nn.Identity())
        with torch.no_grad():
            head.norm1.weight.copy_(torch.tensor([0.3, 0.8, 1.4, 1.7]))
        reference_norm = copy.deepcopy(head.norm1)
        x = torch.randn(2, 4, 16, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        actual = head.build_pyramid_features([x])
        expected = [reference_norm(c).transpose(1, 2).reshape(2, 4, 2, 2)
                    for c in ref_x.chunk(4, dim=-1)]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        sum(t.square().sum() for t in actual).backward()
        sum(t.square().sum() for t in expected).backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(head.norm1.weight.grad, reference_norm.weight.grad)

    def test_change_iou_is_not_miou(self):
        m = metrics([[96544638, 338945], [536252, 1953971]])
        self.assertAlmostEqual(m['f1'], 81.70245522866887)
        self.assertAlmostEqual(m['iou_change'], 69.06521634628979)
        self.assertAlmostEqual(m['miou'], 84.08341987386271)
        self.assertGreater(m['miou'], m['iou_change'])


if __name__ == '__main__':
    unittest.main()
