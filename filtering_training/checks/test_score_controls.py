"""용도: 숫자 가중치의 실제 gradient, prompt 마스킹과 균등 샘플링 예산 검증.
생성일: 2026-10-04
"""

from collections import Counter
import unittest

from filtering_training.common.score_loss import balanced_indices, weighted_score_loss


class ScoreControlTests(unittest.TestCase):
    def test_balancing_preserves_steps_and_uses_only_source_indices(self):
        scores=[1]*100+[2]*3+[3]*6+[4]*12+[5]*29
        indices=balanced_indices(scores)
        self.assertEqual(len(indices),150)
        self.assertEqual(Counter(scores[i] for i in indices),{s:30 for s in range(1,6)})
        self.assertTrue(all(i in range(len(scores)) for i in indices))
        self.assertEqual(indices,balanced_indices(scores))
        with self.assertRaises(ValueError):
            balanced_indices([1,1,5])

    def test_numeric_weight_changes_only_numeric_gradient_and_masks_prompt(self):
        import torch
        logits=torch.zeros((1,5,10),requires_grad=True)
        labels=torch.tensor([[-100,-100,8,4,9]])
        loss=weighted_score_loss(logits,labels,[1,2,3,4,5],7)
        loss.backward()
        # Targets at positions 2/3/4 are predicted from logits at positions 1/2/3.
        self.assertEqual(float(logits.grad[0,0].abs().sum()),0)
        self.assertEqual(float(logits.grad[0,4].abs().sum()),0)
        self.assertAlmostEqual(float(logits.grad[0,2,4]/logits.grad[0,1,8]),7,places=5)

    def test_unit_weight_matches_masked_causal_cross_entropy(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(42)
        logits=torch.randn((1,5,10),requires_grad=True)
        labels=torch.tensor([[-100,-100,8,4,9]])
        expected=F.cross_entropy(logits[:,:-1].reshape(-1,10),labels[:,1:].reshape(-1),ignore_index=-100)
        self.assertTrue(torch.allclose(weighted_score_loss(logits,labels,[1,2,3,4,5],1),expected))
        with self.assertRaises(ValueError):
            weighted_score_loss(logits,torch.tensor([[-100,-100,4,5,9]]),[1,2,3,4,5])


if __name__=="__main__":
    unittest.main()
