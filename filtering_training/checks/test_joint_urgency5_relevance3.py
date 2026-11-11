"""용도: 공동 5/3 헤드 초기화·역전파와 변환된 통과 정책·평가를 검증한다.
생성일: 2026-11-03
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from filtering_training.modeling.joint_urgency5_relevance3 import (
    build_model, evaluate, initial_rank, map_relevance, passed, safety_rank,
)
from filtering_training.common.encoder_training import task_loss


class JointUrgencyRelevanceTests(unittest.TestCase):
    def test_policy_mapping_retains_original_decisions_for_every_pair(self):
        for urgency in range(1,6):
            for relevance in range(1,6):
                self.assertEqual(passed(urgency,map_relevance(relevance)),urgency>=4 or relevance>=4)
        for value in (0,6,True,'3',2.5):
            with self.assertRaises(ValueError):
                map_relevance(value)
        with self.assertRaises(ValueError):
            passed(4,4)

    def test_urgency_initialization_and_rng_match_original_both_heads_train(self):
        from filtering_training.modeling.encoder_score_experiment import build_model as original_model

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config=SimpleNamespace(hidden_size=8,model_type='roberta')
                self.embedding=torch.nn.Embedding(12,8)

            def forward(self,input_ids,attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        config=SimpleNamespace(hidden_size=8,model_type='roberta')
        with patch('transformers.AutoConfig.from_pretrained',return_value=config), \
             patch('transformers.AutoModel.from_pretrained',side_effect=lambda *a,**k:TinyEncoder()):
            torch.manual_seed(42)
            old=original_model()
            old_rng=torch.get_rng_state()
            torch.manual_seed(42)
            new=build_model()
            self.assertTrue(torch.equal(old_rng,torch.get_rng_state()))
        for key,value in old.encoder.state_dict().items():
            self.assertTrue(torch.equal(value,new.encoder.state_dict()[key]))
        self.assertTrue(torch.equal(old.heads[0].weight,new.heads[0].weight))
        self.assertTrue(torch.equal(old.heads[0].bias,new.heads[0].bias))
        x=torch.tensor([[1,2],[3,4]])
        outputs=new(x,torch.ones_like(x))
        self.assertEqual([tuple(y.shape) for y in outputs],[(2,5),(2,3)])
        loss=task_loss(outputs,torch.tensor([0,1]),torch.tensor([4,2]),torch.ones(2),2)
        loss.backward()
        for parameter in (new.heads[0].weight,new.heads[1].weight,new.encoder.embedding.weight):
            self.assertGreater(parameter.grad.abs().sum().item(),0)

    def test_task_loss_respects_original_task_weights_and_partial_batch(self):
        u=torch.tensor([[1.,0,0,0,0],[0,0,0,0,1.],[0,1.,0,0,0]],requires_grad=True)
        r=torch.tensor([[1.,0,0],[0,1.,0],[0,0,1.]],requires_grad=True)
        tasks=torch.tensor([0,1,0])
        labels=torch.tensor([0,1,1])
        actual=task_loss((u,r),tasks,labels,torch.tensor([1.5,.75]),3)
        expected=(torch.nn.functional.cross_entropy(u[[0,2]],labels[[0,2]],reduction='sum')*1.5+
                  torch.nn.functional.cross_entropy(r[[1]],labels[[1]],reduction='sum')*.75)/3
        self.assertTrue(torch.allclose(actual,expected))
        actual.backward()
        self.assertEqual(u.grad[1].abs().sum().item(),0)
        self.assertEqual(r.grad[[0,2]].abs().sum().item(),0)

    def test_metrics_distinguish_urgency_threshold_misses_and_policy_rescue(self):
        rows=[{'notification':{'id':'urgent'},'label':{'urgency_score':5,'relevance_score':4}},
              {'notification':{'id':'noise'},'label':{'urgency_score':1,'relevance_score':1}}]
        report=evaluate(rows,[{'urgency_score':3,'relevance_score':3},{'urgency_score':1,'relevance_score':2}])
        self.assertEqual(report['urgency_threshold_miss_ids'],['urgent'])
        self.assertEqual(report['urgent_policy_miss_ids'],[])
        self.assertEqual(report['policy_correct'],2)
        self.assertEqual(report['relevance_pass_correct'],2)
        self.assertEqual(initial_rank(report)[0],2)
        original=dict(report,policy_correct=65)
        self.assertEqual(safety_rank(original,report)[0],1)
        self.assertEqual(safety_rank(dict(original,policy_correct=64),report)[0],0)

    def test_absent_task_still_has_zero_gradients_for_original_adamw_decay(self):
        u=torch.randn(2,5,requires_grad=True)
        r=torch.randn(2,3,requires_grad=True)
        task_loss((u,r),torch.tensor([0,0]),torch.tensor([1,4]),torch.ones(2),2).backward()
        self.assertIsNotNone(r.grad)
        self.assertEqual(r.grad.abs().sum().item(),0)


if __name__=='__main__':
    unittest.main()
