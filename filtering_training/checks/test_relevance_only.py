"""용도: 관련도 단일 헤드·누적 배치·후보 사용 구분·체크포인트 복원과 지표를 검증한다.
생성일: 2026-11-11
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from filtering_training.common.artifact_io import digest, read
from filtering_training.common.encoder_input import INPUT_SCHEMA, encode_checked, input_text
from filtering_training.common.encoder_model import build_encoder
from filtering_training.common.encoder_training import train_epoch, predict_logits
from filtering_training.modeling.relevance_only import (
    DEFAULTS, candidate_permission, check_settings, evaluate, evaluate_run, origin_weights, rank, train,
)
from filtering_training.preparation.curate_relevance import write_lines


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, 'a': 2, 'b': 3, 'c': 4}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]',
                                  model_input_names=['input_ids', 'attention_mask'])


def row(number, score):
    return {'id': f'row-{number}', 'score': score, 'experiment_group': f'group-{number}',
            'experiment': {'origin': 'pilot', 'label_source': 'assistant'},
            'state': {'notification': {'app_name': 'chat', 'sender': 'sender', 'title': 'a',
                                      'body': 'b', 'timestamp': '2026-11-11T00:00:00Z'},
                      'context': {'active_process': 'Code.exe', 'window_title': 'c', 'recent_windows': []}}}


class RelevanceOnlyTests(unittest.TestCase):
    def test_candidate_data_requires_explicit_experimental_option(self):
        manifest = {'input_schema': INPUT_SCHEMA, 'training_ready': False}
        with self.assertRaisesRegex(ValueError, 'allow-candidate-labels'):
            candidate_permission(manifest, False)
        self.assertTrue(candidate_permission(manifest, True))
        self.assertFalse(candidate_permission(dict(manifest, training_ready=True), False))
        with self.assertRaises(ValueError):
            candidate_permission(dict(manifest, input_schema='old'), True)

    def test_metrics_and_selection_do_not_use_urgency(self):
        rows = [row(1, 1), row(2, 2), row(3, 3)]
        report = evaluate(rows, [1, 3, 2])
        self.assertAlmostEqual(report['exact_accuracy'], 1 / 3)
        self.assertAlmostEqual(report['mae'], 2 / 3)
        self.assertEqual(report['relevant_miss_ids'], ['row-3'])
        self.assertEqual(report['false_relevant_ids'], ['row-2'])
        self.assertNotIn('urgency', report)
        better_f1 = dict(report, macro_f1=.9, exact_accuracy=.4)
        better_accuracy = dict(report, macro_f1=.8, exact_accuracy=.8)
        self.assertGreater(rank(better_f1), rank(better_accuracy))
        self.assertNotIn('score', __import__('json').loads(input_text(rows[0]['state'], 'relevance')))
        for predictions in ([1, 2], [1, True, 3], [1, 2, 4]):
            with self.assertRaises(ValueError):
                evaluate(rows, predictions)

    def test_bad_training_settings_rejected(self):
        for change in ({'epochs': 0}, {'micro_batch': 17}, {'head_lr': float('nan')}, {'warmup_ratio': 1}):
            with self.assertRaises(ValueError):
                check_settings(dict(DEFAULTS, **change))

    def test_equal_source_selection_does_not_hide_new_scenario_regression(self):
        a = {'macro_f1': .99, 'exact_accuracy': .99, 'mae': .01,
             'by_origin': {'balanced': {'macro_f1': 1.}, 'pilot': {'macro_f1': .5}}}
        b = {'macro_f1': .98, 'exact_accuracy': .98, 'mae': .02,
             'by_origin': {'balanced': {'macro_f1': .98}, 'pilot': {'macro_f1': .8}}}
        self.assertGreater(rank(b), rank(a))
        rows = [row(i, 1) for i in range(10)]
        rows[0]['experiment']['origin'] = 'real'
        weights = origin_weights(rows)
        self.assertAlmostEqual(weights['real'], 5)
        self.assertAlmostEqual(sum(weights[r['experiment']['origin']] for r in rows), len(rows))
        reviewed = row(20, 1)
        reviewed['experiment'].update(origin='reviewed', loss_origin='pilot')
        weights = origin_weights(rows + [reviewed])
        self.assertNotIn('reviewed', weights)
        self.assertIn('pilot', weights)

    def test_accumulation_matches_full_batch_including_tail(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(5, 3)

            def forward(self, input_ids, attention_mask):
                return (self.embedding(input_ids[:, 0]),)

        torch.manual_seed(12)
        small, large = Linear(), Linear()
        large.load_state_dict(small.state_dict())
        records = [({'input_ids': [2 + i % 3], 'attention_mask': [1]}, 0, i % 3) for i in range(6)]
        for model, micro in ((small, 1), (large, 4)):
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            result = train_epoch(model, tokenizer(), records, optimizer=optimizer, scheduler=scheduler,
                                 scaler=torch.amp.GradScaler('cuda', enabled=False), weights=torch.ones(1),
                                 seed=42, device='cpu', effective_batch=4, micro_batch=micro,
                                 row_weights=[1., 3., 1., 2., 1., 2.])
            self.assertEqual(result['successful_updates'], 2)
        self.assertTrue(torch.allclose(small.embedding.weight, large.embedding.weight, atol=1e-7))
        outputs = predict_logits(small, tokenizer(), [r[0] for r in records], device='cpu', batch_size=2)
        self.assertEqual([tuple(x.shape) for x in outputs], [(6, 3)])

    def test_only_one_three_class_head_trains_saves_and_reloads(self):
        from transformers import RobertaConfig
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            base = folder / 'base'
            config = RobertaConfig(vocab_size=5, hidden_size=8, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=16, max_position_embeddings=514,
                                   pad_token_id=1, bos_token_id=0, eos_token_id=0)
            config.save_pretrained(base)
            tok = tokenizer()
            rows = {'train': [row(i, i % 3 + 1) for i in range(6)],
                    'validation': [row(i + 10, i % 3 + 1) for i in range(3)]}
            for sample in rows['train'][:2]:
                sample['experiment'].update(origin='reviewed', loss_origin='pilot')
            source = deepcopy(rows)
            dataset = folder / 'dataset'
            write_lines(dataset / 'relevance/validation.jsonl', rows['validation'])
            encoded = {split: encode_checked(tok, [input_text(r['state'], 'relevance') for r in items])[0]
                       for split, items in rows.items()}
            run = folder / 'run'
            cfg = {'base_dir': str(base), 'dataset_dir': str(dataset), 'protected_hashes': {},
                   'output_classes': {'relevance': 3}, 'candidate_labels_used': True}
            settings = dict(DEFAULTS, device='cpu', epochs=1, effective_batch=4, micro_batch=2)
            def tiny(**kwargs):
                kwargs['pretrained'] = False
                return build_encoder(**kwargs)
            with patch('filtering_training.modeling.relevance_only.build_encoder', side_effect=tiny):
                result = train(run, rows, encoded, tok, cfg, settings)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['successful_updates'], 2)
            self.assertEqual(result['checkpoint_sha256'], digest(run / 'model.pt'))
            self.assertEqual(read(run / 'reviewed-training.json')['rows'], 2)
            self.assertEqual(read(run / 'reviewed-training.json')['scope'], 'training_fit_only_not_validation_or_selection')
            state = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
            self.assertEqual(tuple(state['heads.0.weight'].shape), (3, 8))
            self.assertFalse(any('heads.1' in name for name in state))
            report = evaluate_run(run, device='cpu', batch_size=2)
            self.assertEqual(report, read(run / 'validation.json'))
            self.assertEqual(rows, source)
            with self.assertRaises(ValueError):
                train(run, rows, encoded, tok, cfg, settings)
            with (run / 'model.pt').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
                evaluate_run(run)


if __name__ == '__main__':
    unittest.main()
