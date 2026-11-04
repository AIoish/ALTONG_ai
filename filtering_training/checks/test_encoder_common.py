"""용도: Qwen 의존성 분리·과거 입력 재현·앱과 일치하는 학습 입력의 무결성을 검증한다.
생성일: 2026-11-04
"""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from filtering_training.common.artifact_io import ROOT, digest, write
from filtering_training.common.encoder_data import DEFAULT_DATASET, load_windows_prepared, load_evaluation_inputs
from filtering_training.common.encoder_input import (
    INPUT_SCHEMA, WINDOWS_INPUT_SCHEMA, LEGACY_INPUT_SCHEMA, RECENT_WINDOWS_POLICY,
    input_schema_for_run, input_text, legacy_input_text, windows_only_state,
)
from filtering_training.common.validation_data import validation


class EncoderCommonTests(unittest.TestCase):
    def state(self):
        return {'notification': {'id': 'n_1', 'body': 'Please review this contract.'},
                'context': {'active_process': 'Code.exe', 'window_title': 'contract.py',
                            'recent_windows': [{'app_name': 'chrome.exe', 'window_title': 'contract guide'}],
                            'recent_processes': ['chrome.exe']}}

    def test_current_input_drops_only_redundant_model_field(self):
        state = self.state()
        original = deepcopy(state)
        modern = json.loads(input_text(state, 'relevance'))
        legacy = json.loads(legacy_input_text(state, 'relevance'))
        self.assertEqual(legacy.pop('recent_processes'), ['chrome.exe'])
        self.assertEqual(modern, legacy)
        self.assertNotIn('id', modern['notification'])
        self.assertEqual(state, original)
        self.assertEqual(json.loads(input_text(state, 'urgency')), {'notification': {'body': state['notification']['body']}})
        self.assertEqual(input_text(state, 'urgency'), legacy_input_text(state, 'urgency'))

    def test_run_schema_does_not_silently_change_existing_models(self):
        self.assertEqual(input_schema_for_run({}), LEGACY_INPUT_SCHEMA)
        self.assertEqual(input_schema_for_run({'input_schema': INPUT_SCHEMA}), INPUT_SCHEMA)
        with self.assertRaises(ValueError):
            input_schema_for_run({'input_schema': 'unknown'})
        with self.assertRaises(ValueError):
            input_text(self.state(), 'other')

    def test_current_training_imports_without_qwen_helpers(self):
        code = '''
import sys
from filtering_training.modeling import joint_urgency5_relevance3, encoder_score_experiment
from filtering_training.common import encoder_data, validation_data
for name in sys.modules:
    assert not name.startswith('filtering_training.legacy'), name
    assert not any(name.endswith('.' + old) for old in ('digit_score_experiment', 'digit_scores', 'score_tasks', 'score_loss')), name
assert 'src.filtering.prompt' not in sys.modules
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def fake_dataset(self, folder):
        original = {'urgency': [{'state': {'notification': {'body': 'danger'}}, 'expected': {'urgency': '4'}}],
                    'relevance': [{'state': self.state(), 'expected': {'relevance': '4'}}]}
        modern = dict(original['relevance'][0], state=windows_only_state(self.state()))
        mapped = [
            {'state': original['urgency'][0]['state'], 'task': 'urgency', 'original_score': 4, 'score': 4,
             'input_text': input_text(original['urgency'][0]['state'], 'urgency')},
            {'state': modern['state'], 'task': 'relevance', 'original_score': 4, 'score': 3,
             'input_text': input_text(modern['state'], 'relevance')},
        ]
        files = {'relevance/train.jsonl': [modern], 'mapped-training/initial.jsonl': mapped,
                 'mapped-training/safety.jsonl': mapped}
        manifest = {'removed_field': 'recent_processes', 'label_mapping_changed': False, 'files': {},
                    'input_schema': INPUT_SCHEMA, 'recent_windows_policy': RECENT_WINDOWS_POLICY}
        for name, rows in files.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            manifest['files'][name] = {'sha256': digest(path), 'rows': len(rows)}
        write(folder / 'manifest.json', manifest)
        return original, mapped, manifest

    def test_current_hash_and_serialized_input_are_both_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            original, mapped, manifest = self.fake_dataset(folder)
            with patch('filtering_training.common.encoder_data.load_native_prepared', return_value=(original, {})):
                sources, _, stages = load_windows_prepared(folder)
                self.assertNotIn('recent_processes', sources['relevance'][0]['state']['context'])
                self.assertEqual(len(stages['initial']), 2)
            path = folder / 'mapped-training/initial.jsonl'
            # Updating the manifest must not make a stale legacy input acceptable.
            mapped[1]['input_text'] = legacy_input_text(self.state(), 'relevance')
            path.write_text(''.join(json.dumps(row) + '\n' for row in mapped), encoding='utf-8')
            with patch('filtering_training.common.encoder_data.load_native_prepared', return_value=(original, {})):
                with self.assertRaisesRegex(ValueError, 'hash changed'):
                    load_windows_prepared(folder)
                manifest['files']['mapped-training/initial.jsonl']['sha256'] = digest(path)
                write(folder / 'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'input_text'):
                    load_windows_prepared(folder)

    def test_current_rejects_source_label_change_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            original, _, manifest = self.fake_dataset(folder)
            path = folder / 'relevance/train.jsonl'
            changed = json.loads(path.read_text(encoding='utf-8'))
            changed['expected']['relevance'] = '2'
            path.write_text(json.dumps(changed) + '\n', encoding='utf-8')
            manifest['files']['relevance/train.jsonl']['sha256'] = digest(path)
            write(folder / 'manifest.json', manifest)
            with patch('filtering_training.common.encoder_data.load_native_prepared', return_value=(original, {})):
                with self.assertRaisesRegex(ValueError, 'preserve original'):
                    load_windows_prepared(folder)

    @unittest.skipUnless((DEFAULT_DATASET / 'manifest.json').is_file(), 'local frozen datasets are not installed')
    def test_real_preparation_keeps_labels_replay_and_evaluation_split(self):
        from filtering_training.modeling.joint_urgency5_relevance3 import evaluation_sets, prepare
        initial, safety, evals, config = prepare()
        self.assertEqual(len(initial), 15119)
        self.assertEqual(len(safety), 8946)
        self.assertEqual(config['input_schema'], INPUT_SCHEMA)
        self.assertEqual(config['eval_counts'], {'validation': 74, 'safety_development': 40, 'personal100': 100, 'boundary14': 14})
        for record, task in initial + safety:
            if task == 'relevance':
                self.assertNotIn('recent_processes', record['state']['context'])
        legacy = evaluation_sets(schema=LEGACY_INPUT_SCHEMA)
        for name, rows in evals.items():
            self.assertEqual(rows, [windows_only_state(row) for row in legacy[name]])
        self.assertTrue(all(not overlap for overlap in config['normalized_body_overlap'].values()))

    @unittest.skipUnless((ROOT / 'filtering_training/outputs/v5_validation_pool_01/manifest.json').is_file(), 'local validation is not installed')
    def test_frozen_validation_labels_and_urgency_inputs_are_schema_invariant(self):
        old_samples, old_groups, old_texts = validation(schema=LEGACY_INPUT_SCHEMA)
        samples, groups, texts = validation()
        self.assertEqual(groups, old_groups)
        self.assertEqual([s.model_dump(mode='json') for s in samples], [s.model_dump(mode='json') for s in old_samples])
        for old_pair, pair in zip(old_texts, texts):
            self.assertEqual(old_pair[0], pair[0])
            original = json.loads(old_pair[1])
            original.pop('recent_processes')
            current = original['current_window']
            original['recent_windows'] = [w for w in original['recent_windows']
                if (w['app_name'], w['window_title']) != (current['active_process'], current['window_title'])][:3]
            self.assertEqual(original, json.loads(pair[1]))

    def test_previous_windows_matches_app_filter_before_take_and_exact_pair(self):
        state = self.state()
        current = {'app_name': 'Code.exe', 'window_title': 'contract.py'}
        expected = [
            {'app_name': 'Code.exe', 'window_title': 'other.py'},
            {'app_name': 'other.exe', 'window_title': 'contract.py'},
            {'app_name': 'code.exe', 'window_title': 'contract.py'},
        ]
        state['context']['recent_windows'] = [current, expected[0], current, *expected[1:],
                                               {'app_name': 'old.exe', 'window_title': 'old'}]
        original = deepcopy(state)
        modern = windows_only_state(state)
        self.assertEqual(modern['context']['recent_windows'], expected)
        self.assertEqual(json.loads(input_text(state, 'relevance'))['recent_windows'], expected)
        self.assertEqual(state, original)
        for schema in (WINDOWS_INPUT_SCHEMA, LEGACY_INPUT_SCHEMA):
            self.assertEqual(json.loads(input_text(state, 'relevance', schema=schema))['recent_windows'],
                             original['context']['recent_windows'])
            self.assertEqual(input_schema_for_run({'input_schema': schema}), schema)

    def test_converter_preserves_urgency_and_relevance_labels(self):
        from filtering_training.preparation.prepare_previous_windows import transform_record
        state = self.state()
        state['context']['recent_windows'].insert(0, {'app_name': 'Code.exe', 'window_title': 'contract.py'})
        state['context'].pop('recent_processes')
        for task in ('urgency', 'relevance'):
            value = state if task == 'relevance' else {'notification': state['notification']}
            row = {'task': task, 'state': value, 'original_score': 4, 'score': 3 if task == 'relevance' else 4,
                   'input_text': input_text(value, task, schema=WINDOWS_INPUT_SCHEMA)}
            result = transform_record(row)
            self.assertEqual(result['score'], row['score'])
            self.assertEqual(result['original_score'], row['original_score'])
            self.assertEqual(result['state']['notification'], row['state']['notification'])
            if task == 'urgency':
                self.assertEqual(result, row)
            else:
                self.assertEqual(len(result['state']['context']['recent_windows']), 1)
                self.assertEqual(result['input_text'], input_text(result['state'], task))

    def test_old_dataset_contract_is_rejected_for_new_training(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _, _, manifest = self.fake_dataset(folder)
            manifest['input_schema'] = WINDOWS_INPUT_SCHEMA
            write(folder / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'migrate v12'):
                load_windows_prepared(folder)

    def test_evaluation_hash_and_approved_labels_are_both_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _, _, manifest = self.fake_dataset(folder)
            rows = [{'notification': {'id': 'eval1', 'body': 'test'}, 'expected': {'relevance': 3}}]
            path = folder / 'evaluation/validation.jsonl'
            path.parent.mkdir()
            path.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
            manifest['files']['evaluation/validation.jsonl'] = {'sha256': digest(path), 'rows': 1}
            write(folder / 'manifest.json', manifest)
            loaded, _ = load_evaluation_inputs(folder, {'validation': rows})
            self.assertEqual(loaded['validation'], rows)
            changed = deepcopy(rows[0])
            changed['expected']['relevance'] = 1
            path.write_text(json.dumps(changed) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                load_evaluation_inputs(folder, {'validation': rows})
            manifest['files']['evaluation/validation.jsonl']['sha256'] = digest(path)
            write(folder / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'preserve approved'):
                load_evaluation_inputs(folder, {'validation': rows})


if __name__ == '__main__':
    unittest.main()
