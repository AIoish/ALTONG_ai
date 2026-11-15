"""용도: 반응 정책의 원문 보존·멱등성·의미 있는 단문 제외와 새/옛 진단 구분을 검증한다.
생성일: 2026-11-15
"""

from copy import deepcopy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from filtering_training.common.relevance_policy import POLICY, apply_policy, targetless
from filtering_training.quality.audit_relevance_reactions import authored_contexts, make_probes
from filtering_training.common.artifact_io import ROOT, digest, load_lines, read, write
from filtering_training.preparation.curate_relevance import write_lines
from filtering_training.preparation.migrate_relevance_policy import migrate, rescore
from filtering_training.quality.verify_relevance_experiment import verify


class PolicyTests(unittest.TestCase):
    def test_all_contexts_and_apps_use_one_without_raw_text_change(self):
        contexts = authored_contexts()
        old = make_probes(contexts)
        new = make_probes(contexts, label_policy=POLICY)
        self.assertEqual([p['state'] for p in old], [p['state'] for p in new])
        for p in new:
            if targetless(p['body']):
                self.assertEqual(p['reference_score'], 1)
                self.assertEqual(p['label_status'], 'user_confirmed_rule_applied')
        self.assertTrue(any(p['context_id'] == 'approval_target' and p['body'] == '네' and p['reference_score'] == 2 for p in old))

    def test_policy_changes_only_score_and_adds_audit_metadata(self):
        rows = [{'id': 'x', 'score': 2, 'state': {'notification': {'body': '  네  ', 'title': '배포 승인'}}}]
        original = deepcopy(rows)
        new = apply_policy(rows)
        self.assertEqual(rows, original)
        self.assertEqual(new[0]['score'], 1)
        self.assertEqual(new[0]['label_policy']['previous_score'], 2)
        self.assertEqual(new[0]['state'], rows[0]['state'])
        self.assertEqual(apply_policy(new), new)
        self.assertEqual(apply_policy(rows, 'legacy'), rows)

    def test_meaningful_and_uncertain_short_messages_are_untouched(self):
        for body in ('빌드 실패', '배포 완료', '취소', '1', '', '사진을 보냈습니다.', '로그인 ㅋㅋ 오류', 'api 키 확인해주세요'):
            self.assertFalse(targetless(body), body)
            row = {'score': 3, 'state': {'notification': {'body': body}}}
            self.assertEqual(apply_policy([row]), [row])
        for body in ('ㅇ', 'ㅁ', 'ㅇㅇ', 'ㅇㅋ', 'ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ', 'ㅠㅠ', '아니', 'OK', '이모티콘을 보냈습니다.'):
            self.assertTrue(targetless(body), body)

    def test_unknown_policy_and_tampered_policy_score_rejected(self):
        with self.assertRaises(ValueError):
            apply_policy([], 'typo')
        with self.assertRaises(ValueError):
            apply_policy([{'score': 2, 'label_policy': {'version': POLICY}, 'state': {'notification': {'body': '네'}}}])

    def test_cached_rescore_requires_identical_model_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            dataset, run = folder / 'data', folder / 'run'
            (dataset / 'relevance').mkdir(parents=True)
            run.mkdir()
            row = {'id': 'x', 'score': 1, 'experiment': {'origin': 'pilot'}, 'state': authored_contexts()[0]['state']}
            row['state']['notification']['body'] = '네'
            write_lines(dataset / 'relevance/validation.jsonl', [row])
            write(dataset / 'manifest.json', {'label_policy': POLICY})
            (run / 'model.pt').write_bytes(b'checkpoint fixture')
            write(run / 'run_config.json', {'status': 'completed', 'output_classes': {'relevance': 3},
                  'dataset_dir': str(dataset), 'checkpoint_sha256': digest(run / 'model.pt')})
            write_lines(run / 'validation.predictions.jsonl', [{'id': 'x', 'gold': 1, 'predicted': 2}])
            write(run / 'validation.json', {})
            with patch('filtering_training.quality.verify_relevance_experiment.verify', return_value={}):
                report = rescore(dataset, run, folder / 'result.json')
                self.assertEqual(report['exact_correct'], 0)
                # A changed gold in the saved cache cannot be presented as a matching old run.
                write_lines(run / 'validation.predictions.jsonl', [{'id': 'x', 'gold': 2, 'predicted': 2}])
                with self.assertRaisesRegex(ValueError, 'unchanged original inputs'):
                    rescore(dataset, run, folder / 'bad.json')
                write_lines(run / 'validation.predictions.jsonl', [{'id': 'x', 'gold': 1, 'predicted': 2}])
                altered = folder / 'altered'
                (altered / 'relevance').mkdir(parents=True)
                write(altered / 'manifest.json', {'label_policy': POLICY})
                row['state']['notification']['title'] = 'changed notification input'
                write_lines(altered / 'relevance/validation.jsonl', [row])
                with self.assertRaisesRegex(ValueError, 'unchanged original inputs'):
                    rescore(altered, run, folder / 'bad-input.json')

    def test_migration_preserves_source_and_rejects_semantic_tampering(self):
        source = ROOT / 'filtering_training/outputs/v19_relevance_reviewed_reactions_01'
        if not (source / 'manifest.json').exists():
            self.skipTest('local source dataset unavailable')
        before = digest(source / 'relevance/train.jsonl')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'migrated'
            migrate(source, output)
            self.assertEqual(digest(source / 'relevance/train.jsonl'), before)
            with self.assertRaises(ValueError):
                migrate(source, output)
            rows_path = output / 'relevance/train.jsonl'
            rows = load_lines(rows_path)
            rows[0]['state']['notification']['body'] += ' changed'
            write_lines(rows_path, rows)
            manifest = read(output / 'manifest.json')
            manifest['files'][str(rows_path.relative_to(output))]['sha256'] = digest(rows_path)
            write(output / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'migration changed'):
                verify(output, write_report=False)


if __name__ == '__main__':
    unittest.main()
