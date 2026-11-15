"""용도: 채택 한계·평가 세션 누출 차단·제안 없는 검수·명시적 답변만 반영되는지 검증한다.
생성일: 2026-11-15
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from filtering_training.common.artifact_io import digest, load_lines, read, write
from filtering_training.common.relevance_policy import POLICY
from filtering_training.preparation.curate_relevance import write_lines
from filtering_training.preparation.prepare_real_evaluation import partition, prepare, apply_answers, draft_plan
from filtering_training.preparation.prepare_relevance_experiment import load_sources
from filtering_training.quality.relevance_evaluation import compare_real, decide, metrics


def row(identifier, body):
    return {'id': identifier, 'score': None, 'source_number': identifier,
            'state': {'notification': {'app_name': 'Slack', 'title': '새 메시지', 'sender': '담당', 'body': body,
                                      'timestamp': '2026-11-15T00:00:00Z'},
                      'context': {'active_process': 'Code.exe', 'window_title': identifier, 'recent_windows': []}},
            'provenance': {'real_user_data': True, 'received_kst': '2026-11-15 09:00:00', 'quality_flags': []}}


class EvaluationTests(unittest.TestCase):
    def plan(self):
        return {'new_capture_confirmed': True, 'natural_capture_confirmed': True,
                'assignments': {'a': {'session_id': 'one', 'role': 'real_development'},
                                'b': {'session_id': 'two', 'role': 'real_final'}}}

    def rows(self):
        return [row('a', '음향 장비의 전원 연결 상태를 확인해 줘.'), row('b', '주말 산행 코스에서 통제된 구간을 찾아봤어.')]

    def test_half_percentage_point_and_relevant_error_guards(self):
        baseline = {'rows': 985, 'exact_accuracy': 971/985, 'relevant_misses': 1, 'false_relevant': 4}
        candidate = dict(baseline, exact_accuracy=967/985)
        self.assertEqual(decide(baseline, candidate)['development_gate'], 'pass')
        self.assertFalse(decide(baseline, candidate)['deployment_approved'])
        self.assertEqual(decide(baseline, dict(candidate, exact_accuracy=966/985))['development_gate'], 'reject')
        for change in ({'relevant_misses': 2}, {'false_relevant': 5}):
            self.assertEqual(decide(baseline, dict(candidate, **change))['development_gate'], 'reject')

    def test_metrics_keep_exact_scores_and_threshold_separate(self):
        rows = [dict(row(str(i), 'notice'), score=s, experiment={'origin': 'pilot'}) for i,s in enumerate((1,2,3))]
        report = metrics(rows, [2,1,3])
        self.assertEqual(report['confusion_1_2'], 2)
        self.assertEqual(report['relevant_misses'], 0)
        self.assertEqual(report['false_relevant'], 0)
        with self.assertRaises(ValueError):
            metrics(rows, [1, True, 3])

    def test_sessions_known_content_and_duplicate_inputs_cannot_cross_roles(self):
        rows, plan = self.rows(), self.plan()
        original = deepcopy(rows)
        result, _ = partition(rows, plan, [])
        self.assertEqual(rows, original)
        self.assertEqual({r['split'] for r in result}, {'real_development','real_final'})
        with self.assertRaises(ValueError):
            partition(rows, plan, [deepcopy(rows[0])])
        bad = deepcopy(plan);bad['assignments']['b']['session_id']='one'
        with self.assertRaises(ValueError):
            partition(rows, bad, [])
        repeated = deepcopy(rows);repeated[1]['state']['notification']['body']=repeated[0]['state']['notification']['body']
        with self.assertRaises(ValueError):
            partition(repeated, plan, [])
        bad = deepcopy(plan);bad['new_capture_confirmed']=False
        with self.assertRaises(ValueError):
            partition(rows, bad, [])

    def test_blind_review_roundtrip_and_training_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, plan, inventory = [folder/n for n in ('source.jsonl','plan.json','inventory.json')]
            write_lines(source, self.rows());write(plan, self.plan())
            known=folder/'known.jsonl';write_lines(known,[row('past','정수기 필터 교체 기록을 별도로 보관했어.')])
            write(inventory, {'entries': [{'path':str(known),'sha256':digest(known),'role':role}
                                         for role in ('training','development_validation')]})
            bundle = prepare(source, plan, inventory, folder/'prepared')
            packet = read(bundle/'review/packet.json')
            self.assertTrue(packet['blind'])
            self.assertTrue(all(c['suggested_score'] is None for c in packet['cases']))
            self.assertNotIn('내 제안:', (bundle/'review/검수카드.md').read_text(encoding='utf-8'))
            answers=folder/'answers.json'
            write(answers, {'packet_id':packet['packet_id'],'answers':{'R0001':{'score':2}}})
            apply_answers(bundle, answers, folder/'reviewed')
            self.assertEqual(len(load_lines(folder/'reviewed/real_development.jsonl')),1)
            pending=load_lines(folder/'reviewed/pending-or-held.jsonl')
            self.assertEqual(len(pending),1)
            self.assertNotIn('human_reviewed',pending[0]['evaluation_review'])
            empty=folder/'empty.jsonl';write_lines(empty,[])
            with self.assertRaisesRegex(ValueError,'evaluation-only'):
                load_sources(empty,empty,folder/'reviewed/real_development.jsonl')
            with self.assertRaises(ValueError):
                prepare(source, plan, inventory, bundle)

    def test_session_plan_has_no_preapproved_capture_or_split(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp);source=folder/'source.jsonl';target=folder/'plan.json'
            rows=self.rows()
            for r in rows:r['group_id']='session-'+r['id']
            write_lines(source,rows)
            draft_plan(source,target)
            plan=read(target)
            self.assertFalse(plan['new_capture_confirmed'])
            self.assertEqual(set(plan['sessions'].values()),{None})
            with self.assertRaises(ValueError):partition(rows,plan,[])
            plan.update(new_capture_confirmed=True,natural_capture_confirmed=True)
            plan['sessions']={'session-a':'real_development','session-b':'real_final'}
            self.assertEqual(len(partition(rows,plan,[])[0]),2)

    def test_real_model_selection_rejects_final_or_unreviewed_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path=folder/'rows.jsonl'
            for split, human in [('real_final',True),('real_development',False)]:
                item=self.rows()[0]
                item.update(split=split,score=2,evaluation_review={'human_reviewed':human})
                write_lines(path,[item]);write(folder/'manifest.json',{'policy':POLICY,'files':{path.name:digest(path)}})
                with self.assertRaisesRegex(ValueError,'only explicitly reviewed'):
                    compare_real(path,folder/'baseline',folder/'candidate',folder/'result')


if __name__ == '__main__':
    unittest.main()
