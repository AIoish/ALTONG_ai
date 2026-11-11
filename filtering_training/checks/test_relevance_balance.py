"""용도: 관련도 구성 개선이 대비 묶음·희귀 작업·필수 창 근거·고정 예외를 보존하는지 검증한다.
생성일: 2026-11-08
"""

from collections import Counter
from copy import deepcopy
import unittest

from filtering_training.preparation.balance_relevance import (
    APPS, diversify_apps, make_history, minimum_history, rebalance, select_groups, window_catalogue,
)
from filtering_training.preparation.curate_relevance import body_group
from filtering_training.review.review_relevance_dataset import Reviewer


class RelevanceBalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reviewer = Reviewer()

    def row(self, task=0, event='waiting', *, current=0, history=None, identifier=None):
        reviewer = self.reviewer
        recipe = reviewer.tasks[task]
        body = reviewer.definitions['TRAINING_EVENT_PHRASINGS'][event][0].format(**recipe)
        c = reviewer.tasks[current] if current is not None else None
        windows = [{'app_name': reviewer.tasks[i]['process'], 'window_title': reviewer.tasks[i]['goal'] + ' - Reference'}
                   for i in (history or [])]
        row = {'id': identifier or f'{task}_{event}_{current}', 'group_id': body_group(body),
               'split': 'train_only', 'source_score5': 5,
               'state': {'notification': {'app_name': 'Slack', 'sender': recipe['domain'] + ' 담당',
                                         'title': recipe['domain'] + ' 소식', 'body': body,
                                         'timestamp': '2026-11-08T03:00:00Z'},
                         'context': {'active_process': c['process'] if c else '',
                                     'window_title': c['goal'] if c else '', 'recent_windows': windows}}}
        score, rule, reason, method = reviewer.decide(row)
        self.assertIsNotNone(score)
        row['score'] = score
        row['review'] = {'method': method, 'rule': rule, 'reason': reason}
        return row

    def test_group_sampling_preserves_whole_context_contrasts_and_all_tasks(self):
        rows = []
        for event in ('waiting', 'draft'):
            for task in range(5):
                rows.extend([self.row(task, event, current=task), self.row(task, event, current=10)])
        kept, excluded, report = select_groups(rows, self.reviewer, cap=1)
        self.assertLess(len(kept), len(rows))
        selected = Counter(r['group_id'] for r in kept)
        self.assertTrue(all(count == 2 for count in selected.values()))
        self.assertEqual({r['id'] for r in kept} | {r['id'] for r in excluded}, {r['id'] for r in rows})
        tasks = set().union(*(self.reviewer.targets(r['state']['context']['window_title']) for r in kept))
        self.assertTrue(set(range(5)).issubset(tasks))
        self.assertEqual(report['wording_coverage'], 2)

    def test_history_dependent_positive_cannot_lose_its_evidence(self):
        row = self.row(current=None, history=[0, 10, 15])
        required = minimum_history(row, self.reviewer)
        self.assertEqual(required, (0,))
        catalogue = window_catalogue([row, self.row(10, current=10), self.row(15, current=15)], self.reviewer)
        self.assertIsNone(make_history(row, required, 0, catalogue, self.reviewer, 42))
        history = make_history(row, required, 1, catalogue, self.reviewer, 42)
        self.assertEqual(history, row['state']['context']['recent_windows'][:1])

    def test_clear_current_needs_no_history_and_extra_history_is_synthetic(self):
        row = self.row(current=0, history=[0])
        required = minimum_history(row, self.reviewer)
        self.assertEqual(required, ())
        donors = [self.row(10, current=10), self.row(15, current=15), self.row(20, current=20)]
        catalogue = window_catalogue([row] + donors, self.reviewer)
        history = make_history(row, required, 3, catalogue, self.reviewer, 42)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0], row['state']['context']['recent_windows'][0])
        self.assertTrue(all(w['window_title'] != row['state']['context']['window_title'] or
                            w['app_name'] != row['state']['context']['active_process'] for w in history))

    def test_missing_current_negative_can_get_only_unrelated_history(self):
        row = self.row(current=None, history=[])
        self.assertEqual(row['score'], 1)
        donors = [self.row(10, current=10), self.row(15, current=15)]
        catalogue = window_catalogue(donors, self.reviewer)
        history = make_history(row, (), 2, catalogue, self.reviewer, 42)
        self.assertEqual(len(history), 2)
        changed = deepcopy(row)
        changed['state']['context']['recent_windows'] = history
        self.assertEqual(self.reviewer.decide(changed)[0], 1)

    def test_app_balance_changes_no_other_notification_fields(self):
        rows = [self.row(i % 5, current=i % 5, identifier=str(i)) for i in range(12)]
        before = deepcopy(rows)
        changed = diversify_apps(rows, 42)
        counts = Counter(r['state']['notification']['app_name'] for r in changed)
        self.assertEqual(counts, Counter({app: 4 for app in APPS}))
        for a, b in zip(rows, changed):
            b = deepcopy(b)
            b['state']['notification']['app_name'] = a['state']['notification']['app_name']
            self.assertEqual(a, b)
        self.assertEqual(rows, before)

    def test_explicit_app_features_and_fixed_cases_are_not_reassigned(self):
        fixed = self.row(identifier='fixed')
        fixed['review']['method'] = 'individual_case_read'
        app_specific = self.row(identifier='thread')
        app_specific['state']['notification']['body'] = 'Slack 스레드에서 이 작업을 확인해 주세요.'
        changed = diversify_apps([fixed, app_specific], 42)
        self.assertEqual(changed, [fixed, app_specific])

    def test_end_to_end_is_reproducible_and_preserves_labels_and_current_windows(self):
        rows = []
        for event in ('waiting', 'draft'):
            for task in range(5):
                rows.extend([self.row(task, event, current=task), self.row(task, event, current=10)])
        rows[0]['review']['method'] = 'individual_case_read'
        before = deepcopy(rows)
        result = rebalance(rows, self.reviewer, cap=1, seed=42)
        self.assertEqual(result, rebalance(rows, self.reviewer, cap=1, seed=42))
        output, excluded, lineage, selection = result
        source = {r['id']: r for r in rows}
        for row in output:
            parent = source[row['augmentation']['source_id']]
            self.assertEqual(row['score'], parent['score'])
            self.assertEqual(row['state']['context']['window_title'], parent['state']['context']['window_title'])
            notification = deepcopy(row['state']['notification'])
            notification['app_name'] = parent['state']['notification']['app_name']
            self.assertEqual(notification, parent['state']['notification'])
            self.assertFalse(row['training_eligible'])
        fixed = next(r for r in output if r['augmentation']['source_id'] == rows[0]['id'])
        self.assertEqual(fixed['state'], rows[0]['state'])
        self.assertEqual(len(output) + len(excluded), len(rows))
        self.assertEqual(rows, before)


if __name__ == '__main__':
    unittest.main()
