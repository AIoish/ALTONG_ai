"""용도: 관련도 후보 분할에서 정답 충돌·대비 묶음 누출·실제 자료의 검증 유입을 막는다.
생성일: 2026-11-11
"""

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from filtering_training.preparation.curate_relevance import write_lines
from filtering_training.preparation.prepare_relevance_experiment import (
    assign_splits, audit_rows, group_rows, prepare, verify_splits,
)


def row(identifier, score=1, *, body=None, group=None, train_only=False):
    return {'id': identifier, 'group_id': group or identifier, 'score': score,
            'state': {'notification': {'app_name': 'Slack', 'sender': 'sender', 'title': identifier,
                                       'body': body if body is not None else identifier,
                                       'timestamp': '2026-11-11T00:00:00Z'},
                      'context': {'active_process': 'Code.exe', 'window_title': 'current', 'recent_windows': []}},
            'experiment': {'group_hints': [group or identifier], 'train_only': train_only}}


class RelevanceExperimentTests(unittest.TestCase):
    def fixture(self):
        return [row(f'example-{score}-{i}', score) for score in (1, 2, 3) for i in range(8)]

    def test_identical_input_different_label_is_a_conflict(self):
        a = row('a', 1)
        b = deepcopy(a)
        b.update(id='b', score=3)
        self.assertEqual(len(audit_rows([a, b])['exact_input_label_conflicts']), 1)

    def test_sender_time_only_changes_do_not_hide_content_window_warning(self):
        a = row('a', 1)
        b = deepcopy(a)
        b.update(id='b', score=3)
        b['state']['notification']['sender'] = 'other'
        report = audit_rows([a, b])
        self.assertFalse(report['exact_input_label_conflicts'])
        self.assertEqual(len(report['content_window_score_warnings']), 1)

    def test_same_body_in_different_context_may_have_different_labels(self):
        a, b = row('a', 1, body='same'), row('b', 3, body='same')
        b['state']['context']['window_title'] = 'different current task'
        self.assertFalse(audit_rows([a, b])['exact_input_label_conflicts'])
        groups, _ = group_rows([a, b])
        self.assertEqual(groups['a'], groups['b'])

    def test_history_app_variants_and_whole_scenarios_are_grouped(self):
        rows = [row('a', 1, group='scenario'), row('b', 2, group='scenario')]
        groups, _ = group_rows(rows)
        self.assertEqual(groups['a'], groups['b'])

    def test_near_duplicate_bodies_cannot_be_separated(self):
        a = row('a', body='The shared report document version has been saved in the project archive folder.')
        b = row('b', body='The shared report document version has been saved in the project archive folders.')
        groups, near = group_rows([a, b], similarity=0.8)
        self.assertTrue(near)
        self.assertEqual(groups['a'], groups['b'])

    def test_real_group_and_linked_synthetic_body_stay_in_train(self):
        rows = self.fixture() + [row('real', 2, body='linked', train_only=True), row('synthetic', 1, body='linked')]
        groups, near = group_rows(rows)
        result = assign_splits(rows, groups)
        by_id = {r['id']: r for r in result}
        self.assertEqual(by_id['real']['split'], 'train')
        self.assertEqual(by_id['synthetic']['split'], 'train')
        verify_splits(result, rows, groups, near)

    def test_split_is_reproducible_and_keeps_each_class(self):
        rows = self.fixture()
        groups, near = group_rows(rows)
        first = assign_splits(rows, groups, seed=42)
        self.assertEqual(first, assign_splits(list(reversed(rows)), groups, seed=42)[::-1])
        self.assertEqual({r['score'] for r in first if r['split'] == 'validation'}, {1, 2, 3})
        self.assertTrue(all(not r['training_eligible'] for r in first))
        verify_splits(first, rows, groups, near)

    def test_corrupted_state_or_split_is_rejected(self):
        rows = self.fixture()
        groups, near = group_rows(rows)
        first = assign_splits(rows, groups)
        first[0]['score'] = 3
        with self.assertRaises(ValueError):
            verify_splits(first, rows, groups, near)

    def test_bad_labels_ids_and_old_schema_rejected(self):
        for rows in ([row('a', True)], [row('a', 4)], [row('a'), row('a')]):
            with self.assertRaises(ValueError):
                audit_rows(rows)
        old = row('old')
        old['state']['context']['recent_processes'] = []
        with self.assertRaises(ValueError):
            audit_rows([old])

    def test_end_to_end_excludes_held_preserves_sources_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            balanced = self.fixture()
            pilot = [dict(row('pilot', 2), scenario='new-scenario')]
            real = [dict(row('real', 3), review={'human_reviewed': True}),
                    dict(row('held', None), review={'reason_code': 'mock_foreground_hold'})]
            paths = [folder / (name + '.jsonl') for name in ('balanced', 'pilot', 'real')]
            for path, rows in zip(paths, (balanced, pilot, real)):
                write_lines(path, rows)
            protected = [p.read_bytes() for p in paths]
            result = prepare(*paths, folder / 'output')
            self.assertEqual(result['excluded_rows'], 1)
            self.assertEqual(result['human_labeled_real_rows'], 1)
            self.assertFalse(result['training_ready'])
            self.assertEqual(protected, [p.read_bytes() for p in paths])
            with self.assertRaises(ValueError):
                prepare(*paths, folder / 'output')


if __name__ == '__main__':
    unittest.main()
