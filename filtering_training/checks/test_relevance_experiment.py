"""용도: 관련도 후보 분할에서 정답 충돌·대비 묶음 누출·실제 자료의 검증 유입을 막는다.
생성일: 2026-11-11
"""

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from filtering_training.preparation.curate_relevance import write_lines
from filtering_training.common.artifact_io import digest, write
from filtering_training.preparation.prepare_relevance_experiment import (
    assign_splits, audit_rows, group_rows, prepare, reviewed_examples, verify_splits,
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

    def reviewed_fixture(self, folder):
        from filtering_training.quality.audit_relevance_reactions import authored_contexts, make_probes
        probes = []
        for p in make_probes(authored_contexts()):
            if p['context_id'] == 'project_code' and p['app'] == 'KakaoTalk.exe' and p['label_status'] == 'user_confirmed_example':
                p['label_status'] = 'user_confirmed_input'
                p['review'] = {'reviewer': 'user', 'human_reviewed': True,
                               'input_sha256': p['input_sha256'], 'proposed_score': 1}
                probes.append(p)
            elif p['context_id'] == 'approval_target' and p['label_status'] == 'user_confirmed_example':
                probes.append(p)
            elif p['context_id'] == 'project_code' and p['body'] == '넵' and p['app'] == 'KakaoTalk.exe':
                probes.append(p)
        path = folder / 'probes.jsonl'
        write_lines(path, probes)
        write(folder / 'manifest.json', {'files': {'probes.jsonl': digest(path)}})
        return path, probes

    def test_reviewed_import_selects_six_confirmations_and_two_controls_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.reviewed_fixture(Path(temp))
            rows, protected = reviewed_examples(path)
            self.assertEqual(len(rows), 8)
            self.assertEqual(sum(r['score'] == 1 for r in rows), 6)
            self.assertEqual(sum(r['score'] == 2 for r in rows), 2)
            self.assertTrue(all(r['experiment']['train_only'] for r in rows))
            self.assertTrue(all(r['experiment']['loss_origin'] == 'pilot' for r in rows))
            self.assertEqual(sum(r['experiment']['label_source'] == 'user' for r in rows), 6)
            self.assertIn(str(path.resolve()), protected)

    def test_reviewed_import_checks_fingerprints_and_user_attribution(self):
        for mutation in ('hash', 'reviewer'):
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                path, probes = self.reviewed_fixture(folder)
                probe = next(p for p in probes if p['label_status'] == 'user_confirmed_input')
                if mutation == 'hash':
                    probe['state']['notification']['body'] = 'changed'
                else:
                    probe['review']['human_reviewed'] = False
                write_lines(path, probes)
                write(folder / 'manifest.json', {'files': {'probes.jsonl': digest(path)}})
                with self.assertRaises(ValueError):
                    reviewed_examples(path)

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

    def test_context_only_characters_can_use_different_scenarios_without_input_leakage(self):
        a, b = row('reaction-a', 1, body='ㅇ'), row('reaction-b', 2, body='ㅇ')
        groups, _ = group_rows([a, b])
        self.assertNotEqual(groups[a['id']], groups[b['id']])
        b['state'] = deepcopy(a['state'])
        groups, _ = group_rows([a, b])
        self.assertEqual(groups[a['id']], groups[b['id']])

    def test_reference_members_keep_their_split_and_bridging_is_rejected(self):
        rows = self.fixture()
        groups, _ = group_rows(rows)
        frozen = {rows[0]['id']: 'validation', rows[1]['id']: 'train'}
        output = assign_splits(rows, groups, reference=frozen)
        self.assertTrue(all(r['split'] == frozen[r['id']] for r in output if r['id'] in frozen))
        groups[rows[1]['id']] = groups[rows[0]['id']]
        with self.assertRaisesRegex(ValueError, 'bridges frozen'):
            assign_splits(rows, groups, reference=frozen)

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
