"""용도: 전체 관련도 검수에서 현재 창 우선·보조 최근 창·확인된 자료 유용성·예외 입력 추적을 검증한다.
생성일: 2026-11-08
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from filtering_training.common.artifact_io import digest, load_lines, read, write
from filtering_training.common.encoder_input import input_text
from filtering_training.preparation.curate_relevance import key, write_lines
from filtering_training.review.review_relevance_dataset import Reviewer, POLICY, review


class RelevanceReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reviewer = Reviewer()

    def row(self, body, *, current=0, history=None, score=1):
        task = self.reviewer.tasks[0]
        window = '' if current is None else self.reviewer.tasks[current]['goal'] + ' - Editor'
        history = [] if history is None else [
            {'app_name': 'chrome.exe', 'window_title': self.reviewer.tasks[i]['goal']} for i in history]
        return {'id': 'test_only', 'score': score,
                'state': {'notification': {'app_name': 'Slack', 'sender': task['domain'] + ' 담당',
                                          'title': task['domain'] + ' 소식', 'body': body,
                                          'timestamp': '2026-11-08T03:00:00Z'},
                          'context': {'active_process': 'Code.exe' if current is not None else '',
                                      'window_title': window, 'recent_windows': history}}}

    def render(self, family):
        phrase = self.reviewer.definitions['TRAINING_EVENT_PHRASINGS'][family][0]
        return phrase.format(**self.reviewer.tasks[0])

    def test_recent_windows_do_not_override_clear_matching_current(self):
        row = self.row(self.render('community_news'), history=[10], score=1)
        score, rule, _, method = self.reviewer.decide(row)
        self.assertEqual((score, rule), (2, 'community_news'))
        self.assertEqual(method, 'template_substitution_and_context_check')
        row['score'] = 3
        self.assertEqual(self.reviewer.decide(row)[0], 2)

    def test_old_task_history_does_not_override_clear_different_current(self):
        row = self.row(self.render('waiting'), current=10, history=[0], score=3)
        self.assertEqual(self.reviewer.decide(row)[0], 1)

    def test_specific_history_can_resolve_missing_current(self):
        self.assertEqual(self.reviewer.decide(self.row(self.render('reference'), current=None, history=[0]))[0], 3)
        self.assertEqual(self.reviewer.decide(self.row(self.render('reference'), current=None, history=[]))[0], 1)

    def test_archive_links_and_file_formats_remain_useful_for_same_artifact(self):
        for family in ('archive', 'bookmark', 'formats'):
            with self.subTest(family=family):
                self.assertEqual(self.reviewer.decide(self.row(self.render(family)))[0], 3)

    def test_optional_theme_is_topic_only_even_when_old_score_was_three(self):
        row = self.row(self.render('theme'), score=3)
        self.assertEqual(self.reviewer.decide(row)[0], 2)

    def test_unrecognized_sentence_is_not_silently_approved(self):
        row = self.row('New unseen scenario; inspect rather than copying its label.', score=3)
        self.assertIsNone(self.reviewer.decide(row)[0])

    def test_manual_exception_requires_exact_model_input_fingerprint(self):
        row = self.row('A joke about this task.', score=3)
        row['id'] = 'manual_test'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.json'
            policy = read(POLICY)
            policy['manual_cases'] = {row['id']: {'score': 2, 'reason': 'Reviewed as topic-only joke',
                'input_sha256': key(json.loads(input_text(row['state'], 'relevance')))}}
            write(path, policy)
            reviewer = Reviewer(path)
            self.assertEqual(reviewer.decide(row)[0], 2)
            row['state']['context']['window_title'] = 'Unrelated current task'
            self.assertEqual(reviewer.decide(row)[1], 'manual_input_changed')
            self.assertIsNone(reviewer.decide(row)[0])

    def test_review_is_independent_of_notification_app_and_does_not_mutate_rows(self):
        row = self.row(self.render('draft'))
        before = deepcopy(row)
        scores = []
        for app in ('Slack', 'KakaoTalk.exe', 'Teams'):
            variant = deepcopy(row)
            variant['state']['notification']['app_name'] = app
            scores.append(self.reviewer.decide(variant)[0])
        self.assertEqual(scores, [3, 3, 3])
        self.assertEqual(row, before)

    def test_changed_policy_cannot_silently_use_old_implementation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.json'
            policy = read(POLICY)
            policy['confirmed_user_rules']['same_artifact_archive_location'] = 2
            write(path, policy)
            with self.assertRaisesRegex(ValueError, 'changed user policy'):
                Reviewer(path)

    def test_review_writes_complete_trace_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / 'relevance/retained.jsonl'
            rows = [self.row(self.render('community_news'), history=[10], score=1),
                    self.row(self.render('waiting'), current=10, history=[0], score=1)]
            for i, row in enumerate(rows):
                row['id'], row['group_id'] = 'row_' + str(i), 'group_' + str(i)
            write_lines(source, rows)
            original_hash = digest(source)
            write(folder / 'manifest.json', {'files': {'relevance/retained.jsonl': {'sha256': original_hash}},
                                            'protected_hashes': {}})
            summary = review(folder, 'test-batch')
            self.assertEqual(summary['decision_rows'], 2)
            self.assertEqual(summary['coverage'], {'proposed_change': 1, 'proposed_keep': 1})
            proposed = load_lines(folder / 'review/test-batch/proposed.jsonl')
            self.assertEqual([r['score'] for r in proposed], [2, 1])
            self.assertEqual([r['state'] for r in proposed], [r['state'] for r in rows])
            self.assertTrue(all(not r['training_eligible'] for r in proposed))
            self.assertEqual(digest(source), original_hash)
            self.assertTrue((folder / 'review/test-batch/policy-snapshot.json').is_file())
            with self.assertRaisesRegex(ValueError, 'preserved'):
                review(folder, 'test-batch')


if __name__ == '__main__':
    unittest.main()
