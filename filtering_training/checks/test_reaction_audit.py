"""용도: 단문 진단이 원문·미확정 점수·맥락·앱 변형을 구분하고 과도한 동등성 판정을 피하는지 검증한다.
생성일: 2026-11-12
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from filtering_training.common.artifact_io import read
from filtering_training.common.encoder_input import input_text
from filtering_training.quality.audit_relevance_reactions import (
    EXPRESSIONS, authored_contexts, make_probes, summarize, write_first_review,
)


class ReactionAuditTests(unittest.TestCase):
    def setUp(self):
        self.contexts = authored_contexts()
        self.probes = make_probes(self.contexts)

    def test_raw_expressions_and_original_states_preserved(self):
        contexts = deepcopy(self.contexts)
        make_probes(self.contexts)
        self.assertEqual(contexts, self.contexts)
        import json
        for p in self.probes:
            self.assertEqual(p['state']['notification']['body'], p['body'])
            self.assertEqual(json.loads(input_text(p['state'], 'relevance'))['notification']['body'], p['body'])
        self.assertIn('', {p['body'] for p in self.probes})
        self.assertGreaterEqual(sum(map(len, EXPRESSIONS.values())), 40)

    def test_user_decisions_do_not_blanket_label_project_reactions(self):
        pending = [p for p in self.probes if p['context_id'] == 'project_code' and p['body'] in ('넵', '아뇨', 'ㅇㅋㅇㅋ', 'ㅜㅜ')]
        self.assertTrue(all(p['reference_score'] is None and p['label_status'] == 'policy_pending' for p in pending))
        approval = [p for p in self.probes if p['context_id'] == 'approval_target' and p['body'] in ('네', '아니')]
        self.assertEqual(len(approval), 6)
        self.assertTrue(all(p['reference_score'] == 2 and p['label_status'] == 'user_confirmed_example' for p in approval))
        self.assertTrue(all(not p['minimal_pair_comparable'] for p in self.probes if p['context_id'] == 'approval_target'))
        reviewed = [p for p in self.probes if p['context_id'] == 'project_code' and p['body'] in ('ㅇㅇ', 'ㅇㅋ', '네', '아니', 'ㅋㅋ', 'ㅠㅠ') and p['app'] == 'KakaoTalk.exe']
        self.assertEqual(len(reviewed), 6)
        self.assertTrue(all(p['reference_score'] == 1 and p['label_status'] == 'user_confirmed_example' for p in reviewed))
        other_apps = [p for p in self.probes if p['context_id'] == 'project_code' and p['body'] == '네' and p['app'] != 'KakaoTalk.exe']
        self.assertTrue(all(p['reference_score'] is None for p in other_apps))

    def test_pending_rows_are_not_counted_as_wrong(self):
        report = summarize(self.probes, [1] * len(self.probes))
        self.assertGreater(report['unjudged_rows'], 0)
        self.assertNotIn('accuracy', report)
        self.assertNotIn('policy_pending', report['reference_agreement'])
        self.assertEqual(report['minimal_pair_disagreements'], 0)

    def test_only_explicit_minimal_pairs_have_equivalence_checks(self):
        scores = [2 if p['body'] == 'ㅁ' else 1 for p in self.probes]
        report = summarize(self.probes, scores)
        self.assertGreater(report['minimal_pair_disagreements'], 0)
        self.assertTrue(all(set(p['scores']) == {'ㅇ', 'ㅁ'} for p in report['minimal_pairs']))
        self.assertFalse(report['app_variant_disagreements'])

    def test_extra_capture_preserves_its_original_app(self):
        extra = deepcopy(self.contexts[0])
        extra.update(id='capture', scope='training_side_capture')
        extra['state']['notification']['app_name'] = 'Discord'
        probes = make_probes([extra])
        self.assertIn('Discord', {p['app'] for p in probes if p['body'] == 'ㅇ'})
        with self.assertRaises(ValueError):
            make_probes([extra, extra])

    def test_review_cards_preserve_confirmed_scores_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_first_review(folder, self.probes)
            cards = read(folder / 'first-review-inputs.json')['cards']
            self.assertEqual(set(cards), set('ABCDEF'))
            self.assertTrue(all(c['score'] == 1 and len(c['input_sha256']) == 64 for c in cards.values()))

    def test_invalid_prediction_coverage_and_scores_rejected(self):
        for scores in ([1], [True] * len(self.probes), [4] * len(self.probes)):
            with self.assertRaises(ValueError):
                summarize(self.probes, scores)


if __name__ == '__main__':
    unittest.main()
