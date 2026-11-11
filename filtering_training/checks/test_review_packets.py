"""용도: 검수 카드의 입력 연결·명시적 답변·미응답 보존을 검증한다.
생성일: 2026-11-08
"""

from copy import deepcopy
import unittest

from filtering_training.preparation.curate_relevance import key
from filtering_training.review.prepare_real_review_packets import answer_decisions, build_packet, TEMPLATE


class ReviewPacketTests(unittest.TestCase):
    def setUp(self):
        self.decisions = [{'id': str(i), 'input_sha256': f'hash{i}', 'proposed_score': 2,
                           'reviewer': 'assistant', 'human_reviewed': False} for i in (1, 2, 3)]
        self.packet = {'cases': [{'case_id': c, 'members': [
            {'id': str(i), 'input_sha256': f'hash{i}'} for i in nums]} for c, nums in [('A', [1, 2]), ('B', [3])]]}
        self.packet['packet_id'] = key(self.packet)

    def answer(self, **answers):
        return {'packet_id': self.packet['packet_id'], 'answers': answers}

    def test_only_explicit_card_members_change(self):
        original = deepcopy(self.decisions)
        result = answer_decisions(self.packet, self.answer(A={'score': 3}), self.decisions)
        self.assertEqual([r['proposed_score'] for r in result], [3, 3, 2])
        self.assertEqual(result[2], original[2])
        self.assertEqual(self.decisions, original)
        self.assertTrue(result[0]['human_reviewed'])

    def test_hold_is_explicit_and_previous_choices_survive(self):
        first = answer_decisions(self.packet, self.answer(A={'score': 3}), self.decisions)
        second = answer_decisions(self.packet, self.answer(B={'score': None, 'note': 'unknown'}), first)
        self.assertEqual(second[:2], first[:2])
        self.assertIsNone(second[2]['proposed_score'])
        self.assertEqual(second[2]['user_note'], 'unknown')

    def test_unknown_or_missing_answers_rejected(self):
        for answers in ({}, {'Z': {'score': 2}}):
            with self.assertRaises(ValueError):
                answer_decisions(self.packet, self.answer(**answers), self.decisions)

    def test_boolean_and_out_of_scale_scores_rejected(self):
        for score in (True, 0, 4, '2'):
            with self.assertRaises(ValueError):
                answer_decisions(self.packet, self.answer(A={'score': score}), self.decisions)

    def test_changed_packet_and_changed_input_rejected(self):
        tampered = deepcopy(self.packet)
        tampered['cases'][0]['case_id'] = 'C'
        with self.assertRaises(ValueError):
            answer_decisions(tampered, self.answer(A={'score': 2}), self.decisions)
        self.decisions[0]['input_sha256'] = 'other'
        with self.assertRaises(ValueError):
            answer_decisions(self.packet, self.answer(A={'score': 2}), self.decisions)

    def test_ui_has_no_preselected_score_and_no_untrusted_html_insertion(self):
        template = TEMPLATE.read_text(encoding='utf-8')
        self.assertEqual(template.count('__PACKET_JSON__'), 1)
        self.assertIn('let answers={},round=1', template)
        self.assertNotIn('innerHTML', template)
        self.assertIn('textContent=text', template)

    def test_duplicate_members_and_mixed_scores_rejected(self):
        rows = [{'source_number': str(i), 'id': str(i), 'score': score, 'state': {},
                 'review': {'input_sha256': str(i)},
                 'provenance': {'received_kst': '', 'quality_flags': []}}
                for i, score in [(1, 2), (2, 3)]]
        for numbers in ([1, 1], [1, 2]):
            plan = {'previously_held_reason': 'hold', 'quality_labels': {},
                    'cases': [{'case_id': 'A', 'round': 1, 'source_numbers': numbers}]}
            with self.assertRaises(ValueError):
                build_packet(rows, plan)


if __name__ == '__main__':
    unittest.main()
