"""용도: 앱 develop 입력 문자열·C# 토크나이저 재현·특수 토큰 재배치·[batch,2,5] 출력·공동 학습 선택 규칙을 검증한다.
생성일: 2026-11-18
"""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from filtering_training.common.csharp_tokenizer import encode, load_rules, parity
from filtering_training.common.encoder_input import APP_INPUT_SCHEMA, input_text
from filtering_training.modeling.align_special_token_ids import permutation, permuted_backend
from filtering_training.modeling.joint_app_contract import (
    APP_RELEVANCE_SCORE, FILLER_LOGIT, app_logits, app_passed, rank, task_weights, urgency_gate,
)


def csharp_quote(value):
    """Literal port of ModelInputBuilder.Quote for the expected strings below."""
    escapes = {'"': '\\"', '\\': '\\\\', '\b': '\\b', '\f': '\\f', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
    return '"' + ''.join(escapes.get(c, '\\u%04x' % ord(c) if c < ' ' else c) for c in value) + '"'


def csharp_build(n, active, title, windows, relevance):
    s = ('{"notification":{"app_name":' + csharp_quote(n['app_name']) + ',"sender":' + csharp_quote(n['sender'] or '') +
         ',"title":' + csharp_quote(n['title']) + ',"body":' + csharp_quote(n['body']) +
         ',"timestamp":' + csharp_quote(n['timestamp']) + '}')
    if relevance:
        recent = [w for w in windows if w['app_name'] != active or w['window_title'] != title][:3]
        s += (',"current_window":{"active_process":' + csharp_quote(active) + ',"window_title":' + csharp_quote(title) +
              '},"recent_windows":[' + ','.join('{"app_name":' + csharp_quote(w['app_name']) + ',"window_title":' +
                                              csharp_quote(w['window_title']) + '}' for w in recent) +
              '],"recent_processes":[' + ','.join(csharp_quote(w['app_name']) for w in recent) + ']')
    return s + '}'


def state():
    windows = [{'app_name': 'EXCEL.EXE', 'window_title': '분기 합계.xlsx - Excel'},
               {'app_name': 'Code.exe', 'window_title': 'main.py "draft"\tv2'},
               {'app_name': 'chrome.exe', 'window_title': '검토 - Chrome'},
               {'app_name': 'Slack.exe', 'window_title': '#일반'},
               {'app_name': 'explorer.exe', 'window_title': '다운로드'}]
    # Body/title are listed out of app order on purpose; sender is null in the app.
    return {'notification': {'id': 'n-1', 'body': '합계표\n공유\x01했습니다.', 'title': '검증용', 'app_name': 'Slack',
                             'sender': None, 'timestamp': '2026-11-18T00:00:00Z'},
            'context': {'active_process': 'EXCEL.EXE', 'window_title': '분기 합계.xlsx - Excel',
                        'recent_windows': windows, 'recent_processes': ['ignored.exe']}}


def tiny_tokenizer(folder, ids):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    words = ['합계', '##표', '공유', '했', '##습니다', '.', '{', '}', '"', ':', ',', 'Slack', '中']
    vocab = dict(ids)
    for word in words:
        vocab[word] = len(vocab)
    tok = Tokenizer(models.WordPiece(vocab, unk_token='[UNK]', max_input_chars_per_word=100))
    tok.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]', special_tokens=[('[CLS]', ids['[CLS]']), ('[SEP]', ids['[SEP]'])])
    tok.add_special_tokens(list(ids))
    path = Path(folder) / 'tokenizer.json'
    tok.save(str(path))
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]',
                                      sep_token='[SEP]', mask_token='[MASK]', model_input_names=['input_ids', 'attention_mask'])
    return path, wrapped


APP_IDS = {'[CLS]': 0, '[PAD]': 1, '[SEP]': 2, '[UNK]': 3, '[MASK]': 4}
KC_IDS = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
TEXTS = ['합계표 공유했습니다.', '{"app":"Slack"}', '中 합계표  ,  공유', '모르는단어 Slack', '']


class AppInputTests(unittest.TestCase):
    def test_relevance_string_matches_app_builder(self):
        s = state()
        expected = csharp_build(s['notification'], 'EXCEL.EXE', '분기 합계.xlsx - Excel', s['context']['recent_windows'], True)
        self.assertEqual(input_text(s, 'relevance', schema=APP_INPUT_SCHEMA), expected)
        value = json.loads(expected)
        self.assertEqual([w['app_name'] for w in value['recent_windows']], ['Code.exe', 'chrome.exe', 'Slack.exe'])
        self.assertEqual(value['recent_processes'], ['Code.exe', 'chrome.exe', 'Slack.exe'])

    def test_urgency_string_is_notification_only_in_app_order(self):
        s = state()
        expected = csharp_build(s['notification'], '', '', [], False)
        self.assertEqual(input_text(s, 'urgency', schema=APP_INPUT_SCHEMA), expected)
        self.assertTrue(expected.startswith('{"notification":{"app_name":"Slack","sender":"","title"'))

    def test_non_string_fields_are_rejected(self):
        s = state()
        s['notification']['title'] = 3
        with self.assertRaises(ValueError):
            input_text(s, 'urgency', schema=APP_INPUT_SCHEMA)


class AppTokenizerTests(unittest.TestCase):
    def test_port_matches_hugging_face_for_app_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            path, tokenizer = tiny_tokenizer(folder, APP_IDS)
            self.assertEqual(parity(path, TEXTS, tokenizer)['mismatches'], 0)
            self.assertEqual(encode(load_rules(path), '합계표')[0], 0)
            self.assertEqual(encode(load_rules(path), '합계표')[-1], 2)

    def test_app_refuses_other_special_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = tiny_tokenizer(folder, KC_IDS)
            with self.assertRaises(ValueError):
                load_rules(path)

    def test_permutation_moves_only_required_ids(self):
        order = permutation(8, {'[CLS]': 2, '[PAD]': 0, '[SEP]': 3})
        self.assertEqual(order, [2, 0, 3, 1, 4, 5, 6, 7])
        self.assertEqual(permutation(6, {'[CLS]': 0, '[PAD]': 1, '[SEP]': 2}), list(range(6)))
        with self.assertRaises(ValueError):
            permutation(4, {'[CLS]': 0, '[PAD]': 0, '[SEP]': 2})

    def test_permuted_backend_is_relabelled_and_app_loadable(self):
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as folder:
            _, original = tiny_tokenizer(folder, KC_IDS)
            current = {k: original.convert_tokens_to_ids(k) for k in ('[CLS]', '[PAD]', '[SEP]')}
            order = permutation(len(original.get_vocab()), current)
            inverse = [0] * len(order)
            for new, old in enumerate(order):
                inverse[old] = new
            aligned = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(permuted_backend(original, inverse)),
                                              unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]',
                                              model_input_names=['input_ids', 'attention_mask'])
            out = Path(folder) / 'aligned'
            aligned.save_pretrained(str(out))
            for text in TEXTS:
                self.assertEqual(aligned(text)['input_ids'], [inverse[i] for i in original(text)['input_ids']])
            self.assertEqual(parity(out / 'tokenizer.json', TEXTS, aligned)['mismatches'], 0)


class AppOutputAndSelectionTests(unittest.TestCase):
    def test_app_logits_shape_finite_and_relevance_slots(self):
        urgency = torch.randn(3, 5)
        relevance = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]])
        logits = app_logits(urgency, relevance)
        self.assertEqual(tuple(logits.shape), (3, 2, 5))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(logits[:, 0], urgency))
        self.assertEqual((logits[:, 1].argmax(-1) + 1).tolist(), [APP_RELEVANCE_SCORE[s] for s in (1, 2, 3)])
        self.assertTrue((logits[:, 1, 2:4] == FILLER_LOGIT).all())

    def test_app_policy_equals_relevance_three(self):
        for urgency in range(1, 6):
            for relevance in range(1, 4):
                self.assertEqual(app_passed(urgency, APP_RELEVANCE_SCORE[relevance]), urgency >= 4 or relevance == 3)

    def test_task_weights_equalize_task_mass(self):
        records = [(None, 0, 0)] * 6 + [(None, 1, 0)] * 2
        row_weights = [1.] * 6 + [1.5, .5]
        weights = task_weights(records, row_weights)
        mass = [sum(w for (_, t, _), w in zip(records, row_weights) if t == task) * weights[task] for task in (0, 1)]
        self.assertAlmostEqual(mass[0], mass[1])
        with self.assertRaises(ValueError):
            task_weights(records, [1.] * 6 + [0., 0.])

    def test_urgency_gate_precedes_relevance(self):
        reference = {'validation': {'threshold_miss_ids': ['a']}, 'safety_development': {'threshold_miss_ids': []}}
        ok = urgency_gate({'validation': {'threshold_miss_ids': ['b']}, 'safety_development': {'threshold_miss_ids': []}}, reference)
        bad = urgency_gate({'validation': {'threshold_miss_ids': []}, 'safety_development': {'threshold_miss_ids': ['c']}}, reference)
        self.assertTrue(ok['passed'])
        self.assertFalse(bad['passed'])
        weak = {'exact_accuracy': .5, 'macro_f1': .5, 'mae': .5}
        strong = {'exact_accuracy': .9, 'macro_f1': .9, 'mae': .1}
        self.assertGreater(rank(ok, weak), rank(bad, strong))


if __name__ == '__main__':
    unittest.main()
