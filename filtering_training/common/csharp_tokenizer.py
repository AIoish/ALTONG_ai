"""용도: 앱 develop의 C# KlueTokenizer 로딩 조건과 인코딩 동작을 Python으로 재현해 토큰 일치를 검사한다.
생성일: 2026-11-18
"""

import json
from pathlib import Path
import unicodedata

# Ids the app's KlueTokenizer hard-codes when loading and when wrapping each input.
APP_SPECIAL_IDS = {'[CLS]': 0, '[PAD]': 1, '[SEP]': 2}
MAX_TOKENS = 512
_PUNCTUATION_CATEGORIES = {'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'}
# .NET Rune.IsWhiteSpace is the Unicode White_Space property (narrower than str.isspace()).
_WHITE_SPACE = {*range(0x09, 0x0e), 0x20, 0x85, 0xa0, 0x1680, *range(0x2000, 0x200b),
                0x2028, 0x2029, 0x202f, 0x205f, 0x3000}


def load_rules(tokenizer_json):
    """Apply every KlueTokenizer constructor check; raise ValueError where C# would refuse to load."""
    root = json.loads(Path(tokenizer_json).read_text(encoding='utf-8'))
    model, norm = root['model'], root['normalizer'] or {}
    if (model.get('type') != 'WordPiece' or norm.get('type') != 'BertNormalizer' or norm.get('lowercase') is not False
            or norm.get('clean_text') is not True or norm.get('handle_chinese_chars') is not True
            or norm.get('strip_accents') is not None or (root.get('pre_tokenizer') or {}).get('type') != 'BertPreTokenizer'):
        raise ValueError('tokenizer config differs from the app KlueTokenizer contract')
    vocab = model['vocab']
    special = {}
    for token in root['added_tokens']:
        if (token['special'] is not True or token['normalized'] is not False or token['single_word'] is not False
                or token['lstrip'] is not False or token['rstrip'] is not False):
            raise ValueError('unsupported added token flags for the app tokenizer')
        special[token['content']] = token['id']
    if any(vocab.get(name) != value for name, value in APP_SPECIAL_IDS.items()) or \
            (root.get('post_processor') or {}).get('type') != 'TemplateProcessing':
        raise ValueError('special token ids/template differ from the app contract')
    return {'vocab': vocab, 'special': special, 'prefix': model['continuing_subword_prefix'],
            'max_chars': model['max_input_chars_per_word'], 'unknown': vocab[model['unk_token']]}


def _is_cjk(n):
    return (0x4e00 <= n <= 0x9fff or 0x3400 <= n <= 0x4dbf or 0x20000 <= n <= 0x2a6df or 0x2a700 <= n <= 0x2b73f
            or 0x2b740 <= n <= 0x2b81f or 0x2b820 <= n <= 0x2ceaf or 0xf900 <= n <= 0xfaff or 0x2f800 <= n <= 0x2fa1f)


def _is_punctuation(n, category):
    return 33 <= n <= 47 or 58 <= n <= 64 or 91 <= n <= 96 or 123 <= n <= 126 or category in _PUNCTUATION_CATEGORIES


def _wordpiece(rules, word, ids):
    chars = list(word)
    if len(chars) > rules['max_chars']:
        ids.append(rules['unknown'])
        return
    pieces, start = [], 0
    while start < len(chars):
        found, end = None, len(chars)
        while end > start:
            candidate = ('' if start == 0 else rules['prefix']) + ''.join(chars[start:end])
            if candidate in rules['vocab']:
                found = rules['vocab'][candidate]
                break
            end -= 1
        if found is None:
            ids.append(rules['unknown'])
            return
        pieces.append(found)
        start = end
    ids.extend(pieces)


def _encode_ordinary(rules, text, ids):
    word = []

    def flush():
        if word:
            _wordpiece(rules, ''.join(word), ids)
            word.clear()
            check()

    def check():
        if len(ids) > MAX_TOKENS - 1:
            raise ValueError('input exceeds 512 tokens')

    for char in text:
        n, category = ord(char), unicodedata.category(char)
        white = n in _WHITE_SPACE
        if n == 0 or n == 0xfffd or (not white and category in ('Cc', 'Cf')):
            continue
        if white:
            flush()
            continue
        if _is_cjk(n) or _is_punctuation(n, category):
            flush()
            _wordpiece(rules, char, ids)
            check()
        else:
            word.append(char)
    flush()


def encode(rules, text):
    """Mirror KlueTokenizer.Encode, including the literal 0/2 wrapper ids."""
    if len(text.encode('utf-16-le')) // 2 > 65536:
        raise ValueError('input is too large')
    ids, cursor = [APP_SPECIAL_IDS['[CLS]']], 0
    while cursor < len(text):
        found, special = len(text), None
        for token in rules['special']:
            at = text.find(token, cursor)
            if 0 <= at < found:
                found, special = at, token
        _encode_ordinary(rules, text[cursor:found], ids)
        if special is None:
            break
        ids.append(rules['special'][special])
        cursor = found + len(special)
    ids.append(APP_SPECIAL_IDS['[SEP]'])
    if len(ids) > MAX_TOKENS:
        raise ValueError('input exceeds 512 tokens')
    return ids


def parity(tokenizer_json, texts, hf_tokenizer):
    """Compare the app tokenizer port against the Hugging Face tokenizer used for training/export."""
    rules = load_rules(tokenizer_json)
    mismatches = []
    for index, text in enumerate(texts):
        expected = hf_tokenizer(text, truncation=False, return_token_type_ids=False)['input_ids']
        if encode(rules, text) != expected:
            mismatches.append(index)
    return {'texts': len(texts), 'mismatch_indices': mismatches[:20], 'mismatches': len(mismatches)}
