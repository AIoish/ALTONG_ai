"""용도: BERT 계열 사전학습 인코더의 특수 토큰 ID를 앱 KlueTokenizer 규격([CLS]=0,[PAD]=1,[SEP]=2)으로 무손실 재배치한다.
생성일: 2026-11-18
"""

import argparse
import json
from pathlib import Path

from filtering_training.common.artifact_io import ROOT, created_date, digest, write
from filtering_training.common.csharp_tokenizer import APP_SPECIAL_IDS, load_rules

SOURCE = ROOT / 'filtering_training/outputs/models/kcelectra-base-v2022'
TARGET = ROOT / 'filtering_training/outputs/models/kcelectra-base-v2022-app-ids'
SOURCE_REVISION = '87285578b4686b6d190abd40e947a2a416add6de'


def permutation(vocab_size, current, targets=APP_SPECIAL_IDS):
    """Return order with order[new_id] == old_id, moving only the ids needed by the target layout."""
    if len(set(current.values())) != len(current) or any(not 0 <= v < vocab_size for v in current.values()):
        raise ValueError('special token ids must be unique and inside the vocabulary')
    if set(targets) - set(current) or len(set(targets.values())) != len(targets):
        raise ValueError('every target token must exist and target ids must be unique')
    order = list(range(vocab_size))
    for token, new_id in targets.items():
        position = order.index(current[token])
        order[position], order[new_id] = order[new_id], order[position]
    if sorted(order) != list(range(vocab_size)):
        raise ValueError('not a permutation')
    return order


def permuted_backend(tokenizer, inverse):
    """Rewrite every id in the source fast-tokenizer JSON; keep normalizer/pre-tokenizer/template unchanged."""
    root = json.loads(tokenizer.backend_tokenizer.to_str())
    root['model']['vocab'] = {token: inverse[old] for token, old in root['model']['vocab'].items()}
    for token in root['added_tokens']:
        token['id'] = inverse[token['id']]
    for special in (root.get('post_processor') or {}).get('special_tokens', {}).values():
        special['ids'] = [inverse[i] for i in special['ids']]
    if root.get('padding'):
        root['padding']['pad_id'] = inverse[root['padding']['pad_id']]
    return json.dumps(root, ensure_ascii=False)


def text_samples():
    from filtering_training.common.encoder_input import APP_INPUT_SCHEMA, input_text
    from filtering_training.common.artifact_io import load_lines
    student = ROOT / 'filtering_training/outputs/v22_real_seed_distillation_01/student-candidate-02/relevance'
    v13 = ROOT / 'filtering_training/outputs/v13_previous_windows_relevance_01/mapped-training'
    texts = [input_text(r['state'], 'relevance', schema=APP_INPUT_SCHEMA)
             for split in ('train', 'validation') for r in load_lines(student / f'{split}.jsonl')]
    texts += [input_text(r['state'], 'urgency', schema=APP_INPUT_SCHEMA)
              for name in ('initial', 'safety') for r in load_lines(v13 / f'{name}.jsonl') if r['task'] == 'urgency']
    return list(dict.fromkeys(texts))


def align(source=SOURCE, target=TARGET, *, equivalence_rows=64):
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
    from filtering_training.common.csharp_tokenizer import parity
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError('preserve existing aligned model; choose a new target')
    old_tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    vocab_lines = (source / 'vocab.txt').read_text(encoding='utf-8').split('\n')
    if vocab_lines and vocab_lines[-1] == '':
        vocab_lines.pop()
    if len(vocab_lines) != len(old_tokenizer.get_vocab()):
        raise ValueError('vocab.txt does not match the loaded tokenizer')
    current = {name: old_tokenizer.convert_tokens_to_ids(name) for name in APP_SPECIAL_IDS}
    order = permutation(len(vocab_lines), current)
    inverse = [0] * len(order)
    for new_id, old_id in enumerate(order):
        inverse[old_id] = new_id
    target.mkdir(parents=True)
    (target / 'vocab.txt').write_text(''.join(vocab_lines[old] + '\n' for old in order), encoding='utf-8')
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(permuted_backend(old_tokenizer, inverse)),
                                        unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]',
                                        mask_token='[MASK]', model_max_length=512,
                                        model_input_names=['input_ids', 'attention_mask'])
    tokenizer.save_pretrained(str(target))
    model = AutoModel.from_pretrained(str(source), local_files_only=True).eval()
    with torch.no_grad():
        weight = model.get_input_embeddings().weight
        weight.copy_(weight[torch.tensor(order)].clone())
    model.config.pad_token_id = APP_SPECIAL_IDS['[PAD]']
    model.save_pretrained(str(target), safe_serialization=True)

    texts = text_samples()
    new_tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
    old_ids = old_tokenizer(texts, truncation=False, return_token_type_ids=False)['input_ids']
    new_ids = new_tokenizer(texts, truncation=False, return_token_type_ids=False)['input_ids']
    remapped = sum([inverse[i] for i in old] != new for old, new in zip(old_ids, new_ids))
    if remapped:
        raise ValueError(f'aligned tokenizer differs from permuted ids in {remapped} texts')
    app = parity(target / 'tokenizer.json', texts, new_tokenizer)
    if app['mismatches']:
        raise ValueError('app tokenizer port differs: ' + json.dumps(app))
    original = AutoModel.from_pretrained(str(source), local_files_only=True).eval()
    aligned = AutoModel.from_pretrained(str(target), local_files_only=True).eval()
    sample = texts[:equivalence_rows]
    worst = 0.
    with torch.no_grad():
        for old_batch, new_batch in ((old_tokenizer(sample, padding=True, return_tensors='pt', return_token_type_ids=False),
                                      new_tokenizer(sample, padding=True, return_tensors='pt', return_token_type_ids=False)),):
            a = original(**old_batch).last_hidden_state
            b = aligned(**new_batch).last_hidden_state
            mask = old_batch['attention_mask'].bool()
            worst = float((a - b).abs()[mask].max())
    if worst > 1e-4:
        raise ValueError(f'aligned encoder output changed: {worst}')
    manifest = {'purpose': 'Lossless special-token id permutation for the app KlueTokenizer contract',
                'created_date': created_date(), 'source': str(source), 'source_repo': 'beomi/KcELECTRA-base-v2022',
                'source_revision': SOURCE_REVISION, 'license': 'MIT', 'source_special_ids': current,
                'target_special_ids': APP_SPECIAL_IDS, 'moved_ids': {str(n): o for n, o in enumerate(order) if n != o},
                'discriminator_head_dropped': True, 'pad_token_id': APP_SPECIAL_IDS['[PAD]'],
                'texts_checked': len(texts), 'permuted_token_mismatches': remapped, 'app_tokenizer_port': app,
                'encoder_equivalence_rows': len(sample), 'max_abs_hidden_difference': worst,
                'files': {p.name: digest(p) for p in sorted(target.iterdir()) if p.is_file()}}
    load_rules(target / 'tokenizer.json')
    write(target / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--target', type=Path, default=TARGET)
    args = parser.parse_args()
    report = align(args.source, args.target)
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
