"""용도: 관련도 개발용 분할의 출처 해시·전체 행·그룹 보존·실제 토큰 길이를 재검증한다.
생성일: 2026-11-11
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import input_text
from filtering_training.common.relevance_policy import POLICY, apply_policy, targetless
from filtering_training.preparation.prepare_relevance_experiment import (
    OUTPUT, audit_rows, input_key, load_sources, verify_splits,
)


def verify(folder=OUTPUT, *, write_report=True):
    folder = Path(folder)
    manifest = read(folder / 'manifest.json')
    for path, expected in manifest['protected_sources'].items():
        if digest(path) != expected:
            raise ValueError('source hash mismatch: ' + path)
    for name, info in manifest['files'].items():
        if digest(folder / name) != info['sha256']:
            raise ValueError('output hash mismatch: ' + name)
    rows = load_lines(folder / 'relevance/train.jsonl') + load_lines(folder / 'relevance/validation.jsonl')
    migration = manifest.get('policy_migration')
    if migration:
        if migration['version'] != POLICY or manifest.get('label_policy') != POLICY:
            raise ValueError('unknown migration policy')
        parent = Path(migration['source'])
        if parent.resolve() == folder.resolve():
            raise ValueError('migration cannot reference itself')
        verify(parent, write_report=False)
        originals = [r for split in ('train', 'validation') for r in load_lines(parent / f'relevance/{split}.jsonl')]
        sources = apply_policy(originals)
        if rows != sources:
            raise ValueError('migration changed inputs, splits, provenance or unexpected labels')
        expected_changes = [dict(id=a['id'], split=a['split'], input_sha256=input_key(a),
                                 previous_score=a['score'], score=b['score'], policy=POLICY)
                            for a, b in zip(originals, sources) if a['score'] != b['score']]
        if load_lines(folder / 'review/policy-changes.jsonl') != expected_changes:
            raise ValueError('policy change ledger mismatch')
        excluded = load_lines(parent / 'review/excluded.jsonl')
    elif manifest.get('distillation_preparation'):
        from filtering_training.preparation.prepare_relevance_distillation import student_materials
        inputs=manifest['distillation_preparation']
        required=[Path(inputs[k]).resolve()/'manifest.json' for k in ('reference','frozen','teacher','real_review')]
        if inputs.get('decisions'):required.append(Path(inputs['decisions']).resolve())
        if any(str(p) not in manifest['protected_sources'] for p in required):
            raise ValueError('unprotected distillation source')
        sources,excluded,expected_near,changes,_=student_materials(**inputs)
        expected=[r for split in ('train','validation') for r in sources if r['split']==split]
        if rows!=expected:raise ValueError('distillation rows or provenance changed')
        if load_lines(folder/'review/policy-changes.jsonl')!=changes:
            raise ValueError('distillation policy change ledger mismatch')
        if load_lines(folder/'review/near-duplicate-links.jsonl')!=expected_near:
            raise ValueError('distillation grouping ledger mismatch')
    else:
        paths = {r['experiment']['origin']: Path(r['experiment']['source_file']) for r in rows}
        if any(str(p.resolve()) not in manifest['protected_sources'] for p in paths.values()):
            raise ValueError('unprotected source path')
        sources, excluded = load_sources(*(paths[name] for name in ('balanced', 'pilot', 'real')), reviewed=paths.get('reviewed'))
        sources = apply_policy(sources, manifest.get('label_policy', 'legacy'))
    if excluded != load_lines(folder / 'review/excluded.jsonl'):
        raise ValueError('exclusion accounting changed')
    audit = audit_rows(rows)
    if audit['exact_input_label_conflicts'] or audit['content_window_score_warnings']:
        raise ValueError('label conflicts found')
    near = load_lines(folder / 'review/near-duplicate-links.jsonl')
    grouping={'reaction_predicate':targetless} if manifest.get('distillation_preparation') else {}
    result = verify_splits(rows, sources, {r['id']: r['experiment_group'] for r in rows}, near, **grouping)
    if manifest.get('frozen_reference'):
        current = {r['id']: r for r in rows}
        count = 0
        for split in ('train', 'validation'):
            for old in load_lines(Path(manifest['frozen_reference']) / f'relevance/{split}.jsonl'):
                old = apply_policy([old], manifest.get('label_policy', 'legacy'))[0]
                new = current.get(old['id'])
                if new is None or new['split'] != split or new['state'] != old['state'] or new['score'] != old['score']:
                    raise ValueError('frozen reference changed')
                count += 1
        if count != manifest['frozen_reference_rows']:
            raise ValueError('frozen reference accounting mismatch')
        result['frozen_reference_rows_preserved'] = count
    hint_groups = defaultdict(set)
    for row in rows:
        for hint in row['experiment']['group_hints']:
            hint_groups[hint].add(row['experiment_group'])
    if any(len(groups) != 1 for groups in hint_groups.values()):
        raise ValueError('lineage/scenario group was separated')
    from tokenizers import Tokenizer
    tokenizer_path = ROOT / 'filtering_training/outputs/models/klue-roberta-base/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    lengths = [len(e.ids) for e in tokenizer.encode_batch([input_text(r['state'], 'relevance') for r in rows])]
    if max(lengths, default=0) > 512:
        raise ValueError('model input exceeds 512 tokens')
    result.update(purpose='Verify candidate relevance split; not human certification',
                  created_date=created_date(), rows=len(rows), excluded_rows=len(excluded),
                  maximum_tokens=max(lengths, default=0), source_hashes_unchanged=True,
                  lineage_and_scenario_groups_preserved=True, training_ready=False,
                  manifest_sha256=digest(folder / 'manifest.json'), tokenizer_sha256=digest(tokenizer_path))
    if write_report:
        target = folder / 'review/readiness-verification.json'
        if target.exists():
            raise ValueError('preserve prior verification; use a new experiment folder')
        write(target, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    verify(parser.parse_args().folder)
