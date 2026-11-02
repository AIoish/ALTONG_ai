"""용도: 고정된 새 평가 문장을 기존 base FP32 ONNX로 평가하고 알림 누락을 기록한다.
생성일: 2026-10-10
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from filtering_training.modeling.encoder_score_experiment import input_text
from src.filtering.policy import should_pass, POLICY_VERSION
from filtering_training.common.score_tasks import HistorySample

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('urgency', 'relevance')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def measures(rows):
    urgent = [r for r in rows if r['gold']['urgency_score'] >= 4]
    expected_pass = [r for r in rows if r['expected_pass']]
    expected_block = [r for r in rows if not r['expected_pass']]
    misses = [r['id'] for r in expected_pass if not r['actual_pass']]
    urgent_misses = [r['id'] for r in urgent if not r['actual_pass']]
    result = {'rows': len(rows), 'policy_correct': sum(r['expected_pass']==r['actual_pass'] for r in rows),
              'gold_pass_count': len(expected_pass), 'gold_block_count': len(expected_block),
              'policy_miss_ids': misses, 'false_pass_ids': [r['id'] for r in expected_block if r['actual_pass']],
              'urgent_count': len(urgent), 'urgent_policy_miss_ids': urgent_misses,
              'urgent_threshold_miss_ids': [r['id'] for r in urgent if r['prediction']['urgency_score'] < 4]}
    for task in TASKS:
        field = task+'_score'
        matrix = [[0]*5 for _ in range(5)]
        for r in rows:
            matrix[r['gold'][field]-1][r['prediction'][field]-1] += 1
        result[task] = {'exact_correct': sum(r['gold'][field]==r['prediction'][field] for r in rows),
                        'transitions': dict(Counter(f"{r['gold'][field]}->{r['prediction'][field]}" for r in rows)),
                        'confusion_matrix': matrix}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--model-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-predictions', type=Path,
                        help='Saved predictions of this checkpoint on the original 74 rows, for ONNX parity')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evaluation; choose fresh output folder')
    manifest = read(args.dataset_dir/'manifest.json')
    dataset = args.dataset_dir/'dataset.jsonl'
    if digest(dataset) != manifest['dataset_sha256']:
        raise ValueError('Frozen dataset changed')
    raw = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in raw:
        HistorySample.model_validate({k:row[k] for k in ('notification','context','label')})
    if len(raw) != manifest['rows'] or len({r['notification']['id'] for r in raw}) != len(raw):
        raise ValueError('Dataset row count or IDs invalid')
    model = args.model_run/'model-fp32.onnx'
    export = read(args.model_run/'export.json')
    if digest(model) != export['files']['model-fp32.onnx']['sha256']:
        raise ValueError('Model export changed')
    protected = {ROOT/path:hash_ for path,hash_ in manifest['protected_sources'].items()}
    for path in [model,args.model_run/'model.pt',args.model_run/'tokenizer/tokenizer.json']:
        protected[path] = digest(path)
    for path, expected in protected.items():
        if digest(path) != expected:
            raise ValueError(f'Source changed: {path}')
    tokenizer = Tokenizer.from_file(str(args.model_run/'tokenizer/tokenizer.json'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])

    def predict(state):
        scores, logits, probabilities, tokens = {}, [], [], []
        for j, task in enumerate(TASKS):
            encoded = tokenizer.encode(input_text(state, task))
            if len(encoded.ids) > 512:
                raise ValueError('Input exceeds 512 tokens; no silent truncation')
            arrays = {'input_ids':np.array([encoded.ids],dtype=np.int64),
                      'attention_mask':np.array([encoded.attention_mask],dtype=np.int64)}
            values = session.run(None, arrays)[0][0,j]
            scores[task+'_score'] = int(values.argmax())+1
            mass = np.exp(values-values.max())
            mass /= mass.sum()
            logits.append(values.tolist())
            probabilities.append(mass.tolist())
            tokens.append(len(encoded.ids))
        return scores, logits, probabilities, tokens

    # Check deployment serialization and inference against all saved original validation rows.
    reference = args.reference_predictions or args.model_run/'cpu-fp32.predictions.json'
    parity_rows = {r.get('validation_index',i):r for i,r in enumerate(read(reference))}
    mismatches = []
    for index, row in sorted(parity_rows.items()):
        prediction, _, _, _ = predict({'notification':row['notification'],'context':row['context']})
        if prediction != row['prediction']:
            mismatches.append(index)
    if mismatches:
        raise ValueError(f'Original 74-row deployment parity failed: {mismatches}')
    print(json.dumps({'original_validation_parity':len(parity_rows),'new_cases':len(raw)}), flush=True)
    results = []
    for index, row in enumerate(raw,1):
        started = time.perf_counter()
        prediction, logits, probabilities, tokens = predict(row)
        gold = row['label']
        results.append({'id':row['notification']['id'],'group':row.get('evaluation_metadata', {}).get('group', 'unclassified'),
                        'notification':row['notification'],'context':row['context'],'gold':gold,
                        'label_status':row.get('evaluation_metadata', {}).get('label_status', manifest['label_status']),'prediction':prediction,
                        'expected_pass':should_pass(gold['urgency_score'],gold['relevance_score']),
                        'actual_pass':should_pass(prediction['urgency_score'],prediction['relevance_score']),
                        'logits':logits,'probabilities':probabilities,'token_lengths':tokens,
                        'seconds':time.perf_counter()-started})
        if index % 25 == 0:
            print(json.dumps({'completed':index}), flush=True)
    for path, expected in protected.items():
        if digest(path) != expected:
            raise ValueError(f'Source changed during evaluation: {path}')
    report = {'purpose':'Frozen synthetic challenge evaluation of existing base; provisional labels',
              'created_date':'2026-10-10','dataset_sha256':manifest['dataset_sha256'],
              'model_sha256':digest(model),'policy_version':POLICY_VERSION,'model_run':str(args.model_run),
              'label_status':manifest['label_status'],'human_approved_rows':manifest.get('human_approved_rows', 0),
              'training_executed':False,'final_test_used':False,'threshold_changed':False,
              'original_validation_parity_rows':len(parity_rows),'protected_sources_unchanged':True,
              'parity_reference':str(reference),
              'metrics':measures(results),'by_group':{g:measures([r for r in results if r['group']==g]) for g in manifest.get('groups', sorted({r['group'] for r in results}))},
              'timing_scope':'Two CPU ONNX calls including serialization/tokenization; excludes app capture/queue/UI',
              'mean_seconds':sum(r['seconds'] for r in results)/len(results),
              'limitations':manifest['limitations']}
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    lines = ['<!-- 용도: 고정 합성 평가의 예측과 검수할 오판을 기록한다. 생성일: 2026-10-10 -->',
             '# 새 합성 평가 결과', '',
             'AI 제안 라벨 기준의 진단이며 실제 알림 정확도 또는 인간 승인 평가를 뜻하지 않는다.',
             '학습·모델·정책 변경 없음. 기존 검증 74행 예측 일치 확인.','',
             '| 유형 | 건수 | 통과/차단 일치 | 필요한 알림 누락 | 불필요한 통과 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for g,v in report['by_group'].items():
        lines.append(f"| {g} | {v['rows']} | {v['policy_correct']} | {len(v['policy_miss_ids'])} | {len(v['false_pass_ids'])} |")
    lines += ['', '## 정책상 누락 또는 불필요한 통과', '']
    for row in results:
        if row['expected_pass'] == row['actual_pass']:
            continue
        gold,pred = row['gold'],row['prediction']
        lines += [f"### {row['id']} / {'누락' if row['expected_pass'] else '불필요한 통과'}",f"- 본문: {row['notification']['body']}",
                  f"- 현재 창: {row['context']['window_title'] or '빈 창'}",
                  f"- 제안: {gold['urgency_score']}/{gold['relevance_score']} → 예측: {pred['urgency_score']}/{pred['relevance_score']}",'']
    lines += ['## 모든 문장별 점수', '', '| ID | 본문 | 제안 긴급/관련 | 예측 긴급/관련 | 정책 일치 |', '| --- | --- | --- | --- | --- |']
    for row in results:
        g,p = row['gold'],row['prediction']
        body = row['notification']['body'].replace('|','\\|')
        lines.append(f"| {row['id']} | {body} | {g['urgency_score']}/{g['relevance_score']} | {p['urgency_score']}/{p['relevance_score']} | {'O' if row['expected_pass']==row['actual_pass'] else 'X'} |")
    (args.output/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'metrics':report['metrics'],'by_group':report['by_group']},ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
