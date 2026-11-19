"""용도: 앱 develop 계약(단일 ONNX logits[batch,2,5], recent_processes 포함 입력)에 맞춰 긴급도5·관련도3 공동 인코더를 학습·내보낸다.
생성일: 2026-11-18
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time

from filtering_training.common.artifact_io import ROOT, created_date, digest, write
from filtering_training.common.csharp_tokenizer import encode as app_encode, load_rules, parity
from filtering_training.common.encoder_data import DEFAULT_DATASET, load_evaluation_inputs, load_windows_prepared
from filtering_training.common.encoder_input import APP_INPUT_SCHEMA, encode_checked, input_text
from filtering_training.common.encoder_model import build_encoder
from filtering_training.common.encoder_training import classification_metrics, predict_logits, train_epoch
from filtering_training.common.validation_data import evaluation_sets
from filtering_training.modeling import relevance_only
from filtering_training.modeling.joint_urgency5_relevance3 import OLD_SAFETY, baseline, body_key

BASES = {'klue': ROOT / 'filtering_training/outputs/models/klue-roberta-base',
         'kcelectra': ROOT / 'filtering_training/outputs/models/kcelectra-base-v2022-app-ids'}
STUDENT = ROOT / 'filtering_training/outputs/v22_real_seed_distillation_01/student-candidate-02'
TASKS = ('urgency', 'relevance')  # head order read by the app: head0 urgency, head1 relevance
DEFAULTS = {'seed': 42, 'epochs': 3, 'micro_batch': 2, 'effective_batch': 16, 'eval_batch': 8,
            'encoder_lr': 2e-5, 'head_lr': 1e-4, 'warmup_ratio': .1, 'weight_decay': .01,
            'attention': 'sdpa', 'device': 'cuda'}
# The unchanged app takes argmax over five slots and passes relevance >= 4, so relevance
# 1/2/3 is written to slots 1/2/5. Unused slots get a finite filler: the app rejects non-finite logits.
APP_RELEVANCE_SCORE = {1: 1, 2: 2, 3: 5}
FILLER_LOGIT = -1e4
URGENCY_GATE_SETS = ('validation', 'safety_development')


def app_passed(urgency, app_relevance):
    return urgency >= 4 or app_relevance >= 4


def urgency_report(rows, scores):
    gold = [r['label']['urgency_score'] for r in rows]
    ids = [r['notification']['id'] for r in rows]
    result = dict(classification_metrics(gold, scores, 5), rows=len(rows))
    result['threshold_miss_ids'] = [i for i, g, p in zip(ids, gold, scores) if g >= 4 and p < 4]
    result['false_urgent_ids'] = [i for i, g, p in zip(ids, gold, scores) if g < 4 and p >= 4]
    return result


def urgency_gate(reports, reference):
    """Pre-registered: no extra urgency threshold misses on the selection sets versus the app model."""
    checks = {name: {'misses': len(reports[name]['threshold_miss_ids']),
                     'reference_misses': len(reference[name]['threshold_miss_ids'])} for name in URGENCY_GATE_SETS}
    for check in checks.values():
        check['passed'] = check['misses'] <= check['reference_misses']
    return {'passed': all(c['passed'] for c in checks.values()), 'sets': checks}


def rank(gate, relevance):
    return (int(gate['passed']),) + relevance_only.rank(relevance)


def task_weights(records, row_weights):
    """Give each task equal total loss mass; rows keep their within-task weights."""
    mass = [sum(w for (_, task, _), w in zip(records, row_weights) if task == t) for t in range(len(TASKS))]
    if any(m <= 0 for m in mass):
        raise ValueError('both tasks need positive loss mass')
    return [len(records) / (len(TASKS) * m) for m in mass]


def app_logits(urgency, relevance):
    import torch
    filler = torch.full_like(relevance[:, :1], FILLER_LOGIT)
    expanded = torch.cat([relevance[:, :2], filler, filler, relevance[:, 2:3]], dim=1)
    return torch.stack([urgency, expanded], dim=1)


def prepare(encoder, teacher_share, dataset_dir=DEFAULT_DATASET):
    base = BASES[encoder]
    rows, encoded, tokenizer, config = relevance_only.prepare(STUDENT, base, allow_candidates=True, schema=APP_INPUT_SCHEMA)
    relevance_only.configure_distillation(rows, config, teacher_share)
    _, protected, stages = load_windows_prepared(dataset_dir)
    urgency_rows = [row for row, task in stages['safety'] if task == 'urgency']
    evals, evaluation_hashes = load_evaluation_inputs(dataset_dir, evaluation_sets())
    protected.update(evaluation_hashes)
    training_bodies = {body_key(r['state']) for r in urgency_rows}
    overlaps = {name: [r['notification']['id'] for r in items if body_key(r) in training_bodies] for name, items in evals.items()}
    if any(overlaps.values()):
        raise ValueError('urgency training/evaluation normalized-body overlap: ' + json.dumps(overlaps))
    urgency_texts = [input_text(r['state'], 'urgency', schema=APP_INPUT_SCHEMA) for r in urgency_rows]
    urgency_encoded, urgency_lengths = encode_checked(tokenizer, urgency_texts)
    eval_encoded = {name: {task: encode_checked(tokenizer, [input_text(r, task, schema=APP_INPUT_SCHEMA) for r in items])[0]
                           for task in TASKS} for name, items in evals.items()}
    texts = list(dict.fromkeys(
        [input_text(r['state'], 'relevance', schema=APP_INPUT_SCHEMA) for split in rows.values() for r in split] +
        urgency_texts + [input_text(r, t, schema=APP_INPUT_SCHEMA) for items in evals.values() for r in items for t in TASKS]))
    app_tokenizer = parity(base / 'tokenizer.json', texts, tokenizer)
    if app_tokenizer['mismatches']:
        raise ValueError('app C# tokenizer port differs from training tokenizer: ' + json.dumps(app_tokenizer))
    records = ([(e, 0, int(r['expected']['urgency']) - 1) for e, r in zip(urgency_encoded, urgency_rows)] +
               [(e, 1, r['score'] - 1) for e, r in zip(encoded['train'], rows['train'])])
    row_weights = [1.] * len(urgency_rows) + list(config.pop('row_loss_weights'))
    weights = task_weights(records, row_weights)
    for path in [OLD_SAFETY / 'model-fp32.onnx', Path(__file__).resolve(),
                 ROOT / 'filtering_training/common/csharp_tokenizer.py', ROOT / 'filtering_training/common/encoder_input.py']:
        protected[str(path)] = digest(path)
    config.update(
        purpose='Joint urgency5/relevance3 encoder for the unchanged app develop contract; development experiment',
        encoder=encoder, input_schema=APP_INPUT_SCHEMA, output_classes={'urgency': 5, 'relevance': 3},
        app_output={'name': 'logits', 'shape': ['batch', 2, 5], 'heads': list(TASKS),
                    'relevance_score_to_app': APP_RELEVANCE_SCORE, 'filler_logit': FILLER_LOGIT},
        app_policy='urgency >= 4 OR app_relevance >= 4 (unchanged app); equals urgency >= 4 OR relevance == 3',
        urgency_source='v13 safety-stage urgency rows (original 4946 + safety supplement 375 x4)',
        urgency_train_rows=len(urgency_rows), urgency_maximum_tokens=max(urgency_lengths),
        urgency_score_counts=dict(Counter(int(r['expected']['urgency']) for r in urgency_rows)),
        teacher_share=teacher_share, task_loss_weights=dict(zip(TASKS, weights)),
        eval_counts={k: len(v) for k, v in evals.items()}, urgency_normalized_body_overlap=overlaps,
        app_tokenizer_parity=app_tokenizer,
        checkpoint_selection='pre-registered: urgency gate (no extra threshold misses on validation/safety_development '
                             'vs app model) first, then relevance exact accuracy, macro F1, lower MAE; earliest tie',
        diagnostics_used_for_selection=False, single_seed=True, deployment_changed=False)
    config['protected_hashes'].update(protected)
    return {'rows': rows, 'encoded': encoded, 'tokenizer': tokenizer, 'config': config, 'records': records,
            'row_weights': row_weights, 'task_weights': weights, 'evals': evals, 'eval_encoded': eval_encoded}


def reference_reports(prepared):
    """The deployed app model (legacy input) as the urgency reference; its own app pass decisions."""
    from transformers import AutoTokenizer
    from filtering_training.common.encoder_input import LEGACY_INPUT_SCHEMA
    tokenizer = AutoTokenizer.from_pretrained(str(OLD_SAFETY / 'tokenizer'), local_files_only=True)
    legacy = evaluation_sets(schema=LEGACY_INPUT_SCHEMA)
    reports, passes = {}, {}
    for name, rows in prepared['evals'].items():
        values = baseline(legacy[name], tokenizer, OLD_SAFETY)
        reports[name] = urgency_report(rows, [v['urgency_score'] for v in values])
        passes[name] = [app_passed(v['urgency_score'], v['original_relevance5']) for v in values]
    return reports, passes


def evaluate_model(model, prepared, settings):
    tokenizer, device, batch = prepared['tokenizer'], settings['device'], settings['eval_batch']
    _, relevance_logits = predict_logits(model, tokenizer, prepared['encoded']['validation'], device=device, batch_size=batch)
    relevance_scores = (relevance_logits.argmax(-1) + 1).tolist()
    relevance = relevance_only.evaluate(prepared['rows']['validation'], relevance_scores)
    urgency, scores = {}, {}
    for name, rows in prepared['evals'].items():
        u, _ = predict_logits(model, tokenizer, prepared['eval_encoded'][name]['urgency'], device=device, batch_size=batch)
        _, r = predict_logits(model, tokenizer, prepared['eval_encoded'][name]['relevance'], device=device, batch_size=batch)
        scores[name] = {'urgency': (u.argmax(-1) + 1).tolist(), 'relevance': (r.argmax(-1) + 1).tolist()}
        urgency[name] = urgency_report(rows, scores[name]['urgency'])
    return relevance, relevance_scores, urgency, scores


def summary_line(epoch, relevance, urgency, gate):
    return {'epoch': epoch, 'relevance_correct': relevance['exact_correct'], 'relevance_rows': relevance['rows'],
            'relevance3_miss': len(relevance['relevant_miss_ids']), 'relevance3_false': len(relevance['false_relevant_ids']),
            'urgency_validation_correct': urgency['validation']['exact_correct'],
            'urgency_misses': {n: len(urgency[n]['threshold_miss_ids']) for n in URGENCY_GATE_SETS},
            'urgency_gate': gate['passed']}


def train(run, prepared, settings):
    import numpy as np
    import torch
    from transformers import get_linear_schedule_with_warmup
    run = Path(run)
    relevance_only.check_settings(dict(settings, patience=1))
    if run.exists():
        raise ValueError('preserve existing run; choose a new folder')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required')
    config = prepared['config']
    relevance_only.assert_protected(config)
    run.mkdir(parents=True)
    reference, reference_passes = reference_reports(prepared)
    write(run / 'reference-app-model.json', reference)
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    torch.set_num_threads(4)
    device = settings['device']
    model = build_encoder(base=BASES[config['encoder']], head_sizes=(5, 3), attention=settings['attention'],
                          required_model_type=relevance_only.SUPPORTED_MODEL_TYPES).to(device)
    model.encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': settings['encoder_lr']},
                                  {'params': model.heads.parameters(), 'lr': settings['head_lr']}],
                                 weight_decay=settings['weight_decay'], foreach=False)
    records = prepared['records']
    total_updates = math.ceil(len(records) / settings['effective_batch']) * settings['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * settings['warmup_ratio']), total_updates)
    scaler = torch.amp.GradScaler('cuda')
    weights = torch.tensor(prepared['task_weights'], device=device)
    tokenizer = prepared['tokenizer']
    tokenizer.save_pretrained(run / 'tokenizer')
    model.encoder.config.save_pretrained(run / 'encoder')
    config = dict(config, settings=settings, status='training', planned_updates=total_updates,
                  parameters=sum(p.numel() for p in model.parameters()), hardware=torch.cuda.get_device_name(),
                  torch=torch.__version__, transformers=__import__('transformers').__version__)
    write(run / 'run_config.json', config)
    best, history, start = None, [], time.perf_counter()
    try:
        for epoch in range(1, settings['epochs'] + 1):
            epoch_start = time.perf_counter()
            trained = train_epoch(model, tokenizer, records, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                  weights=weights, seed=settings['seed'] + epoch, device=device,
                                  effective_batch=settings['effective_batch'], micro_batch=settings['micro_batch'],
                                  row_weights=prepared['row_weights'],
                                  progress=lambda count: print(json.dumps({'epoch': epoch, 'rows': count, 'total': len(records),
                                                                           'seconds': round(time.perf_counter() - start)}), flush=True))
            relevance, relevance_scores, urgency, scores = evaluate_model(model, prepared, settings)
            gate = urgency_gate(urgency, reference)
            history.append(dict(epoch=epoch, **trained, relevance=relevance, urgency=urgency, urgency_gate=gate,
                                seconds=time.perf_counter() - epoch_start))
            write(run / 'history.json', history)
            write(run / f'epoch-{epoch}.predictions.json', {'relevance_validation': relevance_scores, 'evaluation_sets': scores})
            if best is None or rank(gate, relevance) > best:
                best = rank(gate, relevance)
                torch.save(model.state_dict(), run / 'model.pt')
                config.update(selected_epoch=epoch, selected_relevance=relevance, selected_urgency=urgency,
                              selected_urgency_gate=gate, selected_scores=scores)
            write(run / 'run_config.json', config)
            print(json.dumps(summary_line(epoch, relevance, urgency, gate), ensure_ascii=False), flush=True)
        model.load_state_dict(torch.load(run / 'model.pt', map_location=device, weights_only=True))
        relevance, _, urgency, scores = evaluate_model(model, prepared, settings)
        if relevance != config['selected_relevance'] or scores != config['selected_scores']:
            raise ValueError('restored best checkpoint predictions changed')
        new_passes = {name: [app_passed(u, APP_RELEVANCE_SCORE[r]) for u, r in zip(s['urgency'], s['relevance'])]
                      for name, s in scores.items()}
        config['app_pass_changes_vs_app_model'] = {
            name: {'rows': len(p), 'reference_passed': sum(reference_passes[name]), 'new_passed': sum(p),
                   'changed': sum(a != b for a, b in zip(p, reference_passes[name]))} for name, p in new_passes.items()}
        config.update(status='trained', epochs_completed=len(history), training_seconds=time.perf_counter() - start,
                      successful_updates=sum(h['successful_updates'] for h in history),
                      checkpoint_sha256=digest(run / 'model.pt'))
        write(run / 'run_config.json', config)
        config['app_package'] = export_app_package(model, run, prepared, config)
        relevance_only.assert_protected(config)
        config.update(status='completed', protected_sources_unchanged=True)
        write(run / 'run_config.json', config)
    except Exception as error:
        config.update(status='failed', failure_type=type(error).__name__, failure=str(error)[:500])
        write(run / 'run_config.json', config)
        raise
    return config


def export_app_package(model, run, prepared, config):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    folder = run / 'app-package'
    folder.mkdir()
    model = model.cpu().eval()
    model.encoder.set_attn_implementation('eager')

    class AppContract(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            urgency, relevance = self.inner(input_ids, attention_mask)
            return app_logits(urgency, relevance)

    path = folder / 'model-fp32.onnx'
    tokenizer = prepared['tokenizer']
    sample = tokenizer(['알림 확인', '현재 작업 알림'], padding=True, return_tensors='pt', return_token_type_ids=False)
    torch.onnx.export(AppContract(model).eval(), (sample['input_ids'], sample['attention_mask']), str(path),
                      input_names=['input_ids', 'attention_mask'], output_names=['logits'],
                      dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'},
                                    'logits': {0: 'batch'}}, opset_version=17, dynamo=False, external_data=False)
    onnx.checker.check_model(str(path))
    base = BASES[config['encoder']]
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        shutil.copyfile(base / name, folder / name)
    rules = load_rules(folder / 'tokenizer.json')
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    names = ([i.name for i in session.get_inputs()], [(o.name, o.shape) for o in session.get_outputs()])

    def app_score(text, head):
        # Same steps as KlueFilter.Run: C# token ids, batch 1, all-ones mask, finite check, first maximum.
        ids = np.asarray([app_encode(rules, text)], dtype=np.int64)
        logits = session.run(['logits'], {'input_ids': ids, 'attention_mask': np.ones_like(ids)})[0]
        if logits.shape != (1, 2, 5) or not np.isfinite(logits).all():
            raise ValueError('app contract violated: logits shape or non-finite values')
        return int(np.argmax(logits[0, head])) + 1

    mismatches, times = [], []
    expected = config['selected_scores']
    for name, rows in prepared['evals'].items():
        for i, row in enumerate(rows):
            begin = time.perf_counter()
            u = app_score(input_text(row, 'urgency', schema=APP_INPUT_SCHEMA), 0)
            r = app_score(input_text(row, 'relevance', schema=APP_INPUT_SCHEMA), 1)
            times.append(time.perf_counter() - begin)
            want = (expected[name]['urgency'][i], APP_RELEVANCE_SCORE[expected[name]['relevance'][i]])
            if (u, r) != want:
                mismatches.append({'set': name, 'id': row['notification']['id'], 'onnx': [u, r], 'torch': list(want)})
    validation_rows = prepared['rows']['validation']
    relevance_mismatch = 0
    _, logits = predict_logits(model, tokenizer, prepared['encoded']['validation'], device='cpu', batch_size=8)
    torch_scores = (logits.argmax(-1) + 1).tolist()
    for row, score in zip(validation_rows, torch_scores):
        if app_score(input_text(row['state'], 'relevance', schema=APP_INPUT_SCHEMA), 1) != APP_RELEVANCE_SCORE[score]:
            relevance_mismatch += 1
    report = {'created_date': created_date(), 'onnx_inputs_outputs': names, 'evaluation_rows': len(times),
              'evaluation_score_mismatches': mismatches, 'relevance_validation_rows': len(validation_rows),
              'relevance_validation_mismatches': relevance_mismatch,
              'mean_ms_two_calls': statistics.mean(times) * 1000, 'p95_ms_two_calls': float(np.percentile(times, 95)) * 1000,
              'csharp_executed': False, 'tokenization': 'Python port of app KlueTokenizer (common/csharp_tokenizer.py)'}
    if mismatches or relevance_mismatch:
        write(folder / 'verification.json', report)
        raise ValueError('ONNX app-contract predictions differ from PyTorch')
    manifest = {'purpose': 'Drop-in candidate for models/filtering/klue-base-safety-v1 (app code unchanged); not approved',
                'created_date': created_date(), 'version': f"{config['encoder']}-joint-app-{run.name}",
                'encoder': config['encoder'], 'model_run': run.name,
                'model_sha256': digest(path), 'tokenizer_sha256': digest(folder / 'tokenizer.json'),
                'model_bytes': path.stat().st_size, 'onnxruntime': ort.__version__,
                'inputs': {'input_ids': 'Int64 [batch,sequence]', 'attention_mask': 'Int64 [batch,sequence]'},
                'output': 'logits float [batch,2,5]', 'heads': list(TASKS), 'max_tokens': 512,
                'input_schema': APP_INPUT_SCHEMA, 'relevance_classes_trained': 3,
                'relevance_score_to_app': APP_RELEVANCE_SCORE, 'policy': 'urgency >= 4 OR relevance >= 4',
                'deployment_approved': False, 'csharp_executed': False}
    write(folder / 'manifest.json', manifest)
    write(folder / 'verification.json', report)
    return dict(report, manifest=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=sorted(BASES), required=True)
    parser.add_argument('--teacher-share', type=float, default=0.)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.dry_run and args.run is None:
        parser.error('--run is required for training')
    if args.run and args.run.exists():
        parser.error('existing run folders are preserved; choose a new --run')
    prepared = prepare(args.encoder, args.teacher_share)
    settings = dict(DEFAULTS, seed=args.seed)
    config = prepared['config']
    if args.dry_run:
        keys = ('encoder', 'input_schema', 'train_rows', 'validation_rows', 'urgency_train_rows', 'maximum_tokens',
                'urgency_maximum_tokens', 'urgency_score_counts', 'teacher_share', 'task_loss_weights', 'eval_counts',
                'app_tokenizer_parity', 'candidate_labels_used')
        print(json.dumps(dict({k: config[k] for k in keys}, settings=settings, training_mixture=config['training_mixture'],
                              records=len(prepared['records']), status='dry_run_passed'), ensure_ascii=False, indent=2))
        return
    train(args.run, prepared, settings)


if __name__ == '__main__':
    main()
