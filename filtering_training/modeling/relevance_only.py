"""용도: 창 기반 v3 입력으로 KLUE 관련도 1~3만 학습하고 개발용 검증으로 체크포인트를 선택한다.
생성일: 2026-11-11
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

from filtering_training.common.artifact_io import ROOT, created_date, digest, load_lines, read, write
from filtering_training.common.encoder_input import INPUT_SCHEMA, encode_checked, input_text
from filtering_training.common.encoder_model import build_encoder
from filtering_training.common.encoder_training import classification_metrics, predict_logits, train_epoch
from filtering_training.preparation.curate_relevance import write_lines
from filtering_training.quality.verify_relevance_experiment import verify

BASE = ROOT / 'filtering_training/outputs/models/klue-roberta-base'
DATASET = ROOT / 'filtering_training/outputs/v19_relevance_reviewed_reactions_01'
DEFAULTS = {'seed': 42, 'epochs': 3, 'micro_batch': 2, 'effective_batch': 16,
            'eval_batch': 8, 'encoder_lr': 2e-5, 'head_lr': 1e-4, 'warmup_ratio': .1,
            'weight_decay': .01, 'patience': 2, 'attention': 'sdpa', 'device': 'cuda',
            'balance_origins': True}


def evaluate(rows, scores):
    result = classification_metrics([r['score'] for r in rows], scores, 3)
    result['rows'] = len(rows)
    result['relevant_miss_ids'] = [r['id'] for r, score in zip(rows, scores) if r['score'] == 3 and score != 3]
    result['false_relevant_ids'] = [r['id'] for r, score in zip(rows, scores) if r['score'] != 3 and score == 3]
    result['by_origin'] = {}
    for origin in sorted({r['experiment']['origin'] for r in rows}):
        selected = [(r, s) for r, s in zip(rows, scores) if r['experiment']['origin'] == origin]
        result['by_origin'][origin] = classification_metrics([r['score'] for r, _ in selected], [s for _, s in selected], 3)
    result['by_case_kind'] = {}
    for kind in sorted({r['case_kind'] for r in rows if r.get('case_kind')}):
        selected = [(r, s) for r, s in zip(rows, scores) if r.get('case_kind') == kind]
        result['by_case_kind'][kind] = dict(classification_metrics([r['score'] for r, _ in selected], [s for _, s in selected], 3), rows=len(selected))
    pairs = {}
    for row, score in zip(rows, scores):
        if row.get('invariance_group'):
            pairs.setdefault(row['invariance_group'], []).append((row['score'], score))
    complete = [values for values in pairs.values() if len(values) == 2]
    result['reaction_invariance'] = {'pairs': len(complete),
        'consistent': sum(len({s for _, s in values}) == 1 for values in complete),
        'both_correct': sum(all(g == s for g, s in values) for values in complete)}
    return result


def rank(report):
    # Each source gets one vote; 926 template rows must not drown out fresh cases.
    origin_mean = sum(r['macro_f1'] for r in report['by_origin'].values()) / len(report['by_origin'])
    return origin_mean, report['macro_f1'], report['exact_accuracy'], -report['mae']


def origin_weights(rows):
    counts = Counter(row['experiment'].get('loss_origin', row['experiment']['origin']) for row in rows)
    return {origin: len(rows) / (len(counts) * count) for origin, count in counts.items()}


def check_settings(settings):
    for name in ('epochs', 'micro_batch', 'effective_batch', 'eval_batch', 'patience'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    if settings['micro_batch'] > settings['effective_batch']:
        raise ValueError('micro batch exceeds effective batch')
    for name in ('encoder_lr', 'head_lr'):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError('learning rate must be finite and positive')
    if not 0 <= settings['warmup_ratio'] < 1 or settings['weight_decay'] < 0:
        raise ValueError('invalid optimizer settings')


def candidate_permission(manifest, allow_candidates):
    if manifest.get('input_schema') != INPUT_SCHEMA:
        raise ValueError('expected current previous-windows input schema')
    candidate = manifest.get('training_ready') is not True
    if candidate and not allow_candidates:
        raise ValueError('candidate labels are not approved gold; experimental use requires --allow-candidate-labels')
    return candidate


def prepare(dataset=DATASET, base=BASE, *, allow_candidates=False):
    from transformers import AutoTokenizer
    dataset, base = Path(dataset).resolve(), Path(base).resolve()
    manifest = read(dataset / 'manifest.json')
    candidate = candidate_permission(manifest, allow_candidates)
    # Reuse integrity and leakage checks without overwriting the prior review report.
    verification = verify(dataset, write_report=False)
    if read(base / 'config.json')['model_type'] != 'roberta':
        raise ValueError('this experiment requires a local KLUE RoBERTa base')
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    rows, encoded, maximum = {}, {}, {}
    for split in ('train', 'validation'):
        rows[split] = load_lines(dataset / f'relevance/{split}.jsonl')
        encoded[split], lengths = encode_checked(tokenizer, [input_text(r['state'], 'relevance') for r in rows[split]])
        maximum[split] = max(lengths)
    protected = dict(manifest['protected_sources'])
    for name, info in manifest['files'].items():
        protected[str(dataset / name)] = info['sha256']
    protected[str(dataset / 'manifest.json')] = digest(dataset / 'manifest.json')
    for path in base.iterdir():
        if path.is_file():
            protected[str(path)] = digest(path)
    counts = Counter(r['score'] for r in rows['train'])
    majority = max(range(1, 4), key=lambda s: (counts[s], -s))
    config = {'purpose': 'Relevance-only development experiment; no urgency head or final-test selection',
              'created_date': created_date(), 'input_schema': INPUT_SCHEMA, 'output_classes': {'relevance': 3},
              'dataset_dir': str(dataset), 'base_dir': str(base), 'candidate_labels_used': candidate,
              'label_status': 'mixed_assistant_and_user_candidates' if candidate else 'approved',
              'train_rows': len(rows['train']), 'validation_rows': len(rows['validation']),
              'score_counts': {split: dict(Counter(r['score'] for r in items)) for split, items in rows.items()},
              'maximum_tokens': maximum, 'truncated_rows': 0, 'training_ready_promoted': False,
              'loss': 'class-unweighted three-class CE; optional mean-one origin sample weights; score minus one',
              'pooling': 'first token', 'full_encoder_training': True, 'urgency_model_loaded': False,
              'checkpoint_selection': 'mean source macro F1, then overall macro F1, accuracy, lower MAE; earliest tie',
              'final_test_used': False, 'validation_is_independent_final_test': False,
              'majority_baseline': {'score': majority, 'validation': evaluate(rows['validation'], [majority] * len(rows['validation']))},
              'data_verification': verification, 'protected_hashes': protected,
              'code_hashes': {str(path.relative_to(ROOT)): digest(path) for path in (
                  Path(__file__).resolve(), ROOT / 'filtering_training/common/encoder_model.py',
                  ROOT / 'filtering_training/common/encoder_training.py',
                  ROOT / 'filtering_training/common/encoder_input.py',
                  ROOT / 'filtering_training/quality/verify_relevance_experiment.py')}}
    return rows, encoded, tokenizer, config


def assert_protected(config):
    for path, expected in config['protected_hashes'].items():
        if digest(path) != expected:
            raise ValueError('protected source changed: ' + path)


def predictions(model, tokenizer, encoded, *, device, batch_size):
    logits, = predict_logits(model, tokenizer, encoded, device=device, batch_size=batch_size)
    scores = (logits.argmax(-1) + 1).tolist()
    return scores, logits.softmax(-1).tolist()


def save_predictions(path, rows, scores, probabilities):
    write_lines(path, [{'id': row['id'], 'gold': row['score'], 'predicted': score,
                        'probabilities_1_2_3': probability, 'experiment_group': row['experiment_group'],
                        'origin': row['experiment']['origin'], 'label_source': row['experiment']['label_source'],
                        'case_kind': row.get('case_kind'), 'correct': row['score'] == score}
                       for row, score, probability in zip(rows, scores, probabilities)])


def reviewed_diagnostic(model, tokenizer, rows, encoded, run, prefix, settings):
    indices = [i for i, row in enumerate(rows) if row['experiment']['origin'] == 'reviewed']
    if not indices:
        return None
    selected = [rows[i] for i in indices]
    scores, probabilities = predictions(model, tokenizer, [encoded[i] for i in indices],
                                       device=settings['device'], batch_size=settings['eval_batch'])
    save_predictions(run / (prefix + '.reviewed-training.predictions.jsonl'), selected, scores, probabilities)
    return dict(evaluate(selected, scores), scope='training_fit_only_not_validation_or_selection')


def train(run, rows, encoded, tokenizer, config, settings):
    import numpy as np
    import torch
    from transformers import get_linear_schedule_with_warmup
    run = Path(run)
    check_settings(settings)
    if run.exists():
        raise ValueError('preserve existing run; choose a new folder')
    if settings['device'].startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; choose --device cpu explicitly')
    assert_protected(config)
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    torch.set_num_threads(4)
    device = settings['device']
    model = build_encoder(base=config['base_dir'], head_sizes=(3,), attention=settings['attention'],
                          required_model_type='roberta').to(device)
    model.encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': settings['encoder_lr']},
                                  {'params': model.heads.parameters(), 'lr': settings['head_lr']}],
                                 weight_decay=settings['weight_decay'], foreach=False)
    total_updates = math.ceil(len(rows['train']) / settings['effective_batch']) * settings['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * settings['warmup_ratio']), total_updates)
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
    records = [(item, 0, row['score'] - 1) for item, row in zip(encoded['train'], rows['train'])]
    weights = torch.ones(1, device=device)
    by_origin = origin_weights(rows['train']) if settings['balance_origins'] else {}
    row_weights = [by_origin[row['experiment'].get('loss_origin', row['experiment']['origin'])]
                   for row in rows['train']] if by_origin else None
    run.mkdir(parents=True)
    tokenizer.save_pretrained(run / 'tokenizer')
    model.encoder.config.save_pretrained(run / 'encoder')
    config = dict(config, settings=settings, status='training', origin_loss_weights=by_origin,
                  parameters=sum(p.numel() for p in model.parameters()),
                  hardware=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU',
                  torch=torch.__version__, transformers=__import__('transformers').__version__)
    write(run / 'run_config.json', config)
    best, stale, history = None, 0, []
    start = time.perf_counter()
    try:
        for epoch in range(1, settings['epochs'] + 1):
            epoch_start = time.perf_counter()
            trained = train_epoch(model, tokenizer, records, optimizer=optimizer, scheduler=scheduler,
                                  scaler=scaler, weights=weights, seed=settings['seed'] + epoch, device=device,
                                  effective_batch=settings['effective_batch'], micro_batch=settings['micro_batch'],
                                  row_weights=row_weights,
                                  progress=lambda count: print(json.dumps({'epoch': epoch, 'rows': count,
                                      'total': len(records), 'seconds': round(time.perf_counter() - start)}), flush=True))
            scores, probabilities = predictions(model, tokenizer, encoded['validation'], device=device, batch_size=settings['eval_batch'])
            report = evaluate(rows['validation'], scores)
            diagnostic = reviewed_diagnostic(model, tokenizer, rows['train'], encoded['train'], run, f'epoch-{epoch}', settings)
            history.append(dict(epoch=epoch, **trained, validation=report,
                                reviewed_training_diagnostic=diagnostic, seconds=time.perf_counter() - epoch_start))
            write(run / 'history.json', history)
            save_predictions(run / f'epoch-{epoch}.validation.predictions.jsonl', rows['validation'], scores, probabilities)
            if best is None or rank(report) > best:
                best, stale = rank(report), 0
                torch.save(model.state_dict(), run / 'model.pt')
                config.update(selected_epoch=epoch, selected_validation=report, selected_reviewed_training_diagnostic=diagnostic)
            else:
                stale += 1
            write(run / 'run_config.json', config)
            print(json.dumps({'epoch_complete': epoch, 'macro_f1': report['macro_f1'],
                              'accuracy': report['exact_accuracy'], 'mae': report['mae']}, ensure_ascii=False), flush=True)
            if stale >= settings['patience']:
                break
        model.load_state_dict(torch.load(run / 'model.pt', map_location=device, weights_only=True))
        scores, probabilities = predictions(model, tokenizer, encoded['validation'], device=device, batch_size=settings['eval_batch'])
        report = evaluate(rows['validation'], scores)
        if report != config['selected_validation']:
            raise ValueError('restored best checkpoint predictions changed')
        diagnostic = reviewed_diagnostic(model, tokenizer, rows['train'], encoded['train'], run, 'selected', settings)
        if diagnostic != config['selected_reviewed_training_diagnostic']:
            raise ValueError('restored training diagnostic changed')
        if diagnostic is not None:
            write(run / 'reviewed-training.json', diagnostic)
        save_predictions(run / 'validation.predictions.jsonl', rows['validation'], scores, probabilities)
        write(run / 'validation.json', report)
        write_lines(run / 'misclassified.jsonl', [dict(row, predicted=score, probabilities_1_2_3=probability)
                    for row, score, probability in zip(rows['validation'], scores, probabilities) if row['score'] != score])
        assert_protected(config)
        config.update(status='completed', epochs_completed=len(history), training_seconds=time.perf_counter() - start,
                      successful_updates=sum(h['successful_updates'] for h in history), checkpoint_sha256=digest(run / 'model.pt'))
        write(run / 'run_config.json', config)
    except Exception as error:
        config.update(status='failed', failure_type=type(error).__name__)
        write(run / 'run_config.json', config)
        raise
    return config


def load_run(run, *, device='cpu'):
    import torch
    from transformers import AutoTokenizer
    run = Path(run)
    torch.set_num_threads(4)
    config = read(run / 'run_config.json')
    if config['status'] != 'completed' or config['output_classes'] != {'relevance': 3}:
        raise ValueError('expected completed relevance-only run')
    if digest(run / 'model.pt') != config['checkpoint_sha256']:
        raise ValueError('checkpoint hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(str(run / 'tokenizer'), local_files_only=True)
    model = build_encoder(base=run / 'encoder', pretrained=False, head_sizes=(3,),
                          attention=config['settings']['attention'], required_model_type='roberta')
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True))
    model.to(device)
    return model, tokenizer, config


def evaluate_run(run, *, device='cpu', batch_size=8):
    model, tokenizer, config = load_run(run, device=device)
    assert_protected(config)
    rows = load_lines(Path(config['dataset_dir']) / 'relevance/validation.jsonl')
    encoded, _ = encode_checked(tokenizer, [input_text(r['state'], 'relevance') for r in rows])
    scores, _ = predictions(model, tokenizer, encoded, device=device, batch_size=batch_size)
    return evaluate(rows, scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('train', 'evaluate'), default='train')
    parser.add_argument('--dataset-dir', type=Path, default=DATASET)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--allow-candidate-labels', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--micro-batch', type=int, default=DEFAULTS['micro_batch'])
    parser.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    parser.add_argument('--no-origin-balance', action='store_true')
    args = parser.parse_args()
    if args.action == 'evaluate':
        if not args.run:
            parser.error('--run is required')
        print(json.dumps(evaluate_run(args.run, device=args.device), ensure_ascii=False, indent=2))
        return
    settings = dict(DEFAULTS, epochs=args.epochs, micro_batch=args.micro_batch, seed=args.seed,
                    device=args.device, balance_origins=not args.no_origin_balance)
    check_settings(settings)
    if args.run and args.run.exists():
        parser.error('existing run folders are preserved; choose a new --run')
    if not args.dry_run and args.run is None:
        parser.error('--run is required for training')
    rows, encoded, tokenizer, config = prepare(args.dataset_dir, args.base, allow_candidates=args.allow_candidate_labels)
    if args.dry_run:
        summary = {k: config[k] for k in ('input_schema', 'output_classes', 'candidate_labels_used', 'train_rows', 'validation_rows', 'maximum_tokens', 'urgency_model_loaded', 'final_test_used')}
        print(json.dumps(dict(summary, status='dry_run_passed', settings=settings), ensure_ascii=False, indent=2))
    else:
        train(args.run, rows, encoded, tokenizer, config, settings)


if __name__ == '__main__':
    main()
