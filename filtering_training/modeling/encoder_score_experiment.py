"""용도: 정제 점수 자료로 한국어 공유 Encoder를 학습하고 ONNX·INT8 CPU 성능을 비교한다.
생성일: 2026-10-10
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'filtering_training/outputs/models/klue-roberta-base'
PREPARED = ROOT / 'filtering_training/outputs/v7_relevance_ladder_01/prepared_laya_v2'
REVISION = '02f94ba5e3fcb7e2a58a390b8639b0fac974a8da'
TASKS = ('urgency', 'relevance')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def input_text(state, task):
    """Only observed fields; urgency cannot see any work context or variant id."""
    if task not in TASKS:
        raise ValueError('unknown task')
    notification = {k: v for k, v in state['notification'].items() if k != 'id'}
    value = {'notification': notification}
    if task == 'relevance':
        context = state['context']
        value['current_window'] = {k: context.get(k, '') for k in ('active_process', 'window_title')}
        value['recent_windows'] = [{k: w.get(k, '') for k in ('app_name', 'window_title')}
                                   for w in context.get('recent_windows', [])]
        value['recent_processes'] = context.get('recent_processes', [])
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def validation():
    from filtering_training.modeling.digit_score_experiment import load_validation_for_evaluation, VALIDATION_POOL
    from filtering_training.modeling.laya_score_experiment import request
    samples, groups, _ = load_validation_for_evaluation(VALIDATION_POOL)
    texts = [[input_text(request(s, task, prompt_version='v2')[0], task) for task in TASKS]
             for s in samples]
    return samples, groups, texts


def score(samples, predictions):
    from filtering_training.common.score_tasks import score_metrics
    from sklearn.metrics import confusion_matrix, f1_score
    metrics = score_metrics(samples, predictions)
    for task in TASKS:
        field = task + '_score'
        gold = [getattr(s.label, field) for s in samples]
        pred = [p[field] for p in predictions]
        metrics[field].update(macro_f1=f1_score(gold, pred, labels=[1, 2, 3, 4, 5], average='macro', zero_division=0),
                              confusion_matrix=confusion_matrix(gold, pred, labels=[1, 2, 3, 4, 5]).tolist())
    return metrics


def build_model(config=None, base=BASE, attention="eager"):
    import torch
    from transformers import AutoConfig, AutoModel

    class ScoreEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            model_config = AutoConfig.from_pretrained(str(base)) if config is None else config
            kwargs = {'attn_implementation': attention}
            if model_config.model_type == 'roberta':
                kwargs['add_pooling_layer'] = False
            self.encoder = (AutoModel.from_pretrained(str(base), **kwargs) if config is None
                            else AutoModel.from_config(config, **kwargs))
            self.dropout = torch.nn.Dropout(.1)
            self.heads = torch.nn.ModuleList([torch.nn.Linear(self.encoder.config.hidden_size, 5) for _ in TASKS])

        def forward(self, input_ids, attention_mask):
            states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if self.encoder.config.model_type == 'modernbert':
                weights = attention_mask.unsqueeze(-1).to(states.dtype)
                features = (states * weights).sum(1) / weights.sum(1).clamp_min(1)
            else:
                features = states[:, 0]
            features = self.dropout(features)
            return torch.stack([head(features) for head in self.heads], dim=1)

    return ScoreEncoder()


def encode_checked(tok, texts):
    encoded = tok(texts, truncation=False, padding=False, return_token_type_ids=False)
    lengths = [len(ids) for ids in encoded['input_ids']]
    if max(lengths, default=0) > 512:
        raise ValueError(f'input exceeds 512; no silent truncation: {max(lengths)}')
    return [{'input_ids': ids, 'attention_mask': mask}
            for ids, mask in zip(encoded['input_ids'], encoded['attention_mask'])], lengths


def train(run, base=BASE, micro_batch=4, attention="eager", fused_adam=False, release_cache=False):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from filtering_training.modeling.laya_score_experiment import load_native_prepared
    if run.exists():
        raise ValueError('preserve existing run; choose fresh directory')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    sources, protected = load_native_prepared(PREPARED.relative_to(ROOT))
    samples, groups, texts = validation()
    tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    records, lengths = [], []
    for task_id, task in enumerate(TASKS):
        encoded, sizes = encode_checked(tok, [input_text(r['state'], task) for r in sources[task]])
        records.extend((e, task_id, int(r['expected'][task]) - 1) for e, r in zip(encoded, sources[task]))
        lengths.extend(sizes)
    val_encoded, val_lengths = encode_checked(tok, [t for pair in texts for t in pair])
    # Full provenance includes base model, approved validation and CPU reference predictions.
    for artifact in base.iterdir():
        if artifact.is_file():
            protected[str(artifact)] = digest(artifact)
    for name in ('filtering_training/outputs/v5_validation_pool_01/manifest.json',
                 'filtering_training/outputs/v5_validation_pool_01/approval.json',
                 'filtering_training/outputs/runs/2026-10-10-qwen06-cpu-01/report.json',
                 'filtering_training/outputs/runs/2026-10-10-qwen06-cpu-01/report.predictions.jsonl'):
        protected[name] = digest(name)
    counts = [len(sources[t]) for t in TASKS]
    weights = [len(records) / (2 * n) for n in counts]
    run.mkdir(parents=True)
    cfg = {'purpose': 'shared Korean encoder, two separated inputs and five-class heads; CPU deployment comparison',
           'created_date': '2026-10-10', 'status': 'training', 'model': read(base / 'manifest.json').get('model', 'klue/roberta-base'),
           'model_revision': read(base / 'manifest.json').get('revision', REVISION), 'train_counts': dict(zip(TASKS, counts)), 'seed': 42,
           'epochs_max': 3, 'micro_batch': micro_batch, 'accumulation': 16 // micro_batch, 'effective_batch': 16,
           'encoder_lr': 2e-5, 'head_lr': 1e-4, 'weight_decay': .01, 'warmup_ratio': .1,
           'loss': 'five-class CE; mean-one task weights giving equal nominal total task contributions',
           'task_weights': dict(zip(TASKS, weights)), 'full_encoder_training': True,
           'max_sequence_length': 512, 'truncated_rows': 0, 'train_token_max': max(lengths),
           'validation_token_max': max(val_lengths), 'validation_rows': len(samples), 'final_test_used': False,
           'checkpoint_selection': 'policy correct, then fewer urgent false blocks, then mean exact score accuracy; earliest tie',
           'patience': 2, 'input_policy': 'urgency notification only; relevance observed current/recent app names and titles',
           'dataset_labels_changed': False, 'protected_hashes': protected,
           'attention_implementation': attention, 'adamw_fused': fused_adam, 'release_cuda_cache_per_update': release_cache,
           'cuda_allocator': os.environ.get('PYTORCH_ALLOC_CONF', 'default'),
           'pooling': 'masked mean' if read(base / 'config.json')['model_type'] == 'modernbert' else 'first token',
           'validation_groups': groups, 'hardware': torch.cuda.get_device_name(),
           'torch': torch.__version__, 'transformers': __import__('transformers').__version__}
    cfg['training_function_sha256'] = hashlib.sha256(__import__('inspect').getsource(train).encode()).hexdigest()
    cfg['executed_code_sha256'] = digest(__file__)
    model = build_model(base=base, attention=attention).cuda()
    model.encoder.gradient_checkpointing_enable()
    cfg['parameters'] = sum(p.numel() for p in model.parameters())
    write(run / 'run_config.json', cfg)
    tok.save_pretrained(run / 'tokenizer')
    model.encoder.config.save_pretrained(run / 'encoder')
    optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': 2e-5},
                                  {'params': model.heads.parameters(), 'lr': 1e-4}], weight_decay=.01, foreach=False, fused=fused_adam)
    updates = ((len(records) + 15) // 16) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * .1), updates)
    scaler = torch.amp.GradScaler('cuda')
    task_weights = torch.tensor(weights, device='cuda')
    best, stale, history, total_updates = None, 0, [], 0
    start = time.perf_counter()
    for epoch in range(1, 4):
        epoch_start = time.perf_counter()
        model.train()
        order = list(range(len(records)))
        random.Random(42 + epoch).shuffle(order)
        losses = []
        for offset in range(0, len(order), 16):
            ids = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            if release_cache:
                torch.cuda.empty_cache()
            for micro in range(0, len(ids), micro_batch):
                batch = [records[i] for i in ids[micro:micro + micro_batch]]
                x = tok.pad([b[0] for b in batch], padding=True, return_tensors='pt').to('cuda')
                tasks = torch.tensor([b[1] for b in batch], device='cuda')
                labels = torch.tensor([b[2] for b in batch], device='cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(**x)[torch.arange(len(batch), device='cuda'), tasks]
                    per_row = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='none')
                    loss = (per_row * task_weights[tasks]).sum() / len(ids)
                scaler.scale(loss).backward()
                losses.extend(per_row.detach().cpu().tolist())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                total_updates += 1
            if offset % 1600 == 0:
                print(json.dumps({'epoch': epoch, 'rows': min(offset + 16, len(order)),
                                  'train_ce': statistics.mean(losses[-1600:])}), flush=True)
        model.eval()
        all_logits = []
        with torch.inference_mode():
            for offset in range(0, len(val_encoded), 8):
                x = tok.pad(val_encoded[offset:offset + 8], padding=True, return_tensors='pt').to('cuda')
                all_logits.extend(model(**x).float().cpu().numpy())
        predictions = [{task + '_score': int(all_logits[2 * i + j][j].argmax()) + 1
                        for j, task in enumerate(TASKS)} for i in range(len(samples))]
        metrics = score(samples, predictions)
        rank = (metrics['policy_correct'], -metrics['urgent_false_blocks'],
                sum(metrics[t + '_score']['exact_accuracy'] for t in TASKS))
        row = {'epoch': epoch, 'train_ce': statistics.mean(losses), 'metrics': metrics,
               'seconds': time.perf_counter() - epoch_start, 'successful_updates_total': total_updates}
        history.append(row)
        write(run / 'history.json', history)
        write(run / f'epoch-{epoch}.json', row)
        write(run / f'epoch-{epoch}.predictions.json', [
            {'notification_id': s.notification.id, 'notification': s.notification.model_dump(mode='json'),
             'context': s.context.model_dump(mode='json'), 'gold': s.label.model_dump(mode='json'), 'prediction': p}
            for s, p in zip(samples, predictions)])
        print(json.dumps({'epoch_complete': epoch, 'policy': metrics['policy_accuracy'],
                          'urgent_false_blocks': metrics['urgent_false_blocks'], 'rank': rank}), flush=True)
        if best is None or rank > best:
            best, stale = rank, 0
            torch.save(model.cpu().state_dict(), run / 'model.pt')
            model.cuda()
            cfg.update(selected_epoch=epoch, selected_metrics=metrics)
        else:
            stale += 1
        if stale >= 2:
            break
    for path, sha in protected.items():
        if digest(path) != sha:
            raise ValueError('protected artifact changed: ' + path)
    cfg.update(status='completed', epochs_completed=len(history), training_seconds=time.perf_counter() - start,
               successful_updates=total_updates, checkpoint_sha256=digest(run / 'model.pt'),
               cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write(run / 'run_config.json', cfg)


def export(run, fp32_only=False):
    import torch
    import onnx
    from transformers import AutoConfig, AutoTokenizer
    from onnxruntime.quantization import quantize_dynamic, QuantType
    cfg = read(run / 'run_config.json')
    if cfg['status'] != 'completed' or digest(run / 'model.pt') != cfg['checkpoint_sha256']:
        raise ValueError('completed unchanged checkpoint required')
    model = build_model(AutoConfig.from_pretrained(run / 'encoder')).eval()
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True))
    tok = AutoTokenizer.from_pretrained(run / 'tokenizer')
    x = tok(['알림 확인', '현재 창 알림'], padding=True, return_token_type_ids=False, return_tensors='pt')
    path = run / 'model-fp32.onnx'
    if path.exists() or (run / 'model-int8.onnx').exists():
        raise ValueError('preserve existing exports')
    torch.onnx.export(model, (x['input_ids'], x['attention_mask']), str(path),
                      input_names=['input_ids', 'attention_mask'], output_names=['logits'],
                      dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'},
                                    'attention_mask': {0: 'batch', 1: 'sequence'}, 'logits': {0: 'batch'}},
                      opset_version=17, dynamo=False, external_data=False)
    onnx.checker.check_model(str(path))
    if not fp32_only:
        quantize_dynamic(str(path), str(run / 'model-int8.onnx'), weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul'], per_channel=True, reduce_range=True)
        onnx.checker.check_model(str(run / 'model-int8.onnx'))
    write(run / 'export.json', {'purpose': 'CPU ONNX exports of selected shared encoder',
          'created_date': '2026-10-10', 'source_checkpoint_sha256': cfg['checkpoint_sha256'],
          'quantization': None if fp32_only else 'dynamic per-channel signed INT8 MatMul weights; reduced range for AVX2',
          'onnx': onnx.__version__, 'onnxruntime': __import__('onnxruntime').__version__,
          'files': {name: {'bytes': (run / name).stat().st_size, 'sha256': digest(run / name)}
                    for name in (('model-fp32.onnx',) if fp32_only else ('model-fp32.onnx', 'model-int8.onnx'))}})


def cpu_worker(run, variant):
    """Fresh isolated CPU process; excludes Torch and reports whole-worker RSS."""
    import numpy as np
    import psutil
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(run / 'tokenizer/tokenizer.json'))
    samples, _, texts = validation()
    tokenized = tok.encode_batch([t for pair in texts for t in pair])
    if max(len(e.ids) for e in tokenized) > 512:
        raise ValueError('CPU input exceeds 512 tokens')
    encoded = [{'input_ids': e.ids, 'attention_mask': e.attention_mask} for e in tokenized]
    if 'torch' in sys.modules:
        raise ValueError('CPU measurement must not import Torch')
    process = psutil.Process()
    rss_before = process.memory_info().rss
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    start = time.perf_counter()
    session = ort.InferenceSession(str(run / f'model-{variant}.onnx'), options,
                                  providers=['CPUExecutionProvider'])
    load_seconds = time.perf_counter() - start

    def arrays(items):
        length = max(len(e['input_ids']) for e in items)
        pad_id = read(run / 'encoder/config.json')['pad_token_id']
        ids = np.full((len(items), length), pad_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, e in enumerate(items):
            ids[i, :len(e['input_ids'])] = e['input_ids']
            mask[i, :len(e['attention_mask'])] = e['attention_mask']
        return {'input_ids': ids, 'attention_mask': mask}

    def predict(index):
        # Two separate forward passes, like the Qwen CPU reference; pretokenized input.
        logits = [session.run(None, arrays([encoded[2 * index + j]]))[0][0, j] for j in range(2)]
        return {task + '_score': int(logits[j].argmax()) + 1 for j, task in enumerate(TASKS)}, logits

    predict(0)
    resources, stop = [], threading.Event()
    phase = ['idle']
    def monitor():
        process.cpu_percent()
        while not stop.wait(.2):
            resources.append({'phase': phase[0], 'cpu_percent': process.cpu_percent() / psutil.cpu_count(),
                              'rss_bytes': process.memory_info().rss})
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    time.sleep(30)
    phase[0] = 'active'
    reference = read(ROOT / 'filtering_training/outputs/runs/2026-10-10-qwen06-cpu-01/report.json')
    indices = reference['selected_validation_indices']
    if len(indices) != 100 or len(set(indices[:74])) != 74:
        raise ValueError('CPU reference must contain fixed 74 validation rows then repeats')
    rows, seconds, preds = [], [], {}
    cpu_start = sum(process.cpu_times()[:2])
    wall_start = time.perf_counter()
    for index in indices:
        start = time.perf_counter()
        prediction, logits = predict(index)
        elapsed = time.perf_counter() - start
        seconds.append(elapsed)
        if index in preds and prediction != preds[index]:
            raise ValueError('non-deterministic repeated prediction')
        preds[index] = prediction
        s = samples[index]
        rows.append({'validation_index': index, 'notification_id': s.notification.id,
                     'notification': s.notification.model_dump(mode='json'), 'context': s.context.model_dump(mode='json'),
                     'gold': s.label.model_dump(mode='json'), 'prediction': prediction,
                     'logits': [v.tolist() for v in logits], 'seconds': elapsed})
    cpu_average = (sum(process.cpu_times()[:2]) - cpu_start) / (time.perf_counter() - wall_start) / psutil.cpu_count() * 100
    stop.set()
    thread.join()
    metrics = score(samples, [preds[i] for i in range(len(samples))])
    training_config = read(run / 'run_config.json')
    selected_predictions = read(run / f"epoch-{training_config['selected_epoch']}.predictions.json")
    parity_mismatches = [samples[i].notification.id for i, row in enumerate(selected_predictions)
                         if row['prediction'] != preds[i]]
    report = {'purpose': '100 notification CPU benchmark, two separated score inputs per notification',
              'created_date': '2026-10-10', 'variant': variant, 'threads': 4, 'logical_cpus': psutil.cpu_count(),
              'providers': session.get_providers(), 'model_sha256': digest(run / f'model-{variant}.onnx'),
              'executed_code_sha256': digest(__file__),
              'metrics': metrics, 'mean_seconds': statistics.mean(seconds), 'p50_seconds': float(np.percentile(seconds, 50)),
              'p95_seconds': float(np.percentile(seconds, 95)), 'max_seconds': max(seconds),
              'cpu_average_percent': cpu_average, 'rss_before_session_bytes': rss_before,
              'peak_rss_bytes': max([r['rss_bytes'] for r in resources] + [process.memory_info().rss]),
              'idle_cpu_percent': statistics.mean([r['cpu_percent'] for r in resources if r['phase'] == 'idle']),
              'load_seconds': load_seconds, 'timing_scope': 'pretokenized padded arrays, two sequential ORT calls and argmax; no app input collection',
              'memory_scope': 'entire fresh CPU Python worker including tokenizer/runtime; not C# app; no Torch',
              'torch_imported': 'torch' in sys.modules,
              'pytorch_selected_score_pairs_match': len(parity_mismatches) == 0,
              'pytorch_selected_score_pair_mismatches': parity_mismatches,
              'final_test_used': False, 'resources': resources}
    write(run / f'cpu-{variant}.json', report)
    write(run / f'cpu-{variant}.predictions.json', rows)
    print(json.dumps({k: v for k, v in report.items() if k not in ('resources', 'metrics')}), flush=True)


def quantize_unsigned(run):
    """Compare full-range U8U8 after reduced-range signed INT8 changes policy decisions."""
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    target = run / 'model-int8-u8.onnx'
    if target.exists():
        raise ValueError('preserve existing unsigned export')
    manifest = read(run / 'export.json')
    if digest(run / 'model-fp32.onnx') != manifest['files']['model-fp32.onnx']['sha256']:
        raise ValueError('FP32 source changed')
    quantize_dynamic(str(run / 'model-fp32.onnx'), str(target), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=['MatMul'], per_channel=True, reduce_range=False)
    onnx.checker.check_model(str(target))
    manifest['unsigned_quantization'] = 'full-range dynamic per-channel UINT8 MatMul weights; U8U8 avoids AVX2 signed saturation'
    manifest['files'][target.name] = {'bytes': target.stat().st_size, 'sha256': digest(target)}
    write(run / 'export.json', manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('train', 'export', 'cpu', 'quantize-u8', 'all'), default='all')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--variant', choices=('fp32', 'int8', 'int8-u8'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--fp32-only', action='store_true')
    parser.add_argument('--micro-batch', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--attention', choices=('eager','sdpa'), default='eager')
    parser.add_argument('--fused-adam', action='store_true')
    parser.add_argument('--release-cuda-cache', action='store_true')
    args = parser.parse_args()
    if args.action == 'cpu':
        if args.variant is None:
            parser.error('--variant is required for CPU evaluation')
        cpu_worker(args.run, args.variant)
        return
    if args.action == 'quantize-u8':
        quantize_unsigned(args.run)
        return
    if args.action in ('train', 'all'):
        train(args.run, args.base, args.micro_batch, args.attention, args.fused_adam, args.release_cuda_cache)
    if args.action in ('export', 'all'):
        export(args.run, args.fp32_only)
    if args.action == 'all':
        env = dict(os.environ, USE_TORCH='0', USE_TF='0', TOKENIZERS_PARALLELISM='false')
        for variant in (('fp32',) if args.fp32_only else ('fp32', 'int8')):
            subprocess.run([sys.executable, '-X', 'utf8', '-m', 'filtering_training.modeling.encoder_score_experiment',
                            '--action', 'cpu', '--variant', variant, '--run', str(args.run)], check=True, env=env)


if __name__ == '__main__':
    main()
