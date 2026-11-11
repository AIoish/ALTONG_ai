"""용도: Encoder 학습의 누적 배치·손실·AMP 갱신과 분류 지표를 공통으로 처리한다.
생성일: 2026-11-11
"""

from contextlib import nullcontext
import random
import statistics


def task_loss(outputs, tasks, labels, weights, denominator, sample_weights=None):
    import torch
    loss = sum(logits.sum() * 0 for logits in outputs)
    for task, logits in enumerate(outputs):
        selected = tasks == task
        if selected.any():
            if sample_weights is None:
                total = torch.nn.functional.cross_entropy(logits[selected].float(), labels[selected], reduction='sum')
            else:
                per_row = torch.nn.functional.cross_entropy(logits[selected].float(), labels[selected], reduction='none')
                total = (per_row * sample_weights[selected]).sum()
            loss = loss + total * weights[task] / denominator
    return loss


def classification_metrics(gold, predicted, classes):
    from sklearn.metrics import confusion_matrix, f1_score
    if not gold or len(gold) != len(predicted):
        raise ValueError('nonempty matching predictions required')
    if any(type(v) is not int or not 1 <= v <= classes for values in (gold, predicted) for v in values):
        raise ValueError('class value out of range')
    correct = sum(g == p for g, p in zip(gold, predicted))
    return {'classes': classes, 'exact_correct': correct, 'exact_accuracy': correct / len(gold),
            'macro_f1': float(f1_score(gold, predicted, labels=list(range(1, classes + 1)), average='macro', zero_division=0)),
            'mae': sum(abs(g - p) for g, p in zip(gold, predicted)) / len(gold),
            'confusion_matrix': confusion_matrix(gold, predicted, labels=list(range(1, classes + 1))).tolist()}


def predict_logits(model, tokenizer, encoded, *, device='cuda', batch_size=8):
    import torch
    if not encoded or batch_size < 1:
        raise ValueError('nonempty encoded inputs and positive batch size required')
    model.eval()
    chunks = None
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            batch = tokenizer.pad(encoded[start:start + batch_size], padding=True, return_tensors='pt').to(device)
            outputs = model(**batch)
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.unbind(1)
            if chunks is None:
                chunks = [[] for _ in outputs]
            for head, logits in zip(chunks, outputs):
                head.append(logits.float().cpu())
    return tuple(torch.cat(head, dim=0) for head in chunks)


def train_epoch(model, tokenizer, records, *, optimizer, scheduler, scaler, weights,
                seed, device='cuda', effective_batch=16, micro_batch=4,
                release_cache=False, progress=None, row_weights=None):
    import torch
    if not records or not 0 < micro_batch <= effective_batch:
        raise ValueError('nonempty records and valid batch sizes required')
    if row_weights is not None and (len(row_weights) != len(records) or
            any(not 0 < float(weight) < float('inf') for weight in row_weights)):
        raise ValueError('one positive finite weight per record required')
    model.train()
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    losses, successful = [], 0
    for offset in range(0, len(order), effective_batch):
        indices = order[offset:offset + effective_batch]
        optimizer.zero_grad(set_to_none=True)
        if release_cache and str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        for micro in range(0, len(indices), micro_batch):
            batch = [records[i] for i in indices[micro:micro + micro_batch]]
            encoded = tokenizer.pad([b[0] for b in batch], padding=True, return_tensors='pt').to(device)
            tasks = torch.tensor([b[1] for b in batch], device=device)
            labels = torch.tensor([b[2] for b in batch], device=device)
            sample_weights = (torch.tensor([row_weights[i] for i in indices[micro:micro + micro_batch]], device=device)
                              if row_weights is not None else None)
            precision = torch.autocast('cuda', dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
            with precision:
                outputs = model(**encoded)
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.unbind(1)
                loss = task_loss(outputs, tasks, labels, weights, len(indices), sample_weights)
            scaler.scale(loss).backward()
            losses.append(loss.detach().item())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        previous = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous:
            scheduler.step()
            successful += 1
        if progress is not None and offset % 1600 == 0:
            progress(min(offset + effective_batch, len(order)))
    return {'loss_micro_mean': statistics.mean(losses), 'successful_updates': successful}
