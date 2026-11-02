"""용도: 기존 데이터로 Laya 점수를 학습하고 승인 검증에서 Qwen 결과와 비교한다.
생성일: 2026-10-09
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from filtering_training.common.digit_scores import replace_prompt
from filtering_training.common.score_tasks import messages, score_metrics
from filtering_training.modeling.digit_score_experiment import (
    SOURCE, VALIDATION_POOL, load_validation_for_evaluation,
)
from filtering_training.preparation.prepare_score_experiment import digest, write_json, write_lines

MODEL_ID = "convaiinnovations/laya-multilingual"
MODEL_REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"
MODEL_PATH = Path("filtering_training/outputs/models/laya-multilingual")
PREPARED = Path("filtering_training/outputs/v5_training_pool_01/prepared_continual")
BASELINE = Path("filtering_training/outputs/runs/2026-10-05-qwen06-accumulated-01/after.json")
LEVELS = {
    "urgency": [
        "언제 확인해도 무방", "나중에 확인 가능", "비교적 빠른 확인 필요",
        "현재 작업을 중단하고 확인할 가치", "즉시 확인 또는 대응 필요",
    ],
    "relevance": [
        "작업과 무관", "같은 조직이나 분야지만 거의 무관", "같은 주제지만 작업에 주는 도움 불분명",
        "현재 작업 목적에 직접 도움", "정확한 작업 대상이나 행동과 연결",
    ],
}
# Laya reads a bounded question head, not a generative chat system prompt.
LAYA_INSTRUCTIONS_V2 = {
    "urgency": (
        "알림의 현재 사건·시점·지연 영향을 근거로 긴급도 하나를 선택하세요. "
        "작업 관련성은 별개입니다. 이미 해결된 사건과 앞으로의 일정을 구분하세요. "
        "발신자·광고 마감만으로 높이지 말고 명시된 사실로 판단하세요."
    ),
    "relevance": (
        "알림이 현재 작업 대상·행동에 주는 도움을 선택하세요. 현재 창이 우선입니다. "
        "최근 창은 현재 목적의 보조 근거이며, 현재 제목이 모호할 때 작업 대상을 확인하는 데 쓰세요. "
        "목적을 알 수 없으면 추측하지 마세요. 단어 일치보다 실제 도움을 판단하세요."
    ),
}
LAYA_LEVELS_V2 = {
    "urgency": [
        "확인 시점이 자유로운 선택적 정보",
        "나중에 확인·대응 가능, 당장 영향 없음",
        "비교적 빠른 확인 필요, 즉각 중단 근거 없음",
        "현재 확인할 가치, 진행 중인 일이 응답을 기다림",
        "즉시 대응 필요, 진행 중인 안전 위험·중대한 피해",
    ],
    "relevance": [
        "현재 목적과 무관하거나 목적을 확인할 근거 없음",
        "같은 분야의 주변 정보·장식 광고, 작업 도움 거의 없음",
        "같은 주제의 잡담·일반 정보, 직접 도움 불분명",
        "현재 목적에 직접 유용한 자료·방법·비교 정보",
        "정확한 작업 대상·행동의 요청·결과·해결 절차",
    ],
}


def question(task, instructions, reverse=False, prompt_version="legacy"):
    if prompt_version not in {"legacy", "v2"}:
        raise ValueError("unknown Laya prompt version")
    levels = LEVELS if prompt_version == "legacy" else LAYA_LEVELS_V2
    if prompt_version == "v2":
        instructions = LAYA_INSTRUCTIONS_V2[task]
    order = range(5, 0, -1) if reverse else range(1, 6)
    return {"type": "choice", "instructions": instructions,
            "criteria": {str(i): levels[task][i - 1] for i in order}}


def request(sample, task, reverse=False, prompt_version="legacy"):
    converted = replace_prompt(messages(sample, task), task)
    payload = json.loads(converted[1]["content"])
    if "id" in payload["notification"] or (task == "urgency" and "context" in payload):
        raise ValueError("Laya input must preserve Qwen's context and id separation")
    return payload, {task: question(task, converted[0]["content"], reverse, prompt_version)}


def resolve_prompt_version(model_path, requested=None):
    if requested is not None:
        return requested
    run_file = model_path.parent / "run_config.json"
    if run_file.is_file():
        return json.loads(run_file.read_text(encoding="utf-8")).get("prompt_version", "legacy")
    return "v2"


def question_budget(tok, task, questions):
    """Expose instruction/option clipping that the SDK usage truncation flag omits."""
    from laya.common import build_head, _encode_question_text, render_options
    from laya.train import to_internal
    q = to_internal(task, questions[task])
    ids, markers, stats = build_head(tok, q, 256)
    full = len(_encode_question_text(tok, q["t"] + " question: " + q["ins"], add_special_tokens=False))
    option_lengths = [len(_encode_question_text(tok, " " + text, add_special_tokens=False))
                      for text in render_options(q)]
    return {"instruction_tokens": full, "retained_instruction_tokens": markers[0] - 2,
            "head_tokens": len(ids), "option_tokens": option_lengths,
            "clipped": full != markers[0] - 2 or max(option_lengths) > 48
                       or stats["tokens_per_option"] is not None}


def preflight(model_path, baseline):
    samples, groups, _ = load_validation_for_evaluation(VALIDATION_POOL)
    before = json.loads(baseline.read_text(encoding="utf-8"))
    prediction_file = baseline.with_suffix(".predictions.jsonl")
    if before["split"] != "validation" or before["dataset_sha256"] != digest(SOURCE / "dataset.jsonl"):
        raise ValueError("Qwen baseline uses a different source or split")
    if before["predictions_sha256"] != digest(prediction_file):
        raise ValueError("Qwen baseline predictions changed")
    rows = [json.loads(line) for line in prediction_file.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(samples):
        raise ValueError("baseline and Laya must use the same validation rows")
    for sample, row in zip(samples, rows):
        if (row["notification_id"] != sample.notification.id
                or row["notification"] != sample.notification.model_dump(mode="json")
                or row["context"] != sample.context.model_dump(mode="json")
                or row["gold"] != sample.label.model_dump(mode="json")):
            raise ValueError("baseline validation input or label differs")
    if score_metrics(samples, [row["prediction"] for row in rows]) != before["metrics"]:
        raise ValueError("baseline metrics differ from saved predictions")
    train_counts, hashes = {}, {str(baseline): digest(baseline), str(prediction_file): digest(prediction_file)}
    for task in ("urgency", "relevance"):
        folder = PREPARED / task
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("training_ready") is False or manifest.get("quality_gate", {}).get("pending"):
            raise ValueError("existing training view is not ready")
        for split in ("train", "validation"):
            path = folder / (split + ".jsonl")
            hashes[str(path)] = digest(path)
            if hashes[str(path)] != manifest["splits"][split]["file_sha256"]:
                raise ValueError("existing training view changed")
        train_counts[task] = manifest["splits"]["train"]["count"]
    for path in [SOURCE / "dataset.jsonl", SOURCE / "prepared/test.jsonl",
                 VALIDATION_POOL / "dataset.jsonl", VALIDATION_POOL / "approval.json"]:
        hashes[str(path)] = digest(path)
    model_config = json.loads((model_path / "rl_agent_config.json").read_text(encoding="utf-8"))
    trained = model_config.get("fine_tuned", False)
    if trained:
        run_file = model_path.parent / "run_config.json"
        run = json.loads(run_file.read_text(encoding="utf-8"))
        if run.get("status") != "completed" or run.get("base_revision") != MODEL_REVISION:
            raise ValueError("trained Laya provenance is incomplete")
        if digest(model_path / "model.safetensors") != run["checkpoint_sha256"]:
            raise ValueError("trained Laya weights changed")
        for path, expected in run["preserved_hashes"].items():
            if digest(Path(path)) != expected:
                raise ValueError("training source changed since Laya fine-tuning")
        hashes[str(run_file)] = digest(run_file)
    else:
        revision_file = model_path / ".cache/huggingface/download/model.safetensors.metadata"
        if not revision_file.is_file() or revision_file.read_text(encoding="utf-8").splitlines()[0] != MODEL_REVISION:
            raise ValueError("multilingual checkpoint download revision differs from the comparison plan")
    model_files = [model_path / name for name in (
        "model.safetensors", "rl_agent_config.json", "encoder/config.json",
        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")]
    for path in model_files:
        if not path.is_file():
            raise ValueError(f"download the pinned multilingual checkpoint first: {path}")
        hashes[str(path)] = digest(path)
    return samples, groups, before, {
        "purpose": "read-only Laya comparison preflight; no training, label changes or test inference",
        "created_date": datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
        "model_id": MODEL_ID, "download_revision": MODEL_REVISION, "model_path": str(model_path),
        "training_status": "project-fine-tuned Laya" if trained else "Laya has not been trained on this project; Qwen baseline is trained",
        "existing_train_counts": train_counts, "validation_rows": len(samples),
        "baseline_metrics": before["metrics"], "preserved_hashes": hashes,
        "output_mode": "choice among five score labels; two separate inputs per notification",
        "pending_v7_data_included": False, "final_test_used": False,
    }


def training_row(record, task, prompt_version="legacy"):
    converted = replace_prompt(record["messages"], task)
    payload = json.loads(converted[1]["content"])
    if "id" in payload["notification"] or (task == "urgency" and "context" in payload):
        raise ValueError("training data leaks variant id or work context")
    gold = json.loads(record["messages"][-1]["content"])[task + "_score"]
    if type(gold) is not int or not 1 <= gold <= 5:
        raise ValueError("training score is outside the original label range")
    return {"state": payload, "questions": {task: question(task, converted[0]["content"], prompt_version=prompt_version)},
            "expected": {task: str(gold)}}


def load_native_prepared(folder):
    """Verify frozen v2 files and keep task labels and context separation intact."""
    manifest_file = folder / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("format") != "native_laya_choice_v2" or manifest.get("prompt_version") != "v2":
        raise ValueError("expected native Laya v2 prepared inputs")
    hashes = dict(manifest["protected_hashes"])
    hashes[str(manifest_file)] = digest(manifest_file)
    sources = {}
    for task in ("urgency", "relevance"):
        source = folder / task / "train.jsonl"
        expected = manifest["files"][task + "/train.jsonl"]
        hashes[str(source)] = expected["sha256"]
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        if len(records) != expected["count"] or len(records) != manifest["counts"][task]["total"]:
            raise ValueError("native prepared count changed")
        for row in records:
            if (set(row) != {"state", "questions", "expected"}
                    or row["questions"] != {task: question(task, "", prompt_version="v2")}
                    or set(row["expected"]) != {task}
                    or row["expected"][task] not in {"1", "2", "3", "4", "5"}):
                raise ValueError("native prepared question or target changed")
            if ("id" in row["state"]["notification"]
                    or set(row["state"]) != ({"notification"} if task == "urgency" else {"notification", "context"})):
                raise ValueError("native prepared context separation changed")
        sources[task] = records
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"native prepared protected input changed: {path}")
    return sources, hashes


def task_loss_weights(counts, urgency_share=None):
    """Normalize task loss weights to mean one without duplicating training rows."""
    if urgency_share is None:
        return {task: 1.0 for task in ("urgency", "relevance")}
    if not 0 < urgency_share < 1 or any(counts.get(task, 0) <= 0 for task in ("urgency", "relevance")):
        raise ValueError("urgency loss share must be strictly between zero and one with both tasks present")
    total = sum(counts.values())
    return {"urgency": urgency_share * total / counts["urgency"],
            "relevance": (1 - urgency_share) * total / counts["relevance"]}


def train_task_weighted(model, tok, items, config, device, max_len, head_max_len, weights):
    """Reuse the SDK loop and RNG exactly; weight each single-item CE loss locally."""
    from unittest.mock import patch
    import laya.train as sdk
    if config.micro_batch != 1 or config.loss != "soft-ce":
        raise ValueError("task-weighted comparison requires micro_batch=1 and soft-ce")
    original_encode, original_loss = sdk.encode_item, sdk.soft_ce_loss
    pending = []
    consumed = Counter()

    def encode(tokenizer, item, *args, **kwargs):
        if pending:
            raise ValueError("SDK encoded another item before consuming the previous loss")
        task = item["_filtering_task"]
        encoded = original_encode(tokenizer, item, *args, **kwargs)
        pending.append(task)
        return encoded

    def loss(*args, **kwargs):
        if len(pending) != 1:
            raise ValueError("SDK loss is not paired with one training item")
        task = pending.pop()
        consumed[task] += 1
        return original_loss(*args, **kwargs) * weights[task]

    # Context managers restore the SDK even on failure; no installed package is edited.
    with patch.object(sdk, "encode_item", encode), patch.object(sdk, "soft_ce_loss", loss):
        history = sdk.train_model(model, tok, items, config, device, max_len, head_max_len)
    expected = Counter(item["_filtering_task"] for item in items)
    expected = Counter({task: count * config.epochs for task, count in expected.items()})
    if pending or consumed != expected:
        raise ValueError("weighted loss did not consume the exact training population")
    return history


def train(output, model_path, baseline, device, epochs=1, dry_run=False, prompt_version="v2", prepared_inputs=None,
          urgency_loss_share=None):
    """LoRA on attention plus decision heads; original prepared rows stay on disk unchanged."""
    if output.exists():
        raise ValueError("choose a new training run directory")
    _, _, _, plan = preflight(model_path, baseline)
    if plan["training_status"] == "project-fine-tuned Laya":
        raise ValueError("this first comparison starts from the pinned base checkpoint")
    native_sources = None
    if prepared_inputs is not None:
        if prompt_version != "v2":
            raise ValueError("native prepared inputs require v2 questions")
        native_sources, hashes = load_native_prepared(prepared_inputs)
        plan["preserved_hashes"].update(hashes)
    import laya
    import torch
    from dataclasses import asdict
    from laya.train import TrainConfig, items_from_rows, load_checkpoint, train_model, save_checkpoint
    from laya.common import build_head, serialize_state
    from transformers import AutoTokenizer
    config = TrainConfig(epochs=epochs, micro_batch=1, grad_accum=8, encoder_lr=5e-5,
                         head_lr=1e-4, min_lr=1e-6, weight_decay=.01, loss="soft-ce",
                         shuffle_options=("choice",), calib_max=0, calib_frac=0,
                         seed=42, max_len=1024, head_max_len=256,
                         amp=True, gradient_checkpointing=True, log_every=100)
    config.validate()
    tok = AutoTokenizer.from_pretrained(str(model_path / "tokenizer"), local_files_only=True)
    if prompt_version == "v2":
        for task in ("urgency", "relevance"):
            qs = {task: question(task, "", prompt_version="v2")}
            if question_budget(tok, task, qs)["clipped"]:
                raise ValueError("v2 question instructions or criteria would be clipped")
    rows, converted_hash, counts = [], hashlib.sha256(), {}
    for task in ("urgency", "relevance"):
        source = PREPARED / task / "train.jsonl"
        records = native_sources[task] if native_sources is not None else [
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        counts[task] = len(records)
        if native_sources is None and counts[task] != plan["existing_train_counts"][task]:
            raise ValueError("training row count differs from approved view")
        for record in records:
            row = record if native_sources is not None else training_row(record, task, prompt_version)
            state_ids = tok(serialize_state(row["state"]).replace(tok.mask_token, " "),
                            add_special_tokens=False)["input_ids"]
            from laya.train import to_internal
            head, _, stats = build_head(tok, to_internal(task, row["questions"][task]), 256)
            if stats["options_distinct"] != 5 or len(state_ids) + len(head) + 3 > 1024:
                raise ValueError("Laya would truncate an original training input or its options")
            converted_hash.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            rows.append(row)
    items, skipped = items_from_rows(tok, rows, 1024, 256)
    if skipped or len(items) != sum(counts.values()):
        raise ValueError("Laya must consume every original prepared training row")
    weights = task_loss_weights(counts, urgency_loss_share)
    for item, row in zip(items, rows):
        if len(row["questions"]) != 1:
            raise ValueError("each prepared row must contain exactly one task")
        item["_filtering_task"] = next(iter(row["questions"]))
    run = {"purpose": "one shared Laya encoder for separated urgency and relevance inputs",
           "created_date": datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
           "status": "prepared", "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
           "prompt_version": prompt_version,
           "laya_version": laya.__version__, "train_counts": counts, "train_items": len(items),
           "expected_updates": epochs * ((len(items) + 7) // 8), "training_config": asdict(config),
           "training_view_sha256": converted_hash.hexdigest(), "preserved_hashes": plan["preserved_hashes"],
           "validation_rows_used_for_training": 0, "final_test_used": False, "v7_proposals_used": prepared_inputs is not None,
           "prepared_inputs": str(prepared_inputs) if prepared_inputs is not None else None,
           "task_loss_weights": weights, "urgency_loss_share": urgency_loss_share,
           "task_weighting": "per-row CE weighting; no resampling; mean row weight one",
           "sdk_train_loop_sha256": hashlib.sha256(__import__("inspect").getsource(train_model).encode()).hexdigest(),
           "calibration": "none; no validation calibration or confidence gating",
           "lora": {"r": 8, "alpha": 16, "dropout": .05, "targets": "encoder attention Wqkv and Wo"}}
    if dry_run:
        print(json.dumps({k: v for k, v in run.items() if k != "preserved_hashes"}, ensure_ascii=False, indent=2))
        return run
    torch.manual_seed(42)
    from peft import LoraConfig, get_peft_model
    model, tok, cfg = load_checkpoint(str(model_path))
    targets = [name for name, _ in model.encoder.named_modules()
               if name.endswith(("attn.Wqkv", "attn.Wo"))]
    if not targets:
        raise ValueError("no multilingual encoder attention modules found")
    model.encoder = get_peft_model(model.encoder, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                                          target_modules=targets, bias="none"))
    for parameter in model.act_head.parameters():
        parameter.requires_grad_(False)
    run["lora"]["target_modules"] = targets
    run["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run["status"] = "training"
    run["hardware"] = torch.cuda.get_device_name() if device.startswith("cuda") else "CPU"
    output.mkdir(parents=True)
    write_json(output / "run_config.json", run)
    started = time.perf_counter()
    if urgency_loss_share is None:
        losses = train_model(model, tok, items, config, torch.device(device), 1024, 256)
    else:
        losses = train_task_weighted(model, tok, items, config, torch.device(device), 1024, 256, weights)
    run["training_seconds"] = time.perf_counter() - started
    run["epoch_losses"] = losses
    run["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
    for path, expected in run["preserved_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError("a protected input changed during training")
    # Export an ordinary Laya checkpoint so inference requires no PEFT adapter switching.
    model.encoder = model.encoder.merge_and_unload()
    cfg = dict(cfg, fine_tuned=True, max_len=1024, head_max_len=256,
               training=dict(cfg.get("training", {}), project_lora=run["lora"], project_train=asdict(config)))
    save_checkpoint(model, tok, cfg, str(output / "checkpoint"))
    # Normalize SDK/transformers compatibility before evaluation snapshots file hashes.
    from laya.agent import _fix_tokenizer_config
    _fix_tokenizer_config(str(output / "checkpoint"))
    run["checkpoint_sha256"] = digest(output / "checkpoint/model.safetensors")
    run["status"] = "completed"
    write_json(output / "run_config.json", run)
    print(json.dumps({k: run[k] for k in ("status", "train_items", "expected_updates", "epoch_losses",
                                        "training_seconds", "cuda_peak_allocated_bytes")}, ensure_ascii=False))
    return run


def evaluate(output, model_path, baseline, device, limit=0, reverse=False, dry_run=False, prompt_version=None):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    samples, groups, before, plan = preflight(model_path, baseline)
    prompt_version = resolve_prompt_version(model_path, prompt_version)
    if limit > len(samples):
        raise ValueError("limit exceeds the fixed validation set")
    selected = samples[:limit] if limit else samples
    plan.update(evaluated_rows=len(selected), requested_device=device, reversed_options=reverse,
                prompt_version=prompt_version, question_budget={},
                checkpoint_training_prompt_version=resolve_prompt_version(model_path),
                prompt_matches_checkpoint_default=prompt_version == resolve_prompt_version(model_path))
    if output.exists() or output.with_suffix(".predictions.jsonl").exists():
        raise ValueError("choose a fresh output path")
    if dry_run:
        print(json.dumps({k: v for k, v in plan.items() if k not in {"preserved_hashes", "baseline_metrics"}},
                         ensure_ascii=False, indent=2))
        return plan
    import laya
    import torch
    started = time.perf_counter()
    agent = laya.load(str(model_path), device=device, backend="eager")
    load_seconds = time.perf_counter() - started

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def predict(sample):
        prediction, answers = {}, {}
        for task in ("urgency", "relevance"):
            state, questions = request(sample, task, reverse, prompt_version)
            budget_key = task + ("_history" if hasattr(sample.context, "recent_windows") else "_current")
            if budget_key not in plan["question_budget"]:
                budget = question_budget(agent.tok, task, questions)
                plan["question_budget"][budget_key] = budget
                plan.setdefault("questions", {})[budget_key] = questions[task]
                if prompt_version == "v2" and budget["clipped"]:
                    raise ValueError("v2 question instructions or criteria would be clipped")
            response = agent.predict(state, questions, lang="ko", max_len=1024)
            if response.get("usage", {}).get("truncated") or response.get("usage", {}).get("options"):
                raise ValueError("Laya truncated the input or collapsed score options")
            answer = response["answers"][task]
            choice = str(answer["choice"])
            if choice not in {"1", "2", "3", "4", "5"}:
                raise ValueError("Laya returned an unknown score label")
            prediction[task + "_score"] = int(choice)
            answers[task] = response
        return prediction, answers

    predict(selected[0])  # Warmup is excluded from notification latency.
    rows, predictions, timings = [], [], []
    for sample in selected:
        sync()
        started = time.perf_counter()
        prediction, answers = predict(sample)
        sync()
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        predictions.append(prediction)
        rows.append({"notification_id": sample.notification.id,
                     "notification": sample.notification.model_dump(mode="json"),
                     "context": sample.context.model_dump(mode="json"), "gold": sample.label.model_dump(mode="json"),
                     "prediction": prediction, "laya_responses": answers, "end_to_end_seconds": elapsed})
    for path, expected in plan["preserved_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"a protected input changed during evaluation: {path}")
    full = len(selected) == len(samples)
    report = dict(plan, laya_version=laya.__version__, device=device,
                  hardware=torch.cuda.get_device_name() if device.startswith("cuda") else "CPU",
                  model_parameters=sum(p.numel() for p in agent.model.parameters()),
                  load_seconds=load_seconds, mean_seconds=statistics.mean(timings),
                  median_seconds=statistics.median(timings),
                  metrics=score_metrics(selected, predictions), full_validation=full,
                  timing_comparison="Historical Qwen timing is contextual; not a fresh paired benchmark",
                  qwen_historical_mean_seconds=before["mean_seconds"] if full else None)
    report["metrics_by_set"] = {
        name: score_metrics([s for s in selected if s.notification.id in ids],
                            [p for s, p in zip(selected, predictions) if s.notification.id in ids])
        for name, ids in groups.items() if any(s.notification.id in ids for s in selected)}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_lines(output.with_suffix(".predictions.jsonl"), rows)
    report["predictions_sha256"] = digest(output.with_suffix(".predictions.jsonl"))
    write_json(output, report)
    print(json.dumps({"metrics": report["metrics"], "mean_seconds": report["mean_seconds"]}, ensure_ascii=False))
    return report


def diagnose_train(output, model_path, baseline, device, per_score=100, dry_run=False):
    """Read-only fit check on deterministic, score-balanced original training rows."""
    if per_score <= 0:
        raise ValueError("per-score must be positive")
    if output.exists() or output.with_suffix(".predictions.jsonl").exists():
        raise ValueError("choose a fresh diagnostic output path")
    _, _, _, plan = preflight(model_path, baseline)
    if plan["training_status"] != "project-fine-tuned Laya":
        raise ValueError("fit diagnostics require a completed project checkpoint")
    run = json.loads((model_path.parent / "run_config.json").read_text(encoding="utf-8"))
    prompt_version = run.get("prompt_version", "legacy")
    native_inputs = run.get("prepared_inputs")
    if native_inputs:
        sources, hashes = load_native_prepared(Path(native_inputs))
        plan["preserved_hashes"].update(hashes)
    else:
        sources = {task: [json.loads(line) for line in (PREPARED / task / "train.jsonl").read_text(encoding="utf-8").splitlines()]
                   for task in ("urgency", "relevance")}
    ordered_sources = [(task, index) for task in sources for index in range(len(sources[task]))]
    if run.get("excluded_relevance_rows", 0):
        parent_file = model_path.parent.parent / "run_config.json"
        parent = json.loads(parent_file.read_text(encoding="utf-8"))
        if parent["status"] != "completed" or parent["selected_policy"] != "exclude":
            raise ValueError("two-phase training provenance is incomplete")
        phases = parent["proposed_alternatives"]["exclude"]["phases"]
        excluded = {row["source_index"] for row in parent["conflicting_relevance_rows"]}
        validate_half_indices(phases, {task: len(rows) for task, rows in sources.items()}, excluded)
        if len(excluded) != run["excluded_relevance_rows"]:
            raise ValueError("checkpoint exclusion count differs")
        stages = {"phase50": 1, "phase100": 2}
        if model_path.parent.name not in stages:
            raise ValueError("unknown two-phase checkpoint")
        ordered_sources = [(task, index) for phase in phases[:stages[model_path.parent.name]]
                           for task in sources for index in phase["source_indices"][task]]
        plan["preserved_hashes"][str(parent_file)] = digest(parent_file)
        plan["excluded_relevance_rows"] = len(excluded)
    converted_rows = {}
    converted_hash = hashlib.sha256()
    for task, index in ordered_sources:
        row = sources[task][index] if native_inputs else training_row(sources[task][index], task, prompt_version)
        converted_hash.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        converted_rows[task, index] = row
    if converted_hash.hexdigest() != run["training_view_sha256"]:
        raise ValueError("converted inputs differ from the actual completed training run")
    selected, sampling = [], {}
    for task in ("urgency", "relevance"):
        buckets, input_labels, counts = defaultdict(dict), defaultdict(set), Counter()
        for row_task, index in ordered_sources:
            if row_task != task:
                continue
            row = converted_rows[task, index]
            gold = int(row["expected"][task])
            key = json.dumps({k: row[k] for k in ("state", "questions")}, ensure_ascii=False, sort_keys=True)
            input_labels[key].add(gold)
            counts[gold] += 1
            buckets[gold].setdefault(key, (task, index, row))
        if sum(counts.values()) != run["train_counts"][task] or set(counts) != set(range(1, 6)):
            raise ValueError("training population differs or a score class is missing")
        rng = random.Random(42)
        for score in range(1, 6):
            candidates = list(buckets[score].values())
            selected.extend(rng.sample(candidates, min(per_score, len(candidates))))
        sampling[task] = {"population_by_score": dict(counts),
                          "unique_inputs_by_score": {i: len(buckets[i]) for i in range(1, 6)},
                          "conflicting_unique_inputs": sum(len(labels) > 1 for labels in input_labels.values()),
                          "selected_by_score": {i: min(per_score, len(buckets[i])) for i in range(1, 6)}}
    plan.update(purpose="training-fit diagnosis; sampled training rows, not generalization accuracy",
                split="train", sampling=sampling, sampling_seed=42, per_score=per_score,
                evaluated_task_rows=len(selected), additional_training=False,
                training_view_sha256=run["training_view_sha256"], requested_device=device)
    plan["prompt_version"] = prompt_version
    if dry_run:
        print(json.dumps({k: v for k, v in plan.items() if k not in {"preserved_hashes", "baseline_metrics"}},
                         ensure_ascii=False, indent=2))
        return plan
    import laya
    import torch
    agent = laya.load(str(model_path), device=device, backend="eager")
    predictions = []
    for n, (task, index, row) in enumerate(selected, 1):
        response = agent.predict(row["state"], row["questions"], lang="ko", max_len=1024)
        if response.get("usage", {}).get("truncated") or response.get("usage", {}).get("options"):
            raise ValueError("diagnostic input was truncated or its score options collapsed")
        choice = str(response["answers"][task]["choice"])
        if choice not in {"1", "2", "3", "4", "5"}:
            raise ValueError("invalid diagnostic score")
        predictions.append({"task": task, "source_index": index, "state": row["state"],
                            "questions": row["questions"], "gold": int(row["expected"][task]),
                            "prediction": int(choice), "laya_response": response})
        if n % 100 == 0:
            print(f"diagnostic {n}/{len(selected)}", flush=True)
    metrics = {}
    for task in ("urgency", "relevance"):
        rows = [r for r in predictions if r["task"] == task]
        matrix = [[sum(r["gold"] == gold and r["prediction"] == pred for r in rows)
                   for pred in range(1, 6)] for gold in range(1, 6)]
        metrics[task] = {"count": len(rows), "exact_correct": sum(r["gold"] == r["prediction"] for r in rows),
                         "exact_accuracy": sum(r["gold"] == r["prediction"] for r in rows) / len(rows),
                         "threshold_accuracy": sum((r["gold"] >= 4) == (r["prediction"] >= 4) for r in rows) / len(rows),
                         "confusion_gold_rows_prediction_columns_1_to_5": matrix,
                         "prediction_distribution": dict(Counter(r["prediction"] for r in rows))}
    for path, expected in plan["preserved_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"a protected input changed during diagnosis: {path}")
    plan.update(metrics=metrics, hardware=torch.cuda.get_device_name() if device.startswith("cuda") else "CPU")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_lines(output.with_suffix(".predictions.jsonl"), predictions)
    plan["predictions_sha256"] = digest(output.with_suffix(".predictions.jsonl"))
    write_json(output, plan)
    print(json.dumps({"sampling": sampling, "metrics": metrics}, ensure_ascii=False))
    return plan


def validate_half_indices(phases, counts, excluded):
    if len(phases) != 2:
        raise ValueError("exactly two training phases are required")
    if any(type(i) is not int or not 0 <= i < counts["relevance"] for i in excluded):
        raise ValueError("invalid exclusion index")
    for task, count in counts.items():
        parts = [phase["source_indices"][task] for phase in phases]
        if any(len(part) != len(set(part)) or any(type(i) is not int for i in part) for part in parts):
            raise ValueError("duplicate or invalid phase indices")
        if set(parts[0]) & set(parts[1]):
            raise ValueError("training phases overlap")
        expected = set(range(count)) - (excluded if task == "relevance" else set())
        if set(parts[0]) | set(parts[1]) != expected:
            raise ValueError("training phases omit or add source rows")
        if abs(len(parts[0]) - len(parts[1])) > max(len(expected) * .03, 2):
            raise ValueError("training phases are not approximately equal")
    if set(phases[0]["groups"]) & set(phases[1]["groups"]):
        raise ValueError("declared training groups overlap")


def train_halves(output, model_path, baseline, device, dry_run=False):
    """One pass in two grouped halves; keep optimizer, LR and RNG across evaluation."""
    import copy
    import math
    from dataclasses import asdict
    import laya
    import torch
    from laya.train import (TrainConfig, items_from_rows, load_checkpoint, save_checkpoint,
                            encode_item, draw_option_order, _forward, soft_ce_loss)
    from laya.common import collate_items, build_sequence
    from laya.agent import _fix_tokenizer_config
    from peft import LoraConfig, get_peft_model
    _, _, _, base_plan = preflight(model_path, baseline)
    if base_plan["training_status"] == "project-fine-tuned Laya":
        raise ValueError("quantity comparison must start from the pinned base model")
    config_file = output / "run_config.json"
    run = json.loads(config_file.read_text(encoding="utf-8"))
    if run["status"] not in {"awaiting_conflict_policy", "prepared"}:
        raise ValueError("choose an unstarted prepared two-phase run")
    for name, expected in run["preserved_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"prepared training input changed: {name}")
    phases = run["proposed_alternatives"]["exclude"]["phases"]
    excluded = {r["source_index"] for r in run["conflicting_relevance_rows"]}
    if len(excluded) != 4161:
        raise ValueError("exclusions differ from the user-approved 4,161 rows")
    counts = base_plan["existing_train_counts"]
    validate_half_indices(phases, counts, excluded)
    sources = {task: [json.loads(s) for s in (PREPARED / task / "train.jsonl").read_text(encoding="utf-8").splitlines()]
               for task in ("urgency", "relevance")}
    for task in sources:
        if len(sources[task]) != counts[task]:
            raise ValueError("source count changed")
    for row in run["conflicting_relevance_rows"]:
        source = sources["relevance"][row["source_index"]]
        payload = json.loads(source["messages"][1]["content"])
        if (payload["notification"]["id"] != row["notification_id"]
                or json.loads(source["messages"][-1]["content"])["relevance_score"] != 1):
            raise ValueError("exclusion no longer matches its audited source")
    config = TrainConfig(epochs=1, micro_batch=1, grad_accum=8, encoder_lr=5e-5,
                         head_lr=1e-4, min_lr=1e-6, weight_decay=.01, loss="soft-ce",
                         shuffle_options=("choice",), calib_max=0, calib_frac=0, seed=42,
                         max_len=1024, head_max_len=256, amp=device.startswith("cuda"),
                         gradient_checkpointing=True, log_every=100)
    config.validate()
    torch.manual_seed(42)
    model, tok, cfg = load_checkpoint(str(model_path))
    for task in sources:
        if question_budget(tok, task, {task: question(task, "", prompt_version="v2")})["clipped"]:
            raise ValueError("v2 question is clipped")
    phase_rows, phase_items, body_phase = [], [], {}
    for phase_index, phase in enumerate(phases):
        rows = []
        for task in sources:
            for source_index in phase["source_indices"][task]:
                row = training_row(sources[task][source_index], task, "v2")
                body = "".join(row["state"]["notification"]["body"].split()).casefold()
                if body in body_phase and body_phase[body] != phase_index:
                    raise ValueError("same notification body appears in both phases")
                body_phase[body] = phase_index
                rows.append(row)
        items, skipped = items_from_rows(tok, rows, 1024, 256)
        if skipped or len(items) != len(rows):
            raise ValueError("a selected training row was skipped")
        for item in items:
            _, _, stats, truncation = build_sequence(tok, None, item["q"], 1024, 256,
                                         state_ids=item["state_ids"], return_stats=True, return_truncation_stats=True)
            if truncation["truncated"] or stats["options_distinct"] != 5:
                raise ValueError("selected training input is clipped")
        phase_rows.append(rows)
        phase_items.append(items)
    expected_updates = sum(math.ceil(len(items) / 8) for items in phase_items)
    if dry_run:
        summary = {"excluded_relevance_rows": len(excluded), "phases": [p["train_counts"] for p in phases],
                   "expected_updates": expected_updates, "training_config": asdict(config)}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    targets = [name for name, _ in model.encoder.named_modules() if name.endswith(("attn.Wqkv", "attn.Wo"))]
    if len(targets) != 44:
        raise ValueError("unexpected encoder LoRA targets")
    model.encoder = get_peft_model(model.encoder, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
                                                          target_modules=targets, bias="none"))
    for parameter in model.act_head.parameters():
        parameter.requires_grad_(False)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()
    groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad], "lr": config.encoder_lr},
              {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad], "lr": config.head_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=expected_updates, eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda") if config.amp else None
    parameters = [p for g in groups for p in g["params"]]
    run.update(status="training_50", selected_policy="exclude", excluded_relevance_rows=len(excluded),
               exclusion_authorization="user: exclude incorrect 4,161 rows and compare two training stages",
               training_config=asdict(config), expected_updates=expected_updates, laya_version=laya.__version__,
               trainable_parameters=sum(p.numel() for p in parameters), phase_results=[],
               optimizer_continuity=True, scheduler_continuity=True,
               training_used_validation=False, final_test_used=False)
    write_json(config_file, run)
    order_rng = random.Random(42)
    updates, successful_updates, cumulative_seconds, view_hash = 0, 0, 0.0, hashlib.sha256()
    for stage, (rows, items) in enumerate(zip(phase_rows, phase_items), 1):
        phase_name = "phase50" if stage == 1 else "phase100"
        run["status"] = "training_50" if stage == 1 else "training_100"
        write_json(config_file, run)
        ordered = list(items)
        random.Random(42 + stage - 1).shuffle(ordered)
        total_loss, started = 0.0, time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for index, item in enumerate(ordered):
            encoded = encode_item(tok, item, 1024, 256, draw_option_order(item, order_rng, ("choice",)))
            batch = collate_items([[encoded]], tok.pad_token_id)
            logits = _forward(model, batch, torch.device(device), config.amp, False)
            loss = soft_ce_loss(logits, batch["target"].to(device), batch["marker_mask"].to(device))
            window_start = (index // 8) * 8
            scaled = loss / min(8, len(ordered) - window_start)
            if scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            if (index + 1) % 8 == 0 or index + 1 == len(ordered):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if scaler:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    successful_updates += int(scaler.get_scale() >= old_scale)
                else:
                    optimizer.step()
                    successful_updates += 1
                scheduler.step()
                updates += 1
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            if (index + 1) % 100 == 0:
                print(f"{phase_name} {index + 1}/{len(ordered)} loss {loss.item():.4f}", flush=True)
        phase_seconds = time.perf_counter() - started
        cumulative_seconds += phase_seconds
        for row in rows:
            view_hash.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        for name, expected in run["preserved_hashes"].items():
            if digest(Path(name)) != expected:
                raise ValueError(f"protected source changed: {name}")
        # Export a copy; live LoRA weights, optimizer and scheduler stay intact for stage 2.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if config.amp else None
        model.eval()
        exported = copy.deepcopy(model).cpu()
        exported.encoder = exported.encoder.merge_and_unload()
        phase_dir = output / phase_name
        save_cfg = dict(cfg, fine_tuned=True, max_len=1024, head_max_len=256,
                        training={"project_train": asdict(config), "cumulative_updates": updates,
                                  "prompt_version": "v2", "stage": phase_name})
        save_checkpoint(exported, tok, save_cfg, str(phase_dir / "checkpoint"))
        del exported
        _fix_tokenizer_config(str(phase_dir / "checkpoint"))
        if config.amp:
            torch.cuda.empty_cache()
        phase_run = {"purpose": "cumulative half-data v2 Laya checkpoint", "created_date": run["created_date"],
                     "status": "completed", "base_revision": MODEL_REVISION, "prompt_version": "v2",
                     "preserved_hashes": run["preserved_hashes"], "source_train_counts": counts,
                     "train_counts": {t: sum(phases[j]["train_counts"][t] for j in range(stage)) for t in counts},
                     "training_view_sha256": view_hash.hexdigest(), "cumulative_updates": updates,
                     "successful_optimizer_updates": successful_updates, "training_seconds": cumulative_seconds,
                     "phase_training_seconds": phase_seconds, "phase_mean_loss": total_loss / len(items),
                     "checkpoint_sha256": digest(phase_dir / "checkpoint/model.safetensors"),
                     "excluded_relevance_rows": len(excluded), "validation_rows_used_for_training": 0,
                     "final_test_used": False, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if config.amp else None}
        write_json(phase_dir / "run_config.json", phase_run)
        print(f"{phase_name} saved; running fixed validation", flush=True)
        report = evaluate(phase_dir / "validation.json", phase_dir / "checkpoint", baseline, device, prompt_version="v2")
        torch.set_rng_state(cpu_rng)
        if config.amp:
            torch.cuda.set_rng_state_all(cuda_rng)
            torch.cuda.empty_cache()
        model.train()
        run["phase_results"].append({"stage": phase_name, "training": phase_run,
                                     "validation_metrics": report["metrics"]})
        write_json(config_file, run)
    if updates != expected_updates:
        raise ValueError("optimizer update attempts differ from the planned budget")
    run.update(status="completed", training_seconds=cumulative_seconds, update_attempts=updates,
               successful_optimizer_updates=successful_updates)
    write_json(config_file, run)
    print(json.dumps({"status": run["status"], "phase_metrics": [r["validation_metrics"]["policy_accuracy"] for r in run["phase_results"]],
                      "training_seconds": cumulative_seconds}, ensure_ascii=False))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("evaluate", "train", "train-halves", "diagnose-train"), default="evaluate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all approved validation rows")
    parser.add_argument("--reverse-options", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prompt-version", choices=("legacy", "v2"), default=None,
                        help="default: v2 for new training, checkpoint's training prompt for evaluation")
    parser.add_argument("--per-score", type=int, default=100, help="distinct training inputs per task and score for fit diagnosis")
    parser.add_argument("--prepared-inputs", type=Path, help="verified native Laya v2 input folder for --action train")
    parser.add_argument("--urgency-loss-share", type=float,
                        help="training-only urgency share of total nominal CE weight; no row resampling")
    args = parser.parse_args()
    if args.prepared_inputs is not None and args.action != "train":
        parser.error("--prepared-inputs is only used with --action train")
    if args.urgency_loss_share is not None and args.action != "train":
        parser.error("--urgency-loss-share is only used with --action train")
    if args.action == "train":
        train(args.output, args.model_path, args.baseline, args.device, args.epochs, args.dry_run,
              args.prompt_version or "v2", args.prepared_inputs, args.urgency_loss_share)
    elif args.action == "train-halves":
        train_halves(args.output, args.model_path, args.baseline, args.device, args.dry_run)
    elif args.action == "diagnose-train":
        diagnose_train(args.output, args.model_path, args.baseline, args.device, args.per_score, args.dry_run)
    else:
        evaluate(args.output, args.model_path, args.baseline, args.device,
                 args.limit, args.reverse_options, args.dry_run, args.prompt_version)


if __name__ == "__main__":
    main()
