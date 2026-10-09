"""용도: 전체·점수·분리 어댑터의 검증 점수와 알림 전체 지연을 비교한다.
생성일: 2026-10-03
"""

import argparse
import json
import math
import time
from pathlib import Path

from filtering_training.common.score_tasks import messages, parse_scores, prompt_digest, score_metrics
from filtering_training.modeling.evaluate import load_split_samples
from filtering_training.preparation.prepare_score_experiment import digest, write_json, write_lines
from src.filtering.policy import POLICY_VERSION
from src.filtering.prompt import parse_model_output


def evaluate(source, prepared_root, runs, variant, output, max_samples=0, created_date="2026-10-03"):
    tasks = {"A": ("full",), "B": ("scores",), "C": ("urgency", "relevance")}[variant]
    if output.exists():
        raise ValueError("report already exists")
    samples = load_split_samples(source/"dataset.jsonl", source/"prepared", "validation")
    if max_samples:
        samples = samples[:max_samples]
    configs = {}
    for task in tasks:
        prepared = prepared_root/task
        config = json.loads((runs/task/"run_config.json").read_text(encoding="utf-8"))
        if (config.get("score_task") != task or config["model"] != "Qwen/Qwen3-0.6B"
                or config["dataset_sha256"] != digest(source/"dataset.jsonl")
                or config.get("prompt_sha256") != prompt_digest(task)
                or config["prepared_manifest_sha256"] != digest(prepared/"manifest.json")
                or config["source_manifest_sha256"] != digest(source/"prepared/manifest.json")):
            raise ValueError("adapter does not match task, source split, or prompt")
        configs[task] = config
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("this pilot requires CUDA")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", local_files_only=True, dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(model, runs/tasks[0]/"adapter", adapter_name=tasks[0])
    for task in tasks[1:]:
        model.load_adapter(runs/task/"adapter", adapter_name=task)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started

    def predict(sample):
        values, responses, task_timings, invalid = {}, {}, {}, False
        torch.cuda.synchronize()
        start = time.perf_counter()
        for task in tasks:
            task_start = time.perf_counter()
            model.set_adapter(task)
            inputs = tokenizer.apply_chat_template(messages(sample, task), add_generation_prompt=True,
                        enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
            if inputs["input_ids"].shape[-1] > 768:
                raise ValueError("input exceeds training sequence limit")
            limit = 192 if task == "full" else 64
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            tokens = generated[0][inputs["input_ids"].shape[-1]:]
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            ended = len(tokens) > 0 and int(tokens[-1]) == tokenizer.eos_token_id
            try:
                if len(tokens) >= limit and not ended:
                    raise ValueError("generation reached token limit without EOS")
                values.update(parse_scores(raw, task))
                error = None
            except ValueError as exc:
                invalid = True
                error = str(exc)
            responses[task] = {"raw": raw, "output_tokens": len(tokens), "error": error}
            torch.cuda.synchronize()
            task_timings[task] = time.perf_counter()-task_start
        torch.cuda.synchronize()
        return (None if invalid else values), responses, task_timings, time.perf_counter()-start

    # Warm-up is separately reported and uses one validation case, never a test prompt.
    _, _, _, warmup_seconds = predict(samples[0])
    torch.cuda.reset_peak_memory_stats()
    predictions, rows, times = [], [], []
    for sample in samples:
        prediction, raw, task_times, elapsed = predict(sample)
        predictions.append(prediction)
        times.append(elapsed)
        rows.append({"notification_id": sample.notification.id,
                     "notification": sample.notification.model_dump(mode="json"),
                     "context": sample.context.model_dump(mode="json"), "gold": sample.label.model_dump(),
                     "prediction": prediction, "responses": raw, "task_seconds": task_times,
                     "end_to_end_seconds": elapsed})
        print(f"{variant} {len(rows)}/{len(samples)} {elapsed:.2f}s", flush=True)
    quantization = model.config.quantization_config
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    report = {"purpose": "small synthetic validation comparison; not production integration",
              "created_date": created_date, "variant": variant, "policy_version": POLICY_VERSION,
              "model": "Qwen/Qwen3-0.6B", "model_revision": getattr(model.config, "_commit_hash", None),
              "quantization": quantization, "split": "validation", "sampling": max_samples or None,
              "dataset_sha256": digest(source/"dataset.jsonl"),
              "source_manifest_sha256": digest(source/"prepared/manifest.json"),
              "evaluated_ids": [s.notification.id for s in samples],
              "prompt_hashes": {task: prompt_digest(task) for task in tasks},
              "run_config_hashes": {task: digest(runs/task/"run_config.json") for task in tasks},
              "adapter_bytes": {task: sum(p.stat().st_size for p in (runs/task/"adapter").rglob("*") if p.is_file()) for task in tasks},
              "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "peft": peft.__version__, "device": torch.cuda.get_device_name(0)},
              "generation": {"do_sample": False, "enable_thinking": False,
                             "max_new_tokens": {task: 192 if task == "full" else 64 for task in tasks}},
              "performance": {"load_seconds": load_seconds, "warmup_seconds": warmup_seconds,
                              "mean_seconds": sum(times)/len(times),
                              "p95_seconds": sorted(times)[math.ceil(.95*len(times))-1],
                              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()},
              "metrics": score_metrics(samples, predictions)}
    if variant == "A":
        labels = [parse_model_output(row["responses"]["full"]["raw"]) if row["prediction"] else None for row in rows]
        report["category_exact_correct"] = sum(label is not None and label.category == s.label.category
                                                for s, label in zip(samples, labels))
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.with_suffix(".predictions.jsonl")
    if predictions_path.exists():
        raise ValueError("prediction records already exist")
    write_lines(predictions_path, rows)
    report["predictions_sha256"] = digest(predictions_path)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("filtering_training/outputs/v3_reviewed_01"))
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--created-date", default="2026-10-04")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("max-samples must be nonnegative")
    report = evaluate(args.source, args.prepared_root, args.runs, args.variant, args.output, args.max_samples, args.created_date)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
