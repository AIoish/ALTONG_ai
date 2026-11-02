"""용도: 기존 Qwen 점수 어댑터의 GGUF CPU 정확도·지연·대기 자원을 측정한다.
생성일: 2026-10-10
"""

import argparse
import json
import os
import platform
import random
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from filtering_training.common.digit_scores import replace_prompt
from filtering_training.common.score_tasks import messages, score_metrics
from filtering_training.modeling.digit_score_experiment import (
    VALIDATION_POOL, continuation_snapshot, encode, load_validation_for_evaluation,
)
from filtering_training.preparation.prepare_score_experiment import digest, write_json, write_lines

ASSETS = Path("filtering_training/outputs/models/qwen3-06b-cpu")
RUNS = Path("filtering_training/outputs/runs/2026-10-05-qwen06-accumulated-01")


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def completion_payload(token_ids, task):
    if task not in ("urgency", "relevance"):
        raise ValueError("unknown score task")
    return {"prompt": token_ids, "n_predict": 1, "temperature": 0,
            "samplers": ["temperature"], "grammar": "root ::= [1-5]",
            "repeat_penalty": 1.0, "cache_prompt": False, "return_tokens": True,
            "lora": [{"id": 0 if task == "urgency" else 1, "scale": 1.0}]}


def parse_choice(response, digit_ids):
    value = response.get("content", "")
    if value not in {"1", "2", "3", "4", "5"}:
        raise ValueError("CPU runtime returned an invalid digit")
    if response.get("tokens") != [digit_ids[int(value) - 1]]:
        raise ValueError("CPU runtime changed the single-digit token mapping")
    if response.get("truncated"):
        raise ValueError("CPU runtime truncated the prompt")
    return int(value)


def preflight(assets):
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "ready" or manifest.get("gpu_layers") != 0:
        raise ValueError("CPU assets are not ready")
    hashes = dict(manifest["protected_source_hashes"])
    hashes[str(assets / "manifest.json")] = digest(assets / "manifest.json")
    for name, info in manifest["files"].items():
        hashes[str(assets / name)] = info["sha256"]
    hashes[str(assets / "runtime/llama-server.exe")] = manifest["runtime_executable_sha256"]
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError("CPU deployment source or artifact changed: " + path)
    parents = {task: continuation_snapshot(RUNS / task, task) for task in ("urgency", "relevance")}
    if any(parent["model_revision"] != manifest["base_revision"] for parent in parents.values()):
        raise ValueError("CPU base revision differs from the trained adapters")
    samples, groups, _ = load_validation_for_evaluation(VALIDATION_POOL)
    baseline = json.loads((RUNS / "after.json").read_text(encoding="utf-8"))
    rows = [json.loads(s) for s in (RUNS / "after.predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(samples):
        raise ValueError("baseline validation size differs")
    for sample, row in zip(samples, rows):
        if any(row[k] != value for k, value in (
            ("notification_id", sample.notification.id),
            ("notification", sample.notification.model_dump(mode="json")),
            ("context", sample.context.model_dump(mode="json")),
            ("gold", sample.label.model_dump(mode="json")),
        )):
            raise ValueError("CPU baseline input or gold differs")
    if score_metrics(samples, [r["prediction"] for r in rows]) != baseline["metrics"]:
        raise ValueError("baseline predictions do not reproduce the reported accuracy")
    for name in ("dataset.jsonl", "approval.json", "manifest.json"):
        path = VALIDATION_POOL / name
        hashes[str(path)] = digest(path)
    return manifest, hashes, samples, groups, baseline, rows


def resource_snapshot(process):
    memory = process.memory_info()
    cpu = process.cpu_times()
    return {"cpu_seconds": cpu.user + cpu.system, "rss_bytes": memory.rss,
            "private_bytes": getattr(memory, "private", None)}


class ResourceMonitor:
    def __init__(self, process, logical_cpus):
        self.process, self.logical_cpus = process, logical_cpus
        self.rows, self.phase = [], "idle"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        import psutil
        self.process.cpu_percent()
        psutil.cpu_percent()
        while not self.stop.wait(.2):
            self.rows.append(dict(resource_snapshot(self.process), monotonic=time.perf_counter(),
                                  phase=self.phase,
                                  runtime_cpu_percent_machine=self.process.cpu_percent() / self.logical_cpus,
                                  system_cpu_percent=psutil.cpu_percent()))

    def close(self):
        self.stop.set()
        self.thread.join()


def benchmark(output, assets=ASSETS, threads=4, count=100, idle_seconds=30, dry_run=False):
    if threads < 1 or count < 74 or idle_seconds < 5:
        raise ValueError("use positive threads, at least 74 notifications and at least 5 idle seconds")
    if output.parent.exists():
        raise ValueError("choose a fresh benchmark run folder")
    manifest, hashes, samples, groups, baseline, baseline_rows = preflight(assets)
    from transformers import AutoTokenizer
    import psutil
    base = next(Path(p).parent for p in hashes if p.endswith("model.safetensors") and "snapshots" in p)
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    digit_ids = [tokenizer.encode(str(n), add_special_tokens=False) for n in range(1, 6)]
    if any(len(ids) != 1 for ids in digit_ids):
        raise ValueError("score digits require single tokens")
    digit_ids = [ids[0] for ids in digit_ids]
    if any(continuation_snapshot(RUNS / t, t)["digit_token_ids"] != digit_ids for t in ("urgency", "relevance")):
        raise ValueError("tokenizer differs from trained adapters")
    prepared = []
    for sample in samples:
        prepared.append({task: encode(tokenizer, replace_prompt(messages(sample, task), task))["input_ids"]
                         for task in ("urgency", "relevance")})
    order = list(range(len(samples)))
    random.Random(42).shuffle(order)
    order = [order[i % len(order)] for i in range(count)]
    plan = {"purpose": "CPU-only one-base two-adapter deployment accuracy and resource benchmark",
            "created_date": datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
            "status": "prepared", "threads": threads, "threads_batch": threads,
            "gpu_layers": 0, "llama_release": manifest["llama_release"], "quantization": "Q8_0 base; F16 LoRA",
            "notification_count": count, "forward_requests": 2 * count, "unique_validation_rows": len(samples),
            "selection": "seed 42 shuffled fixed validation repeated; first 74 only for accuracy",
            "selected_validation_indices": order, "idle_seconds": idle_seconds,
            "cache_prompt": False, "server_poll": 0, "context_size": 1024,
            "protected_hashes": hashes, "baseline_metrics": baseline["metrics"],
            "final_test_used": False, "training_performed": False,
            "prompt_tokens": {t: {"min": min(len(p[t]) for p in prepared),
                                     "max": max(len(p[t]) for p in prepared)} for t in ("urgency", "relevance")}}
    if dry_run:
        print(json.dumps({k: v for k, v in plan.items() if k not in ("protected_hashes", "baseline_metrics", "selected_validation_indices")}, indent=2))
        return plan
    output.parent.mkdir(parents=True)
    plan["status"] = "running"
    write_json(output, plan)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    def api(route, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url + route, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    cmd = [str((assets / "runtime/llama-server.exe").resolve()), "-m", str((assets / "base-Q8_0.gguf").resolve()),
           "--lora", str((assets / "urgency-F16.gguf").resolve()),
           "--lora", str((assets / "relevance-F16.gguf").resolve()), "--lora-init-without-apply",
           "-ngl", "0", "-t", str(threads), "-tb", str(threads), "-c", "1024", "-np", "1",
           "--poll", "0", "--poll-batch", "0", "--host", "127.0.0.1", "--port", str(port)]
    monitor = None
    process = None
    log = (output.parent / "server.log").open("w", encoding="utf-8")
    started = time.perf_counter()
    try:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        deadline = started + 90
        while True:
            if process.poll() is not None:
                raise RuntimeError("CPU server exited; inspect server.log")
            try:
                if api("/health").get("status") == "ok": break
            except (urllib.error.URLError, TimeoutError): pass
            if time.perf_counter() > deadline: raise TimeoutError("CPU server startup timed out")
            time.sleep(.1)
        plan["load_seconds"] = time.perf_counter() - started
        adapters = api("/lora-adapters")
        if len(adapters) != 2 or [a["id"] for a in adapters] != [0, 1]:
            raise ValueError("CPU runtime did not register both adapters in order")
        for task in ("urgency", "relevance"):
            text = tokenizer.apply_chat_template(replace_prompt(messages(samples[order[0]], task), task),
                                                 add_generation_prompt=True, enable_thinking=False, tokenize=False)
            if api("/tokenize", {"content": text, "add_special": False})["tokens"] != prepared[order[0]][task]:
                raise ValueError("GGUF and trained tokenizer token ids differ")
            parse_choice(api("/completion", completion_payload(prepared[order[0]][task], task)), digit_ids)
        server = psutil.Process(process.pid)
        logical = psutil.cpu_count(logical=True)
        plan.update(logical_cpus=logical, physical_cpus=psutil.cpu_count(logical=False),
                    processor=platform.processor(), total_ram_bytes=psutil.virtual_memory().total)
        monitor = ResourceMonitor(server, logical)
        monitor.thread.start()
        print("loaded idle measurement", flush=True)
        idle_start = time.perf_counter()
        idle_before = resource_snapshot(server)
        time.sleep(idle_seconds)
        idle_duration = time.perf_counter() - idle_start
        idle_after = resource_snapshot(server)
        monitor.phase = "active"
        predictions, rows = {}, []
        active_start = time.perf_counter()
        active_before = resource_snapshot(server)
        for number, index in enumerate(order):
            notification_start = time.perf_counter()
            before = resource_snapshot(server)
            prediction, timings = {}, {}
            for task in ("urgency", "relevance"):
                response = api("/completion", completion_payload(prepared[index][task], task))
                prediction[task + "_score"] = parse_choice(response, digit_ids)
                timings[task] = response.get("timings", {})
            elapsed = time.perf_counter() - notification_start
            after = resource_snapshot(server)
            if index in predictions and predictions[index] != prediction:
                raise ValueError("repeated input produced different CPU score outputs")
            predictions[index] = prediction
            sample = samples[index]
            rows.append({"notification_id": sample.notification.id, "validation_index": index,
                         "notification": sample.notification.model_dump(mode="json"),
                         "context": sample.context.model_dump(mode="json"), "gold": sample.label.model_dump(mode="json"),
                         "prediction": prediction, "end_to_end_seconds": elapsed,
                         "runtime_cpu_percent_machine": (after["cpu_seconds"] - before["cpu_seconds"]) / elapsed / logical * 100,
                         "runtime_rss_bytes": after["rss_bytes"], "timings": timings})
            if (number + 1) % 10 == 0: print(f"CPU notifications {number+1}/{count}", flush=True)
        active_duration = time.perf_counter() - active_start
        active_after = resource_snapshot(server)
        monitor.close()
        samples_resource = monitor.rows
        monitor = None
        latencies = [r["end_to_end_seconds"] for r in rows]
        def phase_stats(phase):
            selected = [r for r in samples_resource if r["phase"] == phase]
            return {"samples": len(selected),
                    "runtime_cpu_percent_sample_peak": max(r["runtime_cpu_percent_machine"] for r in selected),
                    "system_cpu_percent_mean": statistics.mean(r["system_cpu_percent"] for r in selected),
                    "runtime_rss_bytes_mean": statistics.mean(r["rss_bytes"] for r in selected),
                    "runtime_rss_bytes_peak": max(r["rss_bytes"] for r in selected)}
        plan.update(status="completed", metrics=score_metrics(samples, [predictions[i] for i in range(len(samples))]),
                    latency_seconds={"mean": statistics.mean(latencies), "p50": percentile(latencies, .5),
                                     "p95": percentile(latencies, .95), "p99": percentile(latencies, .99), "max": max(latencies)},
                    idle=dict(phase_stats("idle"), measured_seconds=idle_duration,
                              runtime_cpu_percent_mean=(idle_after["cpu_seconds"]-idle_before["cpu_seconds"])/idle_duration/logical*100),
                    active=dict(phase_stats("active"), measured_seconds=active_duration,
                                runtime_cpu_percent_mean=(active_after["cpu_seconds"]-active_before["cpu_seconds"])/active_duration/logical*100),
                    python_harness_rss_bytes=psutil.Process().memory_info().rss,
                    memory_scope="runtime RSS excludes Python measurement harness; RSS is working set, not private commit",
                    cpu_scope="runtime CPU normalized over all logical CPUs, comparable to task manager total; system includes other apps",
                    metrics_by_set={name:score_metrics([s for s in samples if s.notification.id in ids],
                                                      [predictions[i] for i,s in enumerate(samples) if s.notification.id in ids])
                                    for name, ids in groups.items()},
                    cpu_matches_gpu_prediction_rows=sum(predictions[i] == r["prediction"] for i,r in enumerate(baseline_rows)))
        for path, expected in hashes.items():
            if digest(Path(path)) != expected: raise ValueError("protected source changed during benchmark: " + path)
        write_lines(output.with_suffix(".predictions.jsonl"), rows)
        write_lines(output.parent / "resources.jsonl", samples_resource)
        plan["predictions_sha256"] = digest(output.with_suffix(".predictions.jsonl"))
        plan["resources_sha256"] = digest(output.parent / "resources.jsonl")
        write_json(output, plan)
        print(json.dumps({k:plan[k] for k in ("status", "latency_seconds", "idle", "active", "cpu_matches_gpu_prediction_rows")}, indent=2), flush=True)
        return plan
    except Exception as exc:
        plan.update(status="failed", failure=str(exc))
        write_json(output, plan)
        raise
    finally:
        if monitor is not None: monitor.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--idle-seconds", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    benchmark(args.output, args.assets, args.threads, args.count, args.idle_seconds, args.dry_run)


if __name__ == "__main__":
    main()
