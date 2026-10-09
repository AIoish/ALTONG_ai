"""용도: 고정된 0.6B 점수 실험을 기존 학습기로 실행하고 작업 해시를 기록한다.
생성일: 2026-10-03
"""

import argparse
import json
from pathlib import Path

from filtering_training.common.score_tasks import TASKS, prompt_digest
from filtering_training.modeling.train import train
from filtering_training.preparation.prepare_score_experiment import write_json, digest


def run(prepared, run_dir, smoke=False):
    manifest = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    task = manifest["task"]
    if task not in TASKS or manifest["prompt_sha256"] != prompt_digest(task):
        raise ValueError("task prompt changed since preparation")
    for split in ("train", "validation"):
        if digest(prepared/f"{split}.jsonl") != manifest["splits"][split]["file_sha256"]:
            raise ValueError("prepared records changed")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("run directory must be new; preserve previous adapters")
    from transformers import AutoTokenizer
    from filtering_training.modeling.train import load_training_pairs
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    pairs = load_training_pairs(prepared/"train.jsonl", tokenizer, 768)
    tokens = sum(len(tokenizer(p["prompt"]+p["completion"], add_special_tokens=False)["input_ids"]) for p in pairs)
    config = train(prepared, run_dir, 2 if smoke else len(pairs), 768, 42,
                   "Qwen/Qwen3-0.6B", qlora=True, learning_rate=2e-4)
    config.update({"score_task": task, "prompt_sha256": prompt_digest(task),
                   "created_date": "2026-10-03", "prepared_dir": str(prepared.resolve()),
                   "source_manifest_sha256": manifest["source_manifest_sha256"],
                   "score_experiment": "smoke" if smoke else "one_epoch_fixed_original_data",
                   "training_rows_tokens_one_pass": tokens,
                   "approval_sha256": manifest["candidate_approval_sha256"]})
    write_json(run_dir/"run_config.json", config)
    return {"task": task, "steps": config["max_steps"], "loss": config["training_loss"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.prepared_dir, args.run_dir, args.smoke)))


if __name__ == "__main__":
    main()
