"""Evaluate a base or LoRA filtering model on a fixed prepared split."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from filtering_training import TRAINING_ROOT

from sklearn.metrics import f1_score

from filtering_training.datasets.prepare_dataset import DATASET_PATH, OUTPUT_DIR, load_samples
from src.filtering.policy import should_pass
from src.filtering.prompt import SYSTEM_PROMPT, build_messages, parse_model_output
from src.filtering.schema import FilterLabel, FilteringSample


MODEL_NAME = "Qwen/Qwen3-1.7B"
EVALUATION_DIR = TRAINING_ROOT / "outputs" / "evaluation"
MAX_NEW_TOKENS = 192


def load_split_samples(dataset: Path, prepared_dir: Path, split: str) -> list[FilteringSample]:
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("prepared split does not match the current dataset")
    ids = manifest["splits"][split]["notification_ids"]
    samples_by_id = {sample.notification.id: sample for sample in load_samples(dataset)}
    if len(ids) != len(set(ids)):
        raise ValueError("prepared split contains duplicate IDs")
    try:
        return [samples_by_id[notification_id] for notification_id in ids]
    except KeyError as error:
        raise ValueError("prepared split references an unknown sample") from error


def score_predictions(
    gold: list[FilterLabel], predicted: list[FilterLabel | None]
) -> dict:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and prediction lists must have equal nonzero length")
    count = len(gold)
    valid_pairs = [(answer, prediction) for answer, prediction in zip(gold, predicted)
                   if prediction is not None]
    valid_count = len(valid_pairs)
    result = {"count": count, "valid_json": valid_count,
              "json_valid_rate": valid_count / count}
    for field in ("urgency_score", "relevance_score"):
        differences = [abs(getattr(a, field) - getattr(p, field)) for a, p in valid_pairs]
        result[field] = {
            "exact_accuracy": sum(d == 0 for d in differences) / count,
            "within_one_accuracy": sum(d <= 1 for d in differences) / count,
            "mae_on_valid_json": sum(differences) / valid_count if valid_count else None,
        }
    result["category_macro_f1_on_valid_json"] = (
        f1_score([a.category for a, _ in valid_pairs],
                 [p.category for _, p in valid_pairs],
                 average="macro", zero_division=0)
        if valid_count else None
    )
    correct_policy = 0
    urgent_count = 0
    urgent_blocked = 0
    non_pass_count = 0
    unnecessary_passed = 0
    for answer, prediction in zip(gold, predicted):
        gold_pass = should_pass(answer.urgency_score, answer.relevance_score)
        predicted_pass = (should_pass(prediction.urgency_score, prediction.relevance_score)
                          if prediction is not None else None)
        correct_policy += predicted_pass == gold_pass
        if answer.urgency_score >= 4:
            urgent_count += 1
            urgent_blocked += predicted_pass is False
        if not gold_pass:
            non_pass_count += 1
            unnecessary_passed += predicted_pass is True
    result["policy_accuracy"] = correct_policy / count
    result["urgent_false_block_rate"] = (
        urgent_blocked / urgent_count if urgent_count else None
    )
    result["unnecessary_false_pass_rate"] = (
        unnecessary_passed / non_pass_count if non_pass_count else None
    )
    result["urgent_count"] = urgent_count
    result["gold_block_count"] = non_pass_count
    result["metric_note"] = (
        "Invalid JSON counts as wrong for accuracy; MAE and category F1 use valid JSON only. "
        "Invalid JSON is not classified as PASS or BLOCK."
    )
    return result


def evaluate(
    dataset: Path, prepared_dir: Path, split: str, model_name: str,
    adapter: Path | None, output: Path, example_count: int = 0,
    examples_output: Path | None = None, load_in_4bit: bool = False
) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = load_split_samples(dataset, prepared_dir, split)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {"dtype": "auto", "device_map": "auto"}
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this local 4-bit comparison")
        from transformers import BitsAndBytesConfig
        load_kwargs.update(
            dtype=torch.float16,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            ),
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    predictions: list[FilterLabel | None] = []
    responses = []
    timings = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for sample in samples:
        inputs = tokenizer.apply_chat_template(
            build_messages(sample.notification, sample.context),
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        response = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        responses.append(response)
        print(f"Evaluated {len(responses)}/{len(samples)}: {sample.notification.id}", flush=True)
        try:
            predictions.append(parse_model_output(response))
        except ValueError:
            predictions.append(None)
    quantization_config = getattr(model.config, "quantization_config", None)
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    provenance_path = Path(model_name) / "download_provenance.json"
    report = {
        "model": model_name,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "adapter": str(adapter) if adapter is not None else None,
        "quantization": quantization_config,
        "load_in_4bit_requested": load_in_4bit,
        "checkpoint_provenance": json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else None,
        "performance": {
            "generation_seconds": timings,
            "mean_generation_seconds": sum(timings) / len(timings),
            "model_footprint_bytes": model.get_memory_footprint(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        },
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "split": split,
        "generation": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
                       "enable_thinking": False},
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "metrics": score_predictions([sample.label for sample in samples], predictions),
    }
    if example_count:
        examples = [
            {
                "notification_id": sample.notification.id,
                "model_output": prediction.model_dump() if prediction is not None else None,
                "raw_response": response,
            }
            for sample, prediction, response in zip(samples, predictions, responses)
        ][:example_count]
        examples_json = json.dumps(
            {"prediction_examples": examples}, ensure_ascii=False, indent=2
        ) + "\n"
        print(examples_json)
        if examples_output is not None:
            examples_output.parent.mkdir(parents=True, exist_ok=True)
            examples_output.write_text(examples_json, encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--prepared-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples-output", type=Path)
    parser.add_argument("--show-examples", type=int, default=3,
                        help="print this many model JSON outputs (default: 3)")
    args = parser.parse_args()
    if args.show_examples < 0:
        parser.error("--show-examples must be non-negative")
    output = args.output or EVALUATION_DIR / (
        ("adapter" if args.adapter else "base") + f"_{args.split}.json"
    )
    report = evaluate(args.dataset, args.prepared_dir, args.split,
                      args.model, args.adapter, output, args.show_examples,
                      args.examples_output, args.load_in_4bit)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
