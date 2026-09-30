"""Compare complete predictions on the same frozen evaluation dataset and prompt."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from filtering_training.evaluate import score_predictions
from filtering_training.datasets.prepare_dataset import load_samples
from src.filtering.policy import should_pass
from src.filtering.prompt import CATEGORIES, parse_model_output


def compare(dataset: Path, runs: list[tuple[Path, Path]]) -> dict:
    samples = load_samples(dataset)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    expected_ids = [sample.notification.id for sample in samples]
    prompt_hash = None
    summaries = []
    for report_path, predictions_path in runs:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["dataset_sha256"] != digest:
            raise ValueError("evaluation report does not match the frozen dataset")
        if prompt_hash is None:
            prompt_hash = report["prompt_sha256"]
        elif prompt_hash != report["prompt_sha256"]:
            raise ValueError("evaluation reports use different prompts")
        records = json.loads(predictions_path.read_text(encoding="utf-8"))["prediction_examples"]
        if [record["notification_id"] for record in records] != expected_ids:
            raise ValueError("predictions must cover every dataset ID exactly once in dataset order")
        predictions = [parse_model_output(json.dumps(record["model_output"], ensure_ascii=False))
                       if record["model_output"] is not None else None for record in records]
        metrics = score_predictions([sample.label for sample in samples], predictions)
        if metrics != report["metrics"]:
            raise ValueError("stored metrics disagree with complete predictions")
        errors = []
        per_category = {}
        for category in CATEGORIES:
            indices = [i for i, sample in enumerate(samples) if sample.label.category == category]
            if indices:
                per_category[category] = score_predictions(
                    [samples[i].label for i in indices], [predictions[i] for i in indices])
        category_correct = 0
        invalid_urgent = 0
        for sample, prediction in zip(samples, predictions):
            gold = sample.label
            expected_pass = should_pass(gold.urgency_score, gold.relevance_score)
            actual_pass = should_pass(prediction.urgency_score, prediction.relevance_score) if prediction else None
            invalid_urgent += prediction is None and gold.urgency_score >= 4
            category_correct += prediction is not None and prediction.category == gold.category
            if prediction is None or prediction.model_dump(exclude={"ai_summary_reason"}) != gold.model_dump(exclude={"ai_summary_reason"}):
                errors.append({"notification_id": sample.notification.id,
                               "gold": gold.model_dump(),
                               "prediction": prediction.model_dump() if prediction else None,
                               "policy_error": actual_pass != expected_pass,
                               "expected_pass": expected_pass, "actual_pass": actual_pass})
        summaries.append({"model": report["model"], "adapter": report["adapter"],
                          "report": str(report_path), "metrics": metrics,
                          "category_correct": category_correct,
                          "urgent_invalid_json_count": invalid_urgent,
                          "performance": report.get("performance"),
                          "per_category": per_category, "errors": errors})
    return {"dataset_sha256": digest, "prompt_sha256": prompt_hash,
            "count": len(samples), "category_counts": dict(Counter(s.label.category for s in samples)),
            "note": "Synthetic author-labeled evaluation, not independent human ground truth. "
                    "Compare candidate pipelines; training data and settings can differ.",
            "runs": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run", action="append", nargs=2, type=Path, required=True,
                        metavar=("REPORT", "PREDICTIONS"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.dataset, args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for run in result["runs"]:
        print(json.dumps({"model": run["model"], "policy_accuracy": run["metrics"]["policy_accuracy"],
                          "category_correct": run["category_correct"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
