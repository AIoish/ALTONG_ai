"""Train a small character n-gram baseline for filtering scores on an existing split."""

import argparse
import json
from pathlib import Path

from filtering_training.common.paths import resolve_existing_path

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from filtering_training.modeling.evaluate import load_split_samples, score_predictions
from src.filtering.schema import FilterLabel


def notification_text(sample):
    n = sample.notification
    return f"앱 {n.app_name} 제목 {n.title} 본문 {n.body}"


def context_text(sample):
    c = sample.context
    recent = " ".join(c.recent_processes)
    duration = c.duration_seconds
    bucket = "없음" if duration == 0 else ("짧음" if duration < 120 else "김")
    return (
        f"{notification_text(sample)} 현재 앱 {c.active_process} "
        f"현재 창 {c.window_title} 최근 앱 {recent} 체류 {bucket}"
    )


def fit_model(texts, labels):
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=50000),
        LinearSVC(class_weight="balanced", random_state=42),
    )
    model.fit(texts, labels)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.dataset = resolve_existing_path(args.dataset)
    args.prepared_dir = resolve_existing_path(args.prepared_dir)

    train = load_split_samples(args.dataset, args.prepared_dir, "train")
    validation = load_split_samples(args.dataset, args.prepared_dir, "validation")
    notification_train = [notification_text(s) for s in train]
    notification_validation = [notification_text(s) for s in validation]
    context_train = [context_text(s) for s in train]
    context_validation = [context_text(s) for s in validation]
    urgency = fit_model(notification_train, [s.label.urgency_score for s in train])
    relevance = fit_model(context_train, [s.label.relevance_score for s in train])
    category = fit_model(notification_train, [s.label.category for s in train])
    predicted = [
        FilterLabel(
            urgency_score=int(u), relevance_score=int(r), category=str(c),
            ai_summary_reason="선형 기준선 예측",
        )
        for u, r, c in zip(
            urgency.predict(notification_validation),
            relevance.predict(context_validation),
            category.predict(notification_validation),
        )
    ]
    metrics = score_predictions([s.label for s in validation], predicted)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as file:
        for sample, prediction in zip(validation, predicted):
            file.write(json.dumps({
                "notification_id": sample.notification.id,
                "notification": sample.notification.model_dump(mode="json"),
                "context": sample.context.model_dump(mode="json"),
                "gold": sample.label.model_dump(mode="json"),
                "prediction": prediction.model_dump(mode="json"),
            }, ensure_ascii=False) + "\n")
    for name, model in (("urgency", urgency), ("relevance", relevance), ("category", category)):
        joblib.dump(model, args.run_dir / f"{name}.joblib", compress=3)
    report = {
        "model": "TF-IDF character 2-4 grams + LinearSVC",
        "dataset": str(args.dataset),
        "train_count": len(train),
        "validation_count": len(validation),
        "metrics": metrics,
        "model_bytes": sum(p.stat().st_size for p in args.run_dir.glob("*.joblib")),
        "note": "Pilot baseline; fixed placeholder reason is not production ready.",
    }
    (args.run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
