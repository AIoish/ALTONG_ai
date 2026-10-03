"""Freeze reviewed v3 candidates and prepare splits with context/family isolation."""

from filtering_training.common.paths import OUTPUTS_ROOT

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from filtering_training.generation.expand_v3_dataset import decode_review
from filtering_training.common.dataset import load_samples, to_sft_record
from src.filtering.policy import POLICY_VERSION
from src.filtering.prompt import CATEGORIES, SYSTEM_PROMPT
from src.filtering.schema import FilteringSample


ROOT = OUTPUTS_ROOT
SEMANTIC_FAMILIES = {
    "download_complete": ("v3_15", "v3_61"),
    "clipboard_copy": ("v3_16", "v3_59"),
    "meeting_access": ("v3_05", "v3_29", "v3_31"),
    "document_typo": ("v3_04", "v3_24"),
    "footwear_promotion": ("v3_13", "v3_14", "v3_57"),
    "order_processing_failure": ("v3_01", "v3_02"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supplementary_samples():
    """An independent calendar/advertisement pair, not individually reviewed."""
    cases = [
        ("v3_65", "예약 앱", "수업 예약", "요가 수업 예약 확인",
         "다음 주 토요일 10시 요가 수업이 예약되었습니다. 참석 여부는 전날까지 변경할 수 있습니다.",
         "일정/회의", 2, ("OUTLOOK.EXE", "다음 주 요가 수업 일정 - Outlook"), 5,
         "다음 주에 참석할 수업 예약으로 즉시 대응이 필요하지 않습니다",
         "현재 정리 중인 다음 주 요가 수업 일정과 직접 연결됩니다"),
        ("v3_66", "Gmail", "조명 쇼핑몰", "독서등 할인 안내",
         "책상용 독서등 15% 할인! 이번 주 상품을 비교하고 원하는 밝기를 골라 보세요.",
         "광고/홍보", 1, ("chrome.exe", "독서등 상품 비교 - Chrome"), 4,
         "선택적인 상품 할인 안내로 긴급한 대응이 필요하지 않습니다",
         "현재 독서등을 비교하는 구매 목적에 도움이 됩니다"),
    ]
    samples = []
    for index, (message, app, sender, title, body, category, urgency, context, score, urgent_reason, relation) in enumerate(cases):
        for role, (process, window), relevance in (
            ("related", context, score), ("unrelated", cases[1 - index][7], 1), ("empty", ("", ""), 1)
        ):
            reason = relation if role == "related" else (
                "현재 창의 작업과 다른 주제입니다" if role == "unrelated"
                else "현재 작업 정보가 없어 관련성을 확인할 수 없습니다")
            samples.append(FilteringSample.model_validate({
                "notification": {"id": f"{message}_{role}", "app_name": app, "sender": sender,
                                 "title": title, "body": body, "timestamp": "2026-10-03T07:00:00Z"},
                "context": {"active_process": process, "window_title": window,
                            "last_updated": "2026-10-03T06:59:57Z", "duration_seconds": 90 if process else 0,
                            "recent_processes": [process] if process else []},
                "label": {"urgency_score": urgency, "relevance_score": relevance, "category": category,
                          "ai_summary_reason": f"{reason}; {urgent_reason}."},
            }))
    return samples


def isolated_splits(samples, seed=42):
    by_message = defaultdict(list)
    for sample in samples:
        by_message[sample.notification.id.rsplit("_", 1)[0]].append(sample)
    parent = {message: message for message in by_message}

    def find(message):
        if parent[message] != message:
            parent[message] = find(parent[message])
        return parent[message]

    def join(left, right):
        parent[find(right)] = find(left)

    context_owner = {}
    for message, variants in by_message.items():
        if len(variants) != 3 or len({(s.label.urgency_score, s.label.category) for s in variants}) != 1:
            raise ValueError("notification context variants changed urgency or category")
        for sample in variants:
            context = sample.context
            if context.active_process:
                key = (context.active_process, context.window_title)
                if key in context_owner:
                    join(message, context_owner[key])
                context_owner[key] = message
    for messages in SEMANTIC_FAMILIES.values():
        if not set(messages).issubset(by_message):
            raise ValueError("semantic family references a missing notification")
        for message in messages[1:]:
            join(messages[0], message)
    components = defaultdict(list)
    for message in sorted(by_message):
        components[find(message)].append(message)
    groups = sorted(components.values())
    random.Random(seed).shuffle(groups)
    category_by_message = {m: variants[0].label.category for m, variants in by_message.items()}

    def cover(available):
        # Exactly one independent notification/category in each evaluation split.
        candidates = []
        for index in available:
            categories = Counter(category_by_message[m] for m in groups[index])
            if max(categories.values()) == 1 and len(groups[index]) <= 8:
                candidates.append((index, set(categories)))

        def search(start, chosen, covered):
            if covered == set(CATEGORIES):
                return chosen
            for position in range(start, len(candidates)):
                index, categories = candidates[position]
                if not covered.intersection(categories):
                    found = search(position + 1, chosen + [index], covered | categories)
                    if found is not None:
                        return found
            return None

        result = search(0, [], set())
        if result is None:
            raise ValueError("cannot reserve category-balanced isolated splits")
        return set(result)

    validation = cover(range(len(groups)))
    test = cover([index for index in range(len(groups)) if index not in validation])
    assignments = {}
    component_by_message = {}
    for index, messages in enumerate(groups):
        name = "validation" if index in validation else "test" if index in test else "train"
        component = "component_" + hashlib.sha256("|".join(messages).encode()).hexdigest()[:12]
        for message in messages:
            assignments[message] = name
            component_by_message[message] = component
    splits = {name: [] for name in ("train", "validation", "test")}
    contexts = {name: set() for name in splits}
    for sample in samples:
        name = assignments[sample.notification.id.rsplit("_", 1)[0]]
        splits[name].append(sample)
        if sample.context.active_process:
            contexts[name].add((sample.context.active_process, sample.context.window_title))
    for left, right in itertools.combinations(splits, 2):
        if contexts[left] & contexts[right]:
            raise ValueError("populated window crosses data splits")
    for name, items in splits.items():
        if set(s.label.category for s in items) != set(CATEGORIES):
            raise ValueError(f"categories missing from {name}")
    return splits, component_by_message, sorted(len(group) for group in groups)


def prepare(source_dir, output_dir, approval_text, approval_date, seed=42):
    source = source_dir / "candidates.jsonl"
    original = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if original["dataset_sha256"] != digest(source) or original["policy_version"] != POLICY_VERSION:
        raise ValueError("source or policy changed since review")
    review = decode_review(source_dir / "review.csv")
    if len(review) != 8 or any(row.get("feedback", "").strip() for row in review):
        raise ValueError("review contains edits; resolve before preparing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output already exists; do not overwrite previous approval or split")
    original_samples = load_samples(source)
    if len(original_samples) != 192:
        raise ValueError("expected the 64-message first expansion")
    samples = original_samples + supplementary_samples()
    splits, components, sizes = isolated_splits(samples, seed)
    reviewed_messages = {f"v3_{index:02d}" for index in range(1, 17)} | {row["message_id"] for row in review}
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "dataset.jsonl"
    with dataset.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            file.write(sample.model_dump_json() + "\n")
    approval = {"approval_date": approval_date, "user_statement": approval_text,
                "dataset_sha256": digest(dataset), "review_csv_sha256": digest(source_dir / "review.csv"),
                "seed_approval": json.loads((source_dir / "seed_approval.json").read_text(encoding="utf-8")),
                "human_reviewed_notification_count": len(reviewed_messages),
                "human_reviewed_message_ids": sorted(reviewed_messages),
                "accepted_expansion_sha256": digest(source),
                "accepted_expansion_notification_count": 64,
                "supplementary_authored_notifications": ["v3_65", "v3_66"],
                "remaining_notification_count": 42,
                "remaining_status": "authored and structurally checked; expansion direction accepted, not individually human reviewed",
                "policy_version": POLICY_VERSION}
    (output_dir / "approval.json").write_text(json.dumps(approval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prepared = output_dir / "prepared"
    prepared.mkdir()
    manifest = {"source_sha256": digest(dataset), "policy_version": POLICY_VERSION,
                "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "seed": seed, "status": "small synthetic pilot; not a final quality benchmark",
                "approval_sha256": digest(output_dir / "approval.json"),
                "human_reviewed_notification_count": len(reviewed_messages),
                "split_rule": "connected notifications sharing any populated window or authored semantic family stay together",
                "component_sizes_in_notifications": sizes,
                "populated_context_overlap": 0,
                "empty_context_note": "absence of window information is shared legitimately; focus ON in all splits",
                "splits": {}}
    for name, items in splits.items():
        with (prepared / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as file:
            for sample in items:
                file.write(json.dumps(to_sft_record(sample), ensure_ascii=False) + "\n")
        messages = {s.notification.id.rsplit("_", 1)[0] for s in items}
        manifest["splits"][name] = {"count": len(items), "notification_count": len(messages),
                                    "notification_ids": [s.notification.id for s in items],
                                    "message_ids": sorted(messages),
                                    "families": sorted({components[m] for m in messages}),
                                    "category_counts": dict(Counter(s.label.category for s in items))}
    (prepared / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# 검수 반영 필터링 데이터 v3\n\n"
        "원문 66개, 문맥별 198개입니다. 첫 16개와 경계 사례 8개를 사람이 검수했고, 나머지 42개는 작성·자동 점검한 합성 후보입니다.\n\n"
        "일정·광고에서 독립된 평가 그룹을 확보하기 위해 새 원문 2개를 보충했습니다.\n"
        "학습 50개 원문(150건), 검증 8개 원문(24건), 테스트 8개 원문(24건)으로 나눴습니다.\n"
        "같은 알림의 세 문맥, 동일 창, 지정한 유사 시나리오는 같은 분할에만 속합니다.\n\n"
        "소규모 파일럿용입니다. 테스트는 모델 선택과 학습 조정에 사용하지 않습니다. 일반 PC에서의 성능이나 실제 알림 품질을 보장하는 평가셋은 아닙니다.\n",
        encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "v3_expansion_01")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "v3_reviewed_01")
    parser.add_argument("--approval-text", required=True)
    parser.add_argument("--approval-date", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.source_dir, args.output_dir, args.approval_text, args.approval_date, args.seed)
    print(json.dumps({name: {k: info[k] for k in ("count", "notification_count")}
                      for name, info in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
