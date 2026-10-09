"""용도: 승인된 점수 실험 후보 보존 및 기존 분할을 유지한 작업별 SFT 준비.
생성일: 2026-10-03
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from filtering_training.common.dataset import load_samples
from filtering_training.common.score_tasks import TASKS, prompt_digest, sft_record, unique_urgency_samples
from src.filtering.policy import POLICY_VERSION
from src.filtering.schema import FilteringSample


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def write_lines(path, values):
    path.write_text("".join(json.dumps(v, ensure_ascii=False)+"\n" for v in values), encoding="utf-8")


def reviewed_candidates(review):
    lines = review.read_text(encoding="utf-8").splitlines()
    originals, contexts = {}, {}
    for line in lines:
        if not re.match(r"^\| [1-8] \|", line):
            continue
        cells = [part.strip() for part in line.split("|")[1:-1]]
        if len(cells) == 5:
            originals[int(cells[0])] = cells
        elif len(cells) == 6:
            contexts[int(cells[0])] = cells
    if set(originals) != set(range(1, 9)) or set(contexts) != set(originals):
        raise ValueError("expected eight fully specified review cases")
    samples = []
    for number, row in originals.items():
        app, sender, title = row[1].split(" / ", 2)
        urgency = int(row[4].split(":", 1)[0])
        context_row = contexts[number]
        for role, window, relevance in (
            ("related", context_row[1], int(context_row[2].split(" / ")[0])),
            ("unrelated", context_row[3], int(context_row[4].split(" / ")[0])),
            ("empty", "", int(context_row[5].split(" / ")[0])),
        ):
            if window:
                match = re.fullmatch(r"(.+) \(([^()]+)\)", window)
                if not match:
                    raise ValueError("window must contain an explicit process")
                title_window, process = match.groups()
            else:
                title_window = process = ""
            sample = FilteringSample.model_validate({
                "notification": {"id": f"sep01_{number:02d}_{role}", "app_name": app,
                                 "sender": sender, "title": title, "body": row[2],
                                 "timestamp": "2026-10-03T12:00:00Z"},
                "context": {"active_process": process, "window_title": title_window,
                            "last_updated": "2026-10-03T12:00:00Z",
                            "duration_seconds": 60 if window else 0, "recent_processes": []},
                "label": {"urgency_score": urgency, "relevance_score": relevance,
                          "category": row[3], "ai_summary_reason": row[4].split(":", 1)[1].strip()
                          + (f"; 현재 창 {title_window}에 대한 관련도 제안은 {relevance}입니다."
                             if window else "; 현재 작업 정보가 없어 관련성을 확인할 수 없습니다.")}})
            samples.append(sample)
    unique_urgency_samples(samples)
    return samples


def quarantined_groups(samples, semantic_links):
    """Propagate held-out-family exclusion through shared populated windows."""
    names = {s.notification.id.rsplit("_", 1)[0] for s in samples}
    parent = {name: name for name in names}
    def root(name):
        while parent[name] != name:
            name = parent[name]
        return name
    owners = {}
    for sample in samples:
        group = sample.notification.id.rsplit("_", 1)[0]
        c = sample.context
        if c.active_process:
            key = (c.active_process, c.window_title)
            if key in owners:
                parent[root(group)] = root(owners[key])
            owners[key] = group
    if not set(semantic_links).issubset(names):
        raise ValueError("unknown candidate group in semantic links")
    blocked = {root(name) for name in semantic_links}
    return sorted(name for name in names if root(name) in blocked)


def prepare(source, target, approval_text, approval_date):
    review = target / "review_batch_01.md"
    if not approval_text.strip() or not review.is_file():
        raise ValueError("explicit approval and existing review are required")
    if (target / "approval.json").exists() or (target / "prepared").exists():
        raise ValueError("do not overwrite a recorded approval or prepared experiment")
    dataset = source / "dataset.jsonl"
    source_manifest = source / "prepared/manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != digest(dataset) or manifest["policy_version"] != POLICY_VERSION:
        raise ValueError("source data or policy changed")
    samples = load_samples(dataset)
    by_id = {s.notification.id: s for s in samples}
    selected = {}
    all_ids = []
    for split, info in manifest["splits"].items():
        ids = info["notification_ids"]
        selected[split] = [by_id[i] for i in ids]
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(by_id):
        raise ValueError("source split is not a complete disjoint partition")
    candidates = reviewed_candidates(review)
    normalize = lambda text: re.sub(r"\s+", "", text).casefold()
    original_bodies = {normalize(s.notification.body) for s in samples}
    if any(normalize(s.notification.body) in original_bodies for s in candidates):
        raise ValueError("new body exactly overlaps the source")
    # Conservative authored family links, not an automatic similarity classifier.
    links = {"sep01_01": ["v3_21"], "sep01_02": ["v3_36"]}
    held_messages = {i.rsplit("_", 1)[0] for split in ("validation", "test")
                     for i in manifest["splits"][split]["notification_ids"]}
    if not all(set(v).issubset(held_messages) for v in links.values()):
        raise ValueError("review the semantic exclusion map for this source")
    excluded = quarantined_groups(candidates, links)
    if len(excluded) != 8:
        raise ValueError("this approved batch must remain fully quarantined")
    write_lines(target / "approved_candidates.jsonl", [s.model_dump(mode="json") for s in candidates])
    approval = {"purpose": "preserve approved scores; not added to training",
                "created_date": approval_date, "user_statement": approval_text,
                "review_sha256": digest(review), "candidate_sha256": digest(target/"approved_candidates.jsonl"),
                "reviewed_originals": 8, "reviewed_context_rows": 24,
                "scope": "notification, category, urgency and context relevance; composed reason text is authored",
                "semantic_links_to_heldout": links, "quarantined_message_ids": excluded,
                "semantic_note": "conservative equipment-failure and unrequested-approval families; not exhaustive semantic verification",
                "training_additions": 0, "source_approval_sha256": digest(source/"approval.json")}
    write_json(target / "approval.json", approval)
    prepared_root = target / "prepared"
    prepared_root.mkdir()
    shared = {"purpose": "fixed-split score-separation pilot; no final-test evaluation",
              "created_date": approval_date, "source_dataset": str(dataset.resolve()),
              "source_sha256": digest(dataset), "source_manifest_sha256": digest(source_manifest),
              "policy_version": POLICY_VERSION, "source_reviewed_originals": manifest["human_reviewed_notification_count"],
              "candidate_approval_sha256": digest(target/"approval.json"),
              "quarantined_message_ids": excluded, "training_additions": 0,
              "evaluation_ids": manifest["splits"]["validation"]["notification_ids"]}
    for task in TASKS:
        out = prepared_root / task
        out.mkdir()
        task_manifest = {**shared, "task": task, "prompt_sha256": prompt_digest(task), "splits": {}}
        # Do not create model-ready test prompts in the selection workflow.
        for split in ("train", "validation"):
            values = selected[split]
            if task == "urgency":
                values = unique_urgency_samples(values)
            records = [sft_record(s, task) for s in values]
            write_lines(out/f"{split}.jsonl", records)
            task_manifest["splits"][split] = {
                "count": len(records), "notification_ids": [s.notification.id for s in values],
                "file_sha256": digest(out/f"{split}.jsonl")}
        write_json(out/"manifest.json", task_manifest)
    write_json(target/"experiment_manifest.json", shared)
    return shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("filtering_training/outputs/v3_reviewed_01"))
    parser.add_argument("--target", type=Path, default=Path("filtering_training/outputs/v3_score_separation_01"))
    parser.add_argument("--approval-text", required=True)
    parser.add_argument("--approval-date", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.target, args.approval_text, args.approval_date), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
