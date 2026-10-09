"""용도: 숫자 분류 학습 진단과 필터링 데이터 전체 검수 대상 집계.
생성일: 2026-10-04
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from filtering_training.common.digit_scores import prompt_digest, replace_prompt
from filtering_training.common.score_tasks import parse_scores
from filtering_training.common.score_tasks import load_score_samples
from filtering_training.modeling.digit_score_experiment import load_model, encode
from filtering_training.preparation.prepare_digit_supplement import OUTPUT
from filtering_training.preparation.prepare_score_experiment import digest, write_json, write_lines


def audit_pool_separation(pool, validation, training_file="candidates.jsonl"):
    """Reserve the entire validation pool even when only a preview is approved."""
    pool_manifest = json.loads((pool/"manifest.json").read_text(encoding="utf-8"))
    val_manifest = json.loads((validation/"manifest.json").read_text(encoding="utf-8"))
    if val_manifest.get("intended_split") != "validation":
        raise ValueError("validation pool must be reserved for validation")
    for folder,manifest in ((pool,pool_manifest),(validation,val_manifest)):
        if digest(folder/"candidates.jsonl") != manifest["candidate_sha256"]:
            raise ValueError("candidate pool changed")
    if training_file != "candidates.jsonl":
        refinement = pool_manifest.get("label_revision", {}) if training_file == pool_manifest.get("label_revision", {}).get("dataset_file") else pool_manifest.get("wording_refinement", {})
        if training_file != refinement.get("dataset_file") or digest(pool/training_file) != refinement.get("dataset_sha256"):
            raise ValueError("refined training snapshot changed")
    train = load_score_samples(pool/training_file)
    heldout = load_score_samples(validation/"candidates.jsonl")
    normalize = lambda text:re.sub(r"\s+","",text).casefold()
    def windows(rows):
        result=set()
        for s in rows:
            if s.context.active_process and s.context.window_title.strip():
                result.add((s.context.active_process.casefold(),normalize(s.context.window_title)))
            for window in getattr(s.context,"recent_windows",[]):
                if window.window_title.strip():
                    result.add((window.app_name.casefold(),normalize(window.window_title)))
        return result
    shared_ids = {s.notification.id for s in train} & {s.notification.id for s in heldout}
    shared_bodies = {normalize(s.notification.body) for s in train} & {normalize(s.notification.body) for s in heldout}
    shared_windows = windows(train) & windows(heldout)
    reserved_families=set(val_manifest.get("reserved_families",{}))
    train_families=set(pool_manifest.get("families",{}))
    if (pool/"lineage.jsonl").exists():
        if digest(pool/"lineage.jsonl") != pool_manifest["lineage_sha256"]:
            raise ValueError("training lineage changed")
        for line in (pool/"lineage.jsonl").read_text(encoding="utf-8").splitlines():
            item=json.loads(line)
            for key in ("family","event_family","scenario_family"):
                if item.get(key):
                    train_families.add(item[key])
    shared_families=reserved_families & train_families
    report={"purpose":"training versus reserved validation isolation audit; no model inference",
            "created_date":"2026-10-04","training_rows":len(train),"validation_rows":len(heldout),
            "training_candidate_sha256":digest(pool/"candidates.jsonl"),
            "training_data_file":training_file,"training_data_sha256":digest(pool/training_file),
            "validation_candidate_sha256":digest(validation/"candidates.jsonl"),
            "shared_ids":sorted(shared_ids),"shared_normalized_bodies":sorted(shared_bodies),
            "shared_current_or_recent_windows":sorted(shared_windows),"shared_declared_families":sorted(shared_families),
            "reserved_validation_windows":len(windows(heldout)),
            "passed":not(shared_ids or shared_bodies or shared_windows or shared_families),
            "limitation":"exact normalized body/window and declared family checks; not exhaustive semantic independence",
            "validation_review_scope":"entire candidate pool excluded from training, regardless of partial approval"}
    return report


def run(runs,task,task_run=None):
    import torch
    from peft import PeftModel
    output = runs/("train_fit_"+task+".json")
    if output.exists() or output.with_suffix(".predictions.jsonl").exists():
        raise ValueError("preserve previous fit report")
    task_run = task_run or runs/task
    config = json.loads((task_run/"run_config.json").read_text(encoding="utf-8"))
    assert config["task"] == task and config["prompt_sha256"] == prompt_digest(task)
    assert config["prepared_manifest_sha256"] == digest(OUTPUT/task/"manifest.json")
    tokenizer, model, _, ids = load_model()
    assert config["digit_token_ids"] == ids
    # Load only this task's adapter, independently of validation adapter switching.
    model = PeftModel.from_pretrained(model,task_run/"adapter")
    model.eval()
    records = [json.loads(v) for v in (OUTPUT/task/"train.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((OUTPUT/task/"manifest.json").read_text(encoding="utf-8"))
    assert digest(OUTPUT/task/"train.jsonl") == manifest["splits"]["train"]["file_sha256"]
    rows = []
    for notification_id,record in zip(manifest["splits"]["train"]["notification_ids"],records,strict=True):
        inputs = {k:torch.tensor([v],device=model.device) for k,v in encode(tokenizer,replace_prompt(record["messages"],task)).items()}
        with torch.inference_mode():
            logits = model(**inputs,logits_to_keep=1,use_cache=False).logits[0,-1,ids].float()
        rows.append({"notification_id":notification_id,"gold":parse_scores(record["messages"][-1]["content"],task)[task+"_score"],
            "prediction":int(logits.argmax())+1,"conditional_probabilities":logits.softmax(-1).tolist()})
    def metrics(selected):
        return {"count":len(selected),"exact_correct":sum(r["gold"]==r["prediction"] for r in selected),
            "gold_counts":dict(Counter(r["gold"] for r in selected)),"prediction_counts":dict(Counter(r["prediction"] for r in selected)),
            "urgent_false_blocks":sum(r["gold"]>=4 and r["prediction"]<4 for r in selected) if task=="urgency" else None,
            "urgent_count":sum(r["gold"]>=4 for r in selected) if task=="urgency" else None}
    write_lines(output.with_suffix(".predictions.jsonl"),rows)
    report = {"purpose":"full training-set fit diagnostic; not holdout accuracy", "created_date":"2026-10-04","task":task,
        "run_config_sha256":digest(task_run/"run_config.json"),"train_sha256":digest(OUTPUT/task/"train.jsonl"),
        "predictions_sha256":digest(output.with_suffix(".predictions.jsonl")),"all_train":metrics(rows),
        "approved_supplement_train":metrics([r for r in rows if r["notification_id"].startswith("sep02_")])}
    if config.get("diagnostic_subset"):
        indices = config["selected_prepared_indices"]
        report["selected_train_fit"] = metrics([rows[i] for i in indices])
        report["selected_train_ids"] = [rows[i]["notification_id"] for i in indices]
        report["note"] = "tiny-subset memorization diagnostic; remaining train rows are not a validation split"
    write_json(output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def audit_relevance_review_backlog(folder):
    """Count review candidates across all active rows without changing labels.

    Resource decisions are task-by-task editorial judgments, not measured truth.
    o: direct help; a: applicability uncertain; x: different action/no clear help.
    """
    from collections import defaultdict
    from datetime import datetime
    from zoneinfo import ZoneInfo

    applicability = [
        "ooo ooo ooo ooo ooo", "oao aaa ooo aaa ooo",
        "ooo ooo ooo ooo ooo", "oaa axx oxx aax oaa",
        "ooo ooo ooo aoo ooo", "ooo ooo ooo ooo ooo",
        "oxo xoa xoa oxa oxo", "oxo aox ooo axx ooo",
        "ooo ooo ooo ooo ooo", "oao oao aao oao ooo",
        "oxo xoa xaa aoo oxo", "xxx xxx ooo oao xox",
        "oxa aoo ooa xxo oxa", "oao oox axx oao xoa",
        "ooo ooo ooo ooo ooo", "oxo xxx ooo oxo aox",
        "ooo ooo xox ooo oao", "ooo aoa ooo ooo xox",
        "ooo oox ooo ooo axo", "ooa ooa ooa ooa xxo",
        "ooo ooa ooo ooo aox", "oxo oox oxa oox oxo",
        "oox oox oox xxo oox", "oxa oax xox axo oxo",
        "ooo ooo ooo aao oxo", "ooo ooo ooo ooo ooo",
        "ooo ooo ooo ooo ooo", "ooo ooo ooo ooo ooo",
        "ooa ooo ooa oxo oao", "ooo ooo ooo ooo ooo",
    ]
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bulk = manifest["low_score_supplement"]["bulk_candidates"]
    path = folder / bulk["dataset_file"]
    if digest(path) != bulk["dataset_sha256"]:
        raise ValueError("candidate data changed")
    for source, sha in bulk["preserved_sources"].items():
        if digest(Path(source)) != sha:
            raise ValueError("protected data changed: " + source)
    samples = load_score_samples(path)
    by_id = {row.notification.id: row for row in samples}
    confirmed = set(bulk["approved_ids"]) | set(bulk.get("feedback_adjusted_ids", []))
    new_flags = []
    decisions = []
    for task in bulk["task_records"]:
        index = task["task_index"] - 1
        statuses = applicability[index // 5].split()[index % 5]
        for resource_index, status in enumerate(statuses):
            identifier = task["ids"][17 + resource_index]
            sample = by_id[identifier]
            confirmed_row = identifier in confirmed
            decisions.append({"id": identifier, "goal": task["goal"],
                              "judgment": status, "score": sample.label.relevance_score,
                              "human_feedback_precedence": confirmed_row})
            if not confirmed_row and sample.label.relevance_score >= 4 and status != "o":
                new_flags.append({"id": identifier, "goal": task["goal"],
                                  "reason": "different_action_or_no_clear_help" if status == "x" else "uncertain_direct_help"})
    # Check every remaining low-score row against the authored recipe structure.
    assert len(samples) == 3000 and len(decisions) == 450
    assert all(1 <= sample.label.urgency_score <= 5 and 1 <= sample.label.relevance_score <= 5 for sample in samples)
    normalized = lambda text: re.sub(r"\s+", "", text).casefold()
    assert len({normalized(sample.notification.body) for sample in samples}) == 3000

    old_path = Path(manifest["old_replay_conflict_audit"]["source"])
    old_rows = [json.loads(line) for line in old_path.read_text(encoding="utf-8").splitlines()]
    decoded = []
    for record in old_rows:
        value = json.loads(record["messages"][1]["content"])
        label = parse_scores(record["messages"][-1]["content"], "relevance")["relevance_score"]
        decoded.append({"id": value["notification"]["id"], "body": value["notification"]["body"],
                        "context": value["context"], "score": label})
    assert len({row["id"] for row in decoded}) == len(decoded)
    grouped = defaultdict(list)
    for row in decoded:
        context = row["context"]
        grouped[(normalized(row["body"]), context["active_process"], context["window_title"])].append(row)
    reasons = defaultdict(set)
    threshold_ids = set()
    pair_count = 0
    for values in grouped.values():
        matches = [row for row in values if row["id"].endswith("_matching")]
        switches = [row for row in values if row["id"].endswith("_switched")]
        if len(matches) == len(switches) == 1:
            high, low = matches[0], switches[0]
            if high["score"] != low["score"]:
                pair_count += 1
                reasons[low["id"]].add("same_body_current_window_different_history_score")
                if (high["score"] >= 4) != (low["score"] >= 4):
                    threshold_ids.add(low["id"])
    for row in decoded:
        context = row["context"]
        if not context["window_title"].strip() and not context.get("recent_windows"):
            reasons[row["id"]].add("no_work_context_but_relevance_gold")
        if re.fullmatch(r"train05_\d{3}_archive_(matching|empty_recent)", row["id"]) and row["score"] >= 4:
            reasons[row["id"]].add("archived_past_artifact_without_current_help")
        if re.fullmatch(r"train05_\d{3}_bookmark_(matching|empty_recent)", row["id"]) and row["score"] == 5:
            reasons[row["id"]].add("generic_reference_location_scored_as_exact_target")
    # Urgency train is one notification per original; check all rows for schema,
    # context leakage and contradictory scores for the same exact notification.
    urgency_path = old_path.parent.parent / "urgency" / "train.jsonl"
    urgency_rows = [json.loads(line) for line in urgency_path.read_text(encoding="utf-8").splitlines()]
    urgency_groups = defaultdict(set)
    for record in urgency_rows:
        value = json.loads(record["messages"][1]["content"])
        score = parse_scores(record["messages"][-1]["content"], "urgency")["urgency_score"]
        assert "context" not in value
        urgency_groups[normalized(value["notification"]["body"])].add(score)
    by_old_id = {row["id"]: row for row in decoded}
    counts = Counter(reason for values in reasons.values() for reason in values)
    active_originals = {normalized(row["body"]) for row in decoded}
    old_flagged_originals = {normalized(by_old_id[i]["body"]) for i in reasons}
    new_flag_ids = {row["id"] for row in new_flags}
    assert not active_originals & {normalized(row.notification.body) for row in samples}
    report = {
        "purpose": "full active dataset review backlog; no label edits or model evaluation",
        "created_date": date,
        "scope": {"new_originals_and_rows": len(samples), "old_relevance_rows": len(decoded),
                  "old_urgency_rows": len(urgency_rows), "old_distinct_notification_bodies": len(active_originals),
                  "historical_candidate_versions_not_double_counted": True,
                  "v6_pending_pool_not_selected_for_this_merge": True},
        "method": "read every active row; review all 150 task x 3 resource combinations; apply structural conflict rules to every old row",
        "limitation": "review flags are editorial/structural candidates, not externally confirmed errors; unflagged rows are not all human-reviewed",
        "new": {"flagged_rows_and_originals": len(new_flag_ids), "reason_counts": dict(Counter(row["reason"] for row in new_flags)),
                "already_reviewed_rows": len(confirmed), "flagged_rows": new_flags,
                "resource_decisions": decisions, "applicability_matrix": applicability},
        "old": {"flagged_rows": len(reasons), "flagged_distinct_notification_bodies": len(old_flagged_originals),
                "reason_counts": dict(counts), "history_score_conflict_pairs": pair_count,
                "history_threshold_conflict_pairs": len(threshold_ids),
                "flagged_ids_and_reasons": {identifier: sorted(values) for identifier, values in reasons.items()},
                "urgency_contradictory_exact_body_groups": sum(len(values) > 1 for values in urgency_groups.values()),
                "urgency_semantic_correctness_not_established_by_structural_check": True},
        "combined": {"flagged_rows_without_double_counting": len(reasons) + len(new_flag_ids),
                     "flagged_distinct_notification_bodies": len(old_flagged_originals) + len(new_flag_ids)},
        "source_hashes": {str(path): digest(path), str(old_path): digest(old_path), str(urgency_path): digest(urgency_path)},
        "labels_changed": False, "training_run": False,
    }
    manifest["full_review_backlog_audit"] = report
    write_json(manifest_path, manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path)
    parser.add_argument("--task",choices=("urgency","relevance"))
    parser.add_argument("--task-run",type=Path)
    parser.add_argument("--pool",type=Path)
    parser.add_argument("--validation-pool",type=Path)
    parser.add_argument("--review-audit", type=Path)
    args = parser.parse_args()
    if args.review_audit:
        report = audit_relevance_review_backlog(args.review_audit)
        print(json.dumps({"scope": report["scope"], "new": {key: report["new"][key] for key in ("flagged_rows_and_originals", "reason_counts")},
                          "old": {key: report["old"][key] for key in ("flagged_rows", "flagged_distinct_notification_bodies", "reason_counts", "history_threshold_conflict_pairs")},
                          "combined": report["combined"]}, ensure_ascii=False, indent=2))
    elif args.pool or args.validation_pool:
        if not args.pool or not args.validation_pool:
            parser.error("isolation audit requires --pool and --validation-pool")
        report=audit_pool_separation(args.pool,args.validation_pool)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    else:
        if not args.runs or not args.task:
            parser.error("fit audit requires --runs and --task")
        run(args.runs,args.task,args.task_run)
