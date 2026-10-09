"""용도: 사용자 수정과 승인 이력을 보존하며 새 보강 묶음을 학습 전 확정한다.
생성일: 2026-10-04
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from filtering_training.common.dataset import load_samples
from filtering_training.preparation.prepare_score_experiment import digest, write_json, write_lines
from filtering_training.common.score_tasks import unique_urgency_samples
from filtering_training.common.score_tasks import load_score_samples
from src.filtering.policy import POLICY_VERSION
from src.filtering.schema import FilteringSample
from src.filtering.prompt import parse_model_output


def freeze_unmodified(source, reviewed, output, approval_text):
    if output.exists() or not approval_text.strip():
        raise ValueError("fresh output and explicit approval required")
    manifest = json.loads((reviewed/"manifest.json").read_text(encoding="utf-8"))
    for name,key in (("candidates.jsonl","candidate_sha256"),("review.md","review_sha256")):
        if digest(reviewed/name) != manifest[key]:
            raise ValueError("review materials changed")
    if digest(source/"dataset.jsonl") != manifest["source_dataset_sha256"] or digest(source/"prepared/manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("source splits changed")
    samples,normalizations = [],[]
    for line in (reviewed/"candidates.jsonl").read_text(encoding="utf-8").splitlines():
        sample = FilteringSample.model_validate_json(line)
        if sample.label.category == "보안/인증":
            normalizations.append({"notification_id":sample.notification.id,"field":"category","before":"보안/인증","after":"시스템/보안"})
            sample.label.category = "시스템/보안"
        parse_model_output(json.dumps(sample.label.model_dump(),ensure_ascii=False))
        samples.append(sample)
    originals = unique_urgency_samples(samples)
    train_ids = set(json.loads((source/"prepared/manifest.json").read_text(encoding="utf-8"))["splits"]["train"]["message_ids"])
    if any(not set(item["train_links"]).issubset(train_ids) for item in manifest["families"].values()):
        raise ValueError("declared semantic links must be train-only")
    output.mkdir(parents=True)
    write_lines(output/"dataset.jsonl",[s.model_dump(mode="json") for s in samples])
    write_json(output/"approval.json",{"purpose":"freeze exact human-approved expansion preview", "created_date":"2026-10-04",
        "user_statement":approval_text,"reviewed_originals":len(originals),"reviewed_context_rows":len(samples),
        "review_manifest_sha256":digest(reviewed/"manifest.json"),"candidate_sha256":digest(reviewed/"candidates.jsonl"),
        "review_sha256":digest(reviewed/"review.md"),"dataset_sha256":digest(output/"dataset.jsonl"),
        "scope":"only displayed preview; approval of generation criteria does not individually review generated bulk rows",
        "schema_name_normalizations":normalizations,"scores_changed":False,"training_used":False})
    write_json(output/"manifest.json",{"purpose":"approved train-only expansion preview", "created_date":"2026-10-04",
        "source_sha256":digest(output/"dataset.jsonl"),"source_v3_sha256":digest(source/"dataset.jsonl"),
        "approval_sha256":digest(output/"approval.json"),"families":manifest["families"],
        "originals":len(originals),"rows":len(samples),"policy_version":POLICY_VERSION})
    return {"originals":len(originals),"rows":len(samples),"schema_normalizations":len(normalizations)}


def approve_validation_preview(reviewed, approval_text):
    """Approve exactly the displayed cases, preserving the unreviewed pool."""
    if not approval_text.strip() or (reviewed/"approval.json").exists():
        raise ValueError("explicit approval and an unapproved preview required")
    manifest = json.loads((reviewed/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("intended_split") != "validation" or not manifest.get("displayed_review_ids"):
        raise ValueError("expected a partial validation preview")
    for name,key in (("candidates.jsonl","candidate_sha256"),("review.md","review_sha256")):
        if digest(reviewed/name) != manifest[key]:
            raise ValueError("displayed review changed")
    for item in manifest["protected_sources"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("protected source changed")
    by_id = {s.notification.id:s for s in load_score_samples(reviewed/"candidates.jsonl")}
    ids = manifest["displayed_review_ids"]
    if len(set(ids)) != len(ids) or not set(ids).issubset(by_id):
        raise ValueError("invalid displayed case ids")
    selected = [by_id[id_] for id_ in ids]
    count = len(unique_urgency_samples(selected))
    write_json(reviewed/"approval.json",{
        "purpose":"approval of displayed validation cases only; remaining rows are synthetic candidates",
        "created_date":"2026-10-04","user_statement":approval_text,"intended_split":"validation",
        "scope":"exactly displayed context rows; other variants of the same notification are not individually reviewed",
        "approved_ids":ids,"reviewed_originals":count,"reviewed_context_rows":len(selected),
        "approved_cases":[s.model_dump(mode="json") for s in selected],
        "candidate_sha256":manifest["candidate_sha256"],"review_sha256":manifest["review_sha256"],
        "scores_changed":False,"training_used":False,"model_evaluation_used":False})
    manifest["preview_approval"] = {"file":"approval.json","sha256":digest(reviewed/"approval.json"),
                                    "approved_ids":ids,"new_reviewed_originals":count,"new_reviewed_rows":len(selected)}
    manifest["new_human_reviewed_originals"] = count
    manifest["new_human_reviewed_context_rows"] = len(selected)
    manifest["status"] = "partially_reviewed_validation_candidate_pool"
    write_json(reviewed/"manifest.json",manifest)
    return {"displayed_originals":count,"approved_rows":len(selected),"pool_originals":manifest["originals"]}


def prepare_approved_validation_subset(reviewed):
    """Use existing approval evidence; never promote unreviewed variants."""
    if (reviewed/"dataset.jsonl").exists():
        raise ValueError("preserve frozen validation subset")
    manifest=json.loads((reviewed/"manifest.json").read_text(encoding="utf-8"))
    approval=json.loads((reviewed/"approval.json").read_text(encoding="utf-8"))
    if manifest.get("intended_split")!="validation":
        raise ValueError("expected reserved validation pool")
    if digest(reviewed/"candidates.jsonl")!=manifest["candidate_sha256"] or digest(reviewed/"approval.json")!=manifest["preview_approval"]["sha256"]:
        raise ValueError("validation candidate or approval changed")
    for item in manifest["protected_sources"]:
        if digest(Path(item["path"]))!=item["sha256"]:
            raise ValueError("protected source changed")
    candidates=load_score_samples(reviewed/"candidates.jsonl")
    by_id={s.notification.id:s for s in candidates}
    approved={}
    sources=[]
    for folder_name in ("v5_validation_review_01","v5_history_review_01"):
        folder=reviewed.parent/folder_name
        seed_approval=json.loads((folder/"approval.json").read_text(encoding="utf-8"))
        if seed_approval.get("intended_split")!="validation" or digest(folder/"dataset.jsonl")!=seed_approval["dataset_sha256"]:
            raise ValueError("approved validation seed changed")
        for sample in load_score_samples(folder/"dataset.jsonl"):
            approved[sample.notification.id]=sample.model_dump(mode="json")
        sources.append({"folder":str(folder),"dataset_sha256":digest(folder/"dataset.jsonl"),"approval_sha256":digest(folder/"approval.json")})
    if set(approval["approved_ids"])!={s["notification"]["id"] for s in approval["approved_cases"]}:
        raise ValueError("displayed approval ids differ")
    for item in approval["approved_cases"]:
        approved[item["notification"]["id"]]=item
    for id_,value in approved.items():
        if id_ not in by_id or by_id[id_].model_dump(mode="json")!=value:
            raise ValueError("pool changed an approved validation row")
    selected=[s for s in candidates if s.notification.id in approved]
    write_lines(reviewed/"dataset.jsonl",[s.model_dump(mode="json") for s in selected])
    report={"purpose":"frozen previously approved validation context rows only; remaining candidates excluded from headline metrics",
            "created_date":"2026-10-04","dataset_file":"dataset.jsonl","dataset_sha256":digest(reviewed/"dataset.jsonl"),
            "approved_ids":[s.notification.id for s in selected],"rows":len(selected),"originals":len(unique_urgency_samples(selected)),
            "seed_sources":sources,"preview_approval_sha256":digest(reviewed/"approval.json"),
            "excluded_unreviewed_rows":len(candidates)-len(selected),"training_used":False,"test_inference_used":False}
    manifest["evaluation_set"]=report
    manifest["files"]["dataset.jsonl"]="fifty approved validation rows; all 380 candidate rows remain reserved from training"
    write_json(reviewed/"manifest.json",manifest)
    return report


def freeze_validation(reviewed, output, approval_text):
    """Freeze validation in its existing folder, preserving the candidate snapshot."""
    if output.resolve() != reviewed.resolve() or not approval_text.strip():
        raise ValueError("validation approval requires its existing folder and explicit approval")
    if (output/"approval.json").exists() or (output/"dataset.jsonl").exists():
        raise ValueError("do not overwrite frozen validation approval")
    manifest = json.loads((reviewed/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("intended_split") != "validation" or manifest.get("training_used"):
        raise ValueError("only unused validation candidates can be frozen here")
    revision = manifest.get("revision")
    data_file = revision["dataset_file"] if revision else "candidates.jsonl"
    for name, expected in (("candidates.jsonl",manifest["candidate_sha256"]),
                           ("review.md",revision["review_sha256"] if revision else manifest["review_sha256"])):
        if digest(reviewed/name) != expected:
            raise ValueError("review materials changed")
    if revision:
        if digest(reviewed/data_file) != revision["dataset_sha256"] or digest(reviewed/revision["feedback_file"]) != revision["feedback_sha256"]:
            raise ValueError("revised validation materials changed")
    for item in manifest["protected_sources"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("protected source changed after candidate review")
    samples = load_score_samples(reviewed/data_file)
    if "displayed_review_ids" in manifest and set(manifest["displayed_review_ids"]) != {s.notification.id for s in samples}:
        raise ValueError("partial-review pools require per-case approval; do not freeze the full pool")
    originals = unique_urgency_samples(samples)
    if len(originals) != manifest["originals"] or len(samples) != manifest["rows"]:
        raise ValueError("validation counts changed")
    write_lines(output/"dataset.jsonl",[s.model_dump(mode="json") for s in samples])
    write_json(output/"approval.json",{
        "purpose":"human-approved validation labels; permanently excluded from training",
        "created_date":"2026-10-04","user_statement":approval_text,"intended_split":"validation",
        "reviewed_originals":len(originals),"reviewed_context_rows":len(samples),
        "scope":"only displayed candidates and their declared context variants; no approval of future cases",
        "candidate_sha256":digest(reviewed/"candidates.jsonl"),"review_sha256":digest(reviewed/"review.md"),
        "dataset_sha256":digest(output/"dataset.jsonl"),"candidate_manifest_sha256":digest(reviewed/"manifest.json"),
        "reserved_families":manifest["reserved_families"],"scores_changed":bool(revision),
        "revision":revision,
        "training_used":False,"model_evaluation_used":False,
        "note":"manifest remains the original candidate-creation snapshot; this approval records the current status"})
    return {"originals":len(originals),"rows":len(samples),"intended_split":"validation"}


def freeze(source, revised, output, approval_text):
    review_manifest = json.loads((revised/"manifest.json").read_text(encoding="utf-8"))
    if review_manifest.get("intended_split") == "validation":
        return freeze_validation(revised,output,approval_text)
    if output.exists():
        raise ValueError("do not overwrite frozen approval")
    manifest=json.loads((revised/"manifest.json").read_text(encoding="utf-8"))
    if "feedback_sha256" not in manifest:
        return freeze_unmodified(source,revised,output,approval_text)
    feedback=json.loads((revised/"feedback.json").read_text(encoding="utf-8"))
    if not approval_text.strip():
        raise ValueError("explicit approval statement required")
    for name,key in (("candidates.jsonl","candidate_sha256"),("review.md","review_sha256"),("feedback.json","feedback_sha256")):
        if digest(revised/name)!=manifest[key]:
            raise ValueError("review materials changed")
    if digest(source/"dataset.jsonl")!=manifest["source_dataset_sha256"] or digest(source/"prepared/manifest.json")!=manifest["source_manifest_sha256"]:
        raise ValueError("original splits changed")
    samples=load_samples(revised/"candidates.jsonl")
    by_id={s.notification.id:s for s in samples}
    for correction in feedback["corrections"]:
        if correction["field"]!="relevance_score" or by_id[correction["notification_id"]].label.relevance_score!=correction["after"]:
            raise ValueError("candidate scores differ from explicit user corrections")
    # Scores are the explicitly requested overrides. Reasons are authored provenance,
    # not a new claim that this joke has demonstrable task utility.
    by_id["sep02_06_related"].label.ai_summary_reason=(
        "선택적인 상품 홍보로 대응할 필요가 없습니다; 현재 스페인어 듣기 연습과 주제는 같지만 직접적인 도움은 없으며 사용자 검수 관련도는 2입니다.")
    by_id["sep02_07_related"].label.ai_summary_reason=(
        "농담으로 긴급 대응 요청은 없습니다; 현재 스페인어 동사 변화 학습 주제의 대화이며 사용자 검수에서 관련도 4로 지정했습니다.")
    unique_urgency_samples(samples)
    original_manifest=json.loads((source/"prepared/manifest.json").read_text(encoding="utf-8"))
    train_ids=set(original_manifest["splits"]["train"]["message_ids"])
    if any(not set(item["train_links"]).issubset(train_ids) for item in manifest["families"].values()):
        raise ValueError("a linked family is not train-only")
    source_windows={(s.context.active_process,s.context.window_title) for s in load_samples(source/"dataset.jsonl") if s.context.active_process}
    if any((s.context.active_process,s.context.window_title) in source_windows for s in samples if s.context.active_process):
        raise ValueError("candidate windows overlap original splits")
    output.mkdir(parents=True)
    write_lines(output/"dataset.jsonl",[s.model_dump(mode="json") for s in samples])
    write_json(output/"approval.json",{"purpose":"freeze user-approved score-review batch; not yet added to model training",
        "created_date":"2026-10-04","user_statement":approval_text,
        "approval_context":"proceed after cases 6 and 7 were corrected and their policy consequences displayed",
        "correction_statement":feedback["user_statement"],"revised_manifest_sha256":digest(revised/"manifest.json"),
        "feedback_sha256":digest(revised/"feedback.json"),"review_sha256":digest(revised/"review.md"),
        "dataset_sha256":digest(output/"dataset.jsonl"),"reviewed_originals":8,"reviewed_context_rows":24,
        "scope":"notification/context/category/scores accepted; revised reason text authored from score feedback",
        "guideline_override_ids":feedback["guideline_conflict_ids"],"global_prompt_changed":False,
        "training_used":False,"original_v3_review_scope_unchanged":True})
    write_json(output/"manifest.json",{"purpose":"approved train-only supplement; not holdout evaluation data",
        "created_date":"2026-10-04","source_sha256":digest(output/"dataset.jsonl"),
        "source_v3_sha256":manifest["source_dataset_sha256"],"policy_version":POLICY_VERSION,
        "families":manifest["families"],"urgency_counts":dict(Counter(s.label.urgency_score for s in samples)),
        "relevance_counts":dict(Counter(s.label.relevance_score for s in samples)),
        "approval_sha256":digest(output/"approval.json"),
        "files":{"dataset.jsonl":"approved 8 originals / 24 context variants","approval.json":"approval and correction lineage"},
        "note":"case 7 is an explicit user exception; inclusion in a global-policy run must remain identifiable"})
    return {"originals":8,"context_rows":len(samples),"override_ids":feedback["guideline_conflict_ids"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("filtering_training/outputs/v3_reviewed_01"))
    parser.add_argument("--revised",type=Path,default=Path("filtering_training/outputs/v3_score_review_02_revision_01"))
    parser.add_argument("--output",type=Path,default=Path("filtering_training/outputs/v3_score_review_02_approved_01"))
    parser.add_argument("--approval-text",required=True)
    args=parser.parse_args()
    print(json.dumps(freeze(args.source,args.revised,args.output,args.approval_text),ensure_ascii=False))


if __name__=="__main__":
    main()
