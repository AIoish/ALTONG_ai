"""용도: 승인 보강 자료를 기존 학습에만 추가하고 검증 분할을 보존한다.
생성일: 2026-10-04
"""

from copy import deepcopy
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from filtering_training.common.dataset import load_samples
from filtering_training.common.score_tasks import sft_record, unique_urgency_samples, load_score_samples
from filtering_training.modeling.digit_score_experiment import SOURCE, PREPARED
from filtering_training.preparation.prepare_score_experiment import digest, write_json, write_lines

APPROVED = Path("filtering_training/outputs/v3_score_review_02_approved_01")
OUTPUT = Path("filtering_training/outputs/v3_digit_supplement_01/prepared")


def prepare(output=OUTPUT):
    if output.exists():
        raise ValueError("preserve prepared versions")
    manifest = json.loads((APPROVED/"manifest.json").read_text(encoding="utf-8"))
    approval = json.loads((APPROVED/"approval.json").read_text(encoding="utf-8"))
    assert digest(APPROVED/"dataset.jsonl") == manifest["source_sha256"] == approval["dataset_sha256"]
    assert digest(APPROVED/"approval.json") == manifest["approval_sha256"]
    assert digest(SOURCE/"dataset.jsonl") == manifest["source_v3_sha256"]
    samples = load_samples(APPROVED/"dataset.jsonl")
    assert len(samples) == 24 and len(unique_urgency_samples(samples)) == 8
    by_id = {s.notification.id:s for s in samples}
    assert by_id["sep02_06_related"].label.relevance_score == 2
    assert by_id["sep02_07_related"].label.relevance_score == 4
    source_manifest = json.loads((SOURCE/"prepared/manifest.json").read_text(encoding="utf-8"))
    train_families = set(source_manifest["splits"]["train"]["message_ids"])
    heldout = set(source_manifest["splits"]["validation"]["message_ids"]) | set(source_manifest["splits"]["test"]["message_ids"])
    for info in manifest["families"].values():
        assert set(info["train_links"]).issubset(train_families)
        assert not set(info["train_links"]) & heldout
    originals = load_samples(SOURCE/"dataset.jsonl")
    def body(s):
        value = s.notification.model_dump(mode="json")
        value.pop("id")
        value.pop("timestamp")
        return json.dumps(value,sort_keys=True,ensure_ascii=False)
    assert not {body(s) for s in samples} & {body(s) for s in originals}
    original_windows = {(s.context.active_process,s.context.window_title) for s in originals if s.context.active_process}
    assert not original_windows & {(s.context.active_process,s.context.window_title) for s in samples if s.context.active_process}
    for task in ("urgency","relevance"):
        folder = output/task
        folder.mkdir(parents=True)
        previous = json.loads((PREPARED/task/"manifest.json").read_text(encoding="utf-8"))
        for split in ("train","validation"):
            assert digest(PREPARED/task/f"{split}.jsonl") == previous["splits"][split]["file_sha256"]
        selected = unique_urgency_samples(samples) if task == "urgency" else samples
        records = [json.loads(v) for v in (PREPARED/task/"train.jsonl").read_text(encoding="utf-8").splitlines()]
        write_lines(folder/"train.jsonl",records+[sft_record(s,task) for s in selected])
        (folder/"validation.jsonl").write_bytes((PREPARED/task/"validation.jsonl").read_bytes())
        current = deepcopy(previous)
        current.update(purpose="approved supplement in train only; fixed original validation; no test inference",created_date="2026-10-04")
        current["training_additions"] = len(selected)
        current["splits"]["train"]["count"] += len(selected)
        current["splits"]["train"]["notification_ids"] += [s.notification.id for s in selected]
        current["splits"]["train"]["file_sha256"] = digest(folder/"train.jsonl")
        current["supplement"] = {"dataset_sha256":digest(APPROVED/"dataset.jsonl"),"approval_sha256":digest(APPROVED/"approval.json"),
            "manifest_sha256":digest(APPROVED/"manifest.json"),"original_prepared_manifest_sha256":digest(PREPARED/task/"manifest.json"),
            "guideline_override_ids":approval["guideline_override_ids"],"global_prompt_changed":False,
            "notification_ids":[s.notification.id for s in selected],"train_only":True,
            "leakage_checks":"exact notification/window disjointness and declared family links; not exhaustive semantic proof"}
        write_json(folder/"manifest.json",current)
    return output


def cap_digital_urgency(samples, protected_ids=(), seed=42):
    """Use distinct originals without replacement; preserve earlier training seeds."""
    protected = set(protected_ids)
    digital = [s for s in samples if s.label.urgency_score == 5 and s.label.category == "시스템/보안"]
    other = [s for s in samples if s.label.urgency_score == 5 and s.label.category != "시스템/보안"]
    keep = [s for s in digital if s.notification.id in protected]
    candidates = [s for s in digital if s.notification.id not in protected]
    maximum = len(other)//3  # digital/(digital+other) <= 1/4
    if len(keep) > maximum:
        raise ValueError("protected digital seeds exceed cap; add non-digital coverage first")
    random.Random(seed).shuffle(candidates)
    keep += candidates[:max(0,maximum-len(keep))]
    retained = {s.notification.id for s in keep}
    digital_ids = {s.notification.id for s in digital}
    selected = [s for s in samples if s.notification.id not in digital_ids or s.notification.id in retained]
    report = {"method":"deterministic subset without replacement; natural training only", "seed":seed,
              "urgency5_digital_before":len(digital),"urgency5_other":len(other),
              "urgency5_digital_selected":len(keep),"urgency5_digital_fraction":len(keep)/(len(keep)+len(other)) if keep or other else 0,
              "removed_from_active_train_only":len(digital)-len(keep),"required_mode":"natural"}
    return selected, report


def prepare_pool(pool, output, tokenizer=None):
    if output.exists():
        raise ValueError("preserve prepared versions")
    pool_manifest = json.loads((pool/"manifest.json").read_text(encoding="utf-8"))
    for name,key in (("candidates.jsonl","candidate_sha256"),("lineage.jsonl","lineage_sha256")):
        if digest(pool/name) != pool_manifest[key]:
            raise ValueError("candidate pool changed")
    source_manifest = json.loads((SOURCE/"prepared/manifest.json").read_text(encoding="utf-8"))
    if digest(SOURCE/"dataset.jsonl") != pool_manifest["source_dataset_sha256"] or digest(SOURCE/"prepared/manifest.json") != pool_manifest["source_manifest_sha256"]:
        raise ValueError("original source or splits changed")
    urgent_approved = SOURCE.parent/"v4_urgent_review_01_approved"
    if digest(urgent_approved/"dataset.jsonl") != pool_manifest["approved_dataset_sha256"] or digest(urgent_approved/"approval.json") != pool_manifest["approval_sha256"]:
        raise ValueError("pool approval snapshot changed")
    holdout = set(source_manifest["splits"]["validation"]["message_ids"]) | set(source_manifest["splits"]["test"]["message_ids"])
    train_families = set(source_manifest["splits"]["train"]["message_ids"])
    lineage = [json.loads(v) for v in (pool/"lineage.jsonl").read_text(encoding="utf-8").splitlines()]
    if any(not set(info.get("train_links",())).issubset(train_families) or set(info.get("train_links",())) & holdout for info in lineage):
        raise ValueError("declared semantic link overlaps heldout data")
    additions = load_samples(pool/"candidates.jsonl")
    if len({s.notification.id for s in additions}) != len(additions) or {s.notification.id for s in additions} != {info["notification_id"] for info in lineage}:
        raise ValueError("candidate and lineage ids differ")
    originals = load_samples(SOURCE/"dataset.jsonl")
    normalize = lambda text:re.sub(r"\s+","",text).casefold()
    bodies = {normalize(s.notification.body) for s in originals}
    windows = {(s.context.active_process,s.context.window_title) for s in originals if s.context.active_process}
    if any(normalize(s.notification.body) in bodies or (s.context.active_process and (s.context.active_process,s.context.window_title) in windows) for s in additions):
        raise ValueError("candidate overlaps original notification or task window")
    snapshots = {}
    approved_samples = []
    for folder in (APPROVED,SOURCE.parent/"v3_expansion_review_01_approved",SOURCE.parent/"v4_urgent_review_01_approved"):
        approval = json.loads((folder/"approval.json").read_text(encoding="utf-8"))
        info = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        if digest(folder/"dataset.jsonl") != approval["dataset_sha256"] or digest(folder/"approval.json") != info["approval_sha256"] or info["source_v3_sha256"] != digest(SOURCE/"dataset.jsonl"):
            raise ValueError("approved source changed")
        if any(not set(item["train_links"]).issubset(train_families) for item in info["families"].values()):
            raise ValueError("approved family overlaps heldout data")
        approved_samples += load_samples(folder/"dataset.jsonl")
        snapshots[str(folder)] = {"dataset_sha256":digest(folder/"dataset.jsonl"),"approval_sha256":digest(folder/"approval.json")}
    all_by_id = {s.notification.id:s for s in originals+approved_samples}
    pool_ids = {s.notification.id for s in additions}
    overlap_approved = [s for s in approved_samples if s.notification.id in pool_ids]
    pool_by_id = {s.notification.id:s for s in additions}
    if any(s.model_dump(mode="json") != pool_by_id[s.notification.id].model_dump(mode="json") for s in overlap_approved):
        raise ValueError("pool changed an approved record")
    extra = [s for s in load_samples(SOURCE.parent/"v3_expansion_review_01_approved/dataset.jsonl") if s.notification.id not in pool_ids]
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",local_files_only=True)
    from filtering_training.common.digit_scores import replace_prompt, prompt_digest as digit_prompt_digest
    plans = {}
    for task in ("urgency","relevance"):
        baseline = OUTPUT/task
        previous = json.loads((baseline/"manifest.json").read_text(encoding="utf-8"))
        for split in ("train","validation"):
            if digest(baseline/f"{split}.jsonl") != previous["splits"][split]["file_sha256"]:
                raise ValueError("baseline split changed")
        seeds = [all_by_id[id_] for id_ in previous["splits"]["train"]["notification_ids"]]
        selected = seeds+extra+additions
        sampling = {"required_mode":"natural","method":"all relevance contexts"}
        if task == "urgency":
            selected = unique_urgency_samples(selected)
            selected,sampling = cap_digital_urgency(selected,[s.notification.id for s in seeds])
        random.Random(42).shuffle(selected)
        if len({s.notification.id for s in selected}) != len(selected):
            raise ValueError("duplicate selected ids")
        records = [sft_record(s,task) for s in selected]
        lengths = []
        for record in records:
            payload = json.loads(record["messages"][1]["content"])
            if task == "urgency" and ("context" in payload or "id" in payload["notification"]):
                raise ValueError("urgency prompt leaks context or variant id")
            encoded = tokenizer.apply_chat_template(replace_prompt(record["messages"],task),add_generation_prompt=True,
                         enable_thinking=False,tokenize=True,return_dict=True)
            lengths.append(len(encoded["input_ids"]))
        if not lengths or max(lengths) > 768:
            raise ValueError("prepared prompt exceeds training token limit")
        plans[task] = (selected,records,previous,sampling,{"count":len(lengths),"max":max(lengths),"min":min(lengths),"mean":sum(lengths)/len(lengths),"limit":768})
    for task,(selected,records,previous,sampling,lengths) in plans.items():
        folder = output/task
        folder.mkdir(parents=True)
        write_lines(folder/"train.jsonl",records)
        (folder/"validation.jsonl").write_bytes((OUTPUT/task/"validation.jsonl").read_bytes())
        current = deepcopy(previous)
        current.update(purpose="large synthetic pool plus approved seeds; fixed validation; no final-test inference",created_date="2026-10-04")
        current["splits"]["train"] = {"count":len(records),"notification_ids":[s.notification.id for s in selected],"file_sha256":digest(folder/"train.jsonl")}
        current["training_additions"] = len(records)-previous["splits"]["train"]["count"]
        current["pool"] = {"manifest_sha256":digest(pool/"manifest.json"),"candidate_sha256":digest(pool/"candidates.jsonl"),
                           "lineage_sha256":digest(pool/"lineage.jsonl"),"approved_sources":snapshots,"review_scope":"bulk remains synthetic; only approval snapshots are individually reviewed"}
        current["sampling"] = sampling
        current["training_score_counts"] = dict(Counter(getattr(s.label,task+"_score") for s in selected))
        current["token_lengths"] = lengths
        current["digit_prompt_sha256"] = digit_prompt_digest(task)
        current["training_ready"] = True
        current["training_used"] = False
        current["test_inference_used"] = False
        current["leakage_checks"] = "fixed heldout ids and validation bytes; exact body/window exclusion; declared family links; not exhaustive semantic proof"
        write_json(folder/"manifest.json",current)
    return {task:{"train_rows":len(value[1]),"sampling":value[3],"token_lengths":value[4]} for task,value in plans.items()}


def prepare_history_pool(pool, validation, output, tokenizer=None):
    """Prepare a gated draft, preserving old train rows and fixed validation."""
    from filtering_training.common.digit_scores import replace_prompt, prompt_digest as digit_digest
    from filtering_training.quality.audit_digit_training import audit_pool_separation
    if output.exists():
        raise ValueError("preserve prepared versions")
    pool_manifest = json.loads((pool/"manifest.json").read_text(encoding="utf-8"))
    refinement = pool_manifest.get("label_revision", pool_manifest["wording_refinement"])
    approval = json.loads((pool/"approval.json").read_text(encoding="utf-8"))
    approval_hash = refinement.get("approval_sha256", pool_manifest["revision"]["approval_sha256"])
    if digest(pool/"approval.json") != approval_hash:
        raise ValueError("training approval changed")
    audit = audit_pool_separation(pool,validation,refinement["dataset_file"])
    if not audit["passed"]:
        raise ValueError("refined pool overlaps reserved validation")
    samples = load_score_samples(pool/refinement["dataset_file"])
    by_id = {s.notification.id:s for s in samples}
    approved_cases=approval["approved_cases"]+approval.get("wording_review",{}).get("approved_cases",[])
    for item in approved_cases:
        if by_id[item["notification"]["id"]].model_dump(mode="json") != item:
            raise ValueError("approved training case changed")
    additions = [s for s in samples if s.notification.id.startswith("train05_")]
    baseline = pool.parent/"v4_training_pool_01/prepared"
    if digest(baseline.parent/"candidates.jsonl") != pool_manifest["parent_candidate_sha256"]:
        raise ValueError("parent training pool changed")
    heldout = load_score_samples(validation/"candidates.jsonl")
    reserved_ids={s.notification.id for s in heldout}
    normalize=lambda text:re.sub(r"\s+","",text).casefold()
    reserved_bodies={normalize(s.notification.body) for s in heldout}
    reserved_windows=set()
    def windows(context):
        result=set()
        if context.get("active_process") and context.get("window_title","").strip():
            result.add((context["active_process"].casefold(),normalize(context["window_title"])))
        for window in context.get("recent_windows",[]):
            if window["window_title"].strip():
                result.add((window["app_name"].casefold(),normalize(window["window_title"])))
        return result
    for sample in heldout:
        reserved_windows |= windows(sample.context.model_dump(mode="json"))
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",local_files_only=True)
    plans={}
    for task in ("urgency","relevance"):
        previous=json.loads((baseline/task/"manifest.json").read_text(encoding="utf-8"))
        if previous["source_sha256"] != digest(SOURCE/"dataset.jsonl") or previous["source_manifest_sha256"] != digest(SOURCE/"prepared/manifest.json"):
            raise ValueError("original source or split changed")
        for split in ("train","validation"):
            if digest(baseline/task/f"{split}.jsonl") != previous["splits"][split]["file_sha256"]:
                raise ValueError("previous prepared split changed")
        old=[json.loads(line) for line in (baseline/task/"train.jsonl").read_text(encoding="utf-8").splitlines()]
        selected=unique_urgency_samples(additions) if task=="urgency" else additions
        records=old+[sft_record(s,task) for s in selected]
        ids=previous["splits"]["train"]["notification_ids"]+[s.notification.id for s in selected]
        if len(ids)!=len(records) or len(set(ids))!=len(ids) or set(ids)&reserved_ids:
            raise ValueError("training ids duplicated or reserved")
        history_records=[sft_record(s,task) for s in heldout]
        lengths=[]
        for record in records:
            payload=json.loads(record["messages"][1]["content"])
            if normalize(payload["notification"]["body"]) in reserved_bodies or windows(payload.get("context",{}))&reserved_windows:
                raise ValueError("combined training overlaps reserved body or window")
            if task=="urgency" and ("context" in payload or "id" in payload["notification"]):
                raise ValueError("urgency input leaks context or variant id")
        for record in records+history_records:
            effective=replace_prompt(record["messages"],task)
            if "id" in json.loads(effective[1]["content"])["notification"]:
                raise ValueError("effective input leaks variant id")
            lengths.append(len(tokenizer.apply_chat_template(effective,add_generation_prompt=True,
                           enable_thinking=False,tokenize=True,return_dict=True)["input_ids"]))
        if max(lengths)>768:
            raise ValueError("prepared prompt exceeds training token limit")
        plans[task]=(records,ids,history_records,previous,{
            "count":len(lengths),"max":max(lengths),"mean":sum(lengths)/len(lengths),"limit":768})
    for task,(records,ids,history_records,previous,lengths) in plans.items():
        folder=output/task
        folder.mkdir(parents=True)
        write_lines(folder/"train.jsonl",records)
        (folder/"validation.jsonl").write_bytes((baseline/task/"validation.jsonl").read_bytes())
        write_lines(folder/"validation_history.jsonl",history_records)
        current=deepcopy(previous)
        current.update(purpose="history expansion draft; fixed legacy validation and separately reserved history validation",
                       created_date="2026-10-04",training_ready=False,training_used=False,test_inference_used=False)
        current["splits"]["train"]={"count":len(records),"notification_ids":ids,"file_sha256":digest(folder/"train.jsonl")}
        current["splits"]["validation_history"]={"count":len(history_records),
              "notification_ids":[s.notification.id for s in heldout],"file_sha256":digest(folder/"validation_history.jsonl"),
              "review_scope":"partial human review; remaining labels synthetic; separate from fixed legacy validation"}
        current["training_additions"]=len(records)-previous["splits"]["train"]["count"]
        current["training_score_counts"]=dict(Counter(json.loads(r["messages"][-1]["content"])[task+"_score"] for r in records))
        current["token_lengths"]=lengths
        current["digit_prompt_hashes"]={"legacy":digit_digest(task),"history":digit_digest(task,history=True)}
        current["pool"]={"dataset_sha256":refinement["dataset_sha256"],"dataset_file":refinement["dataset_file"],
             "approval_sha256":digest(pool/"approval.json"),"manifest_sha256":digest(pool/"manifest.json"),
             "parent_prepared_manifest_sha256":digest(baseline/task/"manifest.json"),
             "review_scope":"only ten displayed new rows approved; refined wording and remaining rows synthetic"}
        current["quality_gate"]={"pending":pool_manifest["quality_gate"]["pending"],"training_ready":False}
        current["validation_isolation_audit"]=audit
        write_json(folder/"manifest.json",current)
    return {task:{"train_rows":len(value[0]),"new_rows":len(value[0])-value[3]["splits"]["train"]["count"],
                 "legacy_validation_rows":value[3]["splits"]["validation"]["count"],
                 "history_validation_rows":len(value[2]),"token_lengths":value[4],"training_ready":False}
            for task,value in plans.items()}


def prepare_continual_pool(pool, output, tokenizer=None):
    """Select new notifications plus previously approved replay; never train."""
    from filtering_training.common.digit_scores import replace_prompt, prompt_digest as digit_digest
    from filtering_training.modeling.digit_score_experiment import load_validation_for_evaluation, VALIDATION_POOL
    if output.exists():
        raise ValueError("preserve prepared versions")
    pool_manifest=json.loads((pool/"manifest.json").read_text(encoding="utf-8"))
    revision=pool_manifest["label_revision"]
    if digest(pool/revision["dataset_file"])!=revision["dataset_sha256"] or digest(pool/"approval.json")!=revision["approval_sha256"]:
        raise ValueError("corrected training candidates changed")
    replay=[]
    sources=[]
    for name in ("v3_score_review_02_approved_01","v3_expansion_review_01_approved","v4_urgent_review_01_approved"):
        folder=pool.parent/name
        approved=json.loads((folder/"approval.json").read_text(encoding="utf-8"))
        source_info=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        if (digest(folder/"dataset.jsonl")!=approved["dataset_sha256"]
            or digest(folder/"approval.json")!=source_info["approval_sha256"]
            or source_info["source_v3_sha256"]!=digest(SOURCE/"dataset.jsonl")):
            raise ValueError("approved replay source changed")
        replay+=load_score_samples(folder/"dataset.jsonl")
        sources.append({"folder":str(folder),"dataset_sha256":digest(folder/"dataset.jsonl"),"approval_sha256":digest(folder/"approval.json")})
    def fingerprint(notification):
        value=dict(notification)
        value.pop("id",None)
        return json.dumps(value,ensure_ascii=False,sort_keys=True)
    replay_ids={s.notification.id for s in replay}
    replay_notifications={fingerprint(s.notification.model_dump(mode="json")) for s in unique_urgency_samples(replay)}
    heldout,_,_=load_validation_for_evaluation(VALIDATION_POOL)
    val_bodies={re.sub(r"\s+","",s.notification.body).casefold() for s in heldout}
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",local_files_only=True)
    plans={}
    for task in ("urgency","relevance"):
        folder=pool/"prepared"/task
        old_manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        if old_manifest["pool"]["dataset_sha256"]!=revision["dataset_sha256"]:
            raise ValueError("draft does not use current corrected candidates")
        for split in ("train","validation"):
            if digest(folder/f"{split}.jsonl")!=old_manifest["splits"][split]["file_sha256"]:
                raise ValueError("draft training or validation changed")
        all_records=[json.loads(line) for line in (folder/"train.jsonl").read_text(encoding="utf-8").splitlines()]
        selected=[]
        for id_,record in zip(old_manifest["splits"]["train"]["notification_ids"],all_records,strict=True):
            payload=json.loads(record["messages"][1]["content"])
            is_new=id_.startswith("train05_")
            is_replay=id_ in replay_ids if task=="relevance" else fingerprint(payload["notification"]) in replay_notifications
            if is_new or is_replay:
                if re.sub(r"\s+","",payload["notification"]["body"]).casefold() in val_bodies:
                    raise ValueError("selected training overlaps validation")
                selected.append((id_,record,is_new))
        if task=="urgency":
            # Earlier digital-event capping omitted one approved original.
            # Restore it from approval evidence, never from held-out data.
            selected_notifications={fingerprint(json.loads(record["messages"][1]["content"])["notification"])
                                    for _,record,_ in selected}
            for sample in unique_urgency_samples(replay):
                if fingerprint(sample.notification.model_dump(mode="json")) not in selected_notifications:
                    selected.append((sample.notification.id,sft_record(sample,task),False))
        expected_new=4910 if task=="urgency" else pool_manifest["new_rows"]
        expected_replay=len(replay_notifications) if task=="urgency" else len(replay_ids)
        if sum(x[2] for x in selected)!=expected_new or sum(not x[2] for x in selected)!=expected_replay:
            raise ValueError("new or approved replay rows missing")
        random.Random(42).shuffle(selected)
        lengths=[]
        for _,record,_ in selected:
            effective=replace_prompt(record["messages"],task)
            payload=json.loads(effective[1]["content"])
            if re.sub(r"\s+","",payload["notification"]["body"]).casefold() in val_bodies:
                raise ValueError("selected training overlaps validation")
            if "id" in payload["notification"] or (task=="urgency" and "context" in payload):
                raise ValueError("effective input leaks context or id")
            lengths.append(len(tokenizer.apply_chat_template(effective,add_generation_prompt=True,
                               enable_thinking=False,tokenize=True,return_dict=True)["input_ids"]))
        if max(lengths)>768:
            raise ValueError("prepared prompt exceeds training token limit")
        plans[task]=(selected,old_manifest,{"max":max(lengths),"count":len(lengths),"limit":768},expected_new,expected_replay)
    for task,(selected,previous,lengths,new_count,replay_count) in plans.items():
        folder=output/task; folder.mkdir(parents=True)
        write_lines(folder/"train.jsonl",[record for _,record,_ in selected])
        (folder/"validation.jsonl").write_bytes((pool/"prepared"/task/"validation.jsonl").read_bytes())
        current=deepcopy(previous)
        current.update(purpose="new notifications plus previously human-approved replay; adapter continuation prepared only",
                       created_date="2026-10-04",training_ready=True,training_used=False)
        current["splits"].pop("validation_history",None)
        current["splits"]["train"]={"count":len(selected),"notification_ids":[id_ for id_,_,_ in selected],"file_sha256":digest(folder/"train.jsonl")}
        current["sampling"]={"required_mode":"natural","method":"all new rows and approved replay once each; no duplicate oversampling","seed":42}
        current["continuation_data"]={"new_rows":new_count,"replay_rows":replay_count,"replay_sources":sources,
              "source_draft_manifest_sha256":digest(pool/"prepared"/task/"manifest.json"),"review_scope":"new bulk remains synthetic; individual approval limited to displayed cases"}
        current["token_lengths"]=lengths
        current["training_score_counts"]=dict(Counter(json.loads(record["messages"][-1]["content"])[task+"_score"] for _,record,_ in selected))
        current["digit_prompt_hashes"]={"legacy":digit_digest(task),"history":digit_digest(task,history=True)}
        current["quality_gate"]={"pending":[],"training_ready":True,"note":"readiness does not execute or schedule training"}
        current["validation_pool"]={"folder":str(VALIDATION_POOL),"dataset_sha256":digest(VALIDATION_POOL/"dataset.jsonl"),"combined_rows":len(heldout)}
        write_json(folder/"manifest.json",current)
    return {task:{"train_rows":len(value[0]),"new_rows":value[3],"replay_rows":value[4],"max_input_tokens":value[2]["max"]} for task,value in plans.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--validation-pool",type=Path)
    parser.add_argument("--continual",action="store_true",help="prepare new data plus approved replay; never train")
    args = parser.parse_args()
    if args.pool:
        if args.output is None: parser.error("--pool requires --output")
        result=prepare_continual_pool(args.pool,args.output) if args.continual else prepare_history_pool(args.pool,args.validation_pool,args.output) if args.validation_pool else prepare_pool(args.pool,args.output)
        print(json.dumps(result,ensure_ascii=False))
    else:
        print(prepare(args.output or OUTPUT))
