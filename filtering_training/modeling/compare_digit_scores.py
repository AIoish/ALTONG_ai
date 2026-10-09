"""용도: 숫자 분류 실험의 분할·샘플링·확률·정책 지표를 재검증해 비교한다.
생성일: 2026-10-04
"""

from collections import Counter
import argparse
import json
from pathlib import Path

from filtering_training.common.digit_scores import prompt_digest
from filtering_training.common.score_loss import balanced_indices
from filtering_training.common.score_tasks import parse_scores, score_metrics
from filtering_training.modeling.digit_score_experiment import SOURCE, PREPARED
from filtering_training.modeling.evaluate import load_split_samples
from filtering_training.preparation.prepare_digit_supplement import APPROVED, OUTPUT
from src.filtering.policy import should_pass
from filtering_training.preparation.prepare_score_experiment import digest, write_json


def save_comparison(path, check_only, value):
    if check_only:
        existing = json.loads(path.read_text(encoding="utf-8"))
        assert existing == json.loads(json.dumps(value)), "stored comparison differs"
    else:
        write_json(path,value)


def verify_report(folder, samples):
    report = json.loads((folder/"validation.json").read_text(encoding="utf-8"))
    path = folder/"validation.predictions.jsonl"
    assert report["dataset_sha256"] == digest(SOURCE/"dataset.jsonl")
    assert report["split"] == "validation" and report["predictions_sha256"] == digest(path)
    rows = [json.loads(v) for v in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(samples) == 24
    for sample,row in zip(samples,rows):
        assert row["notification_id"] == sample.notification.id
        assert row["notification"] == sample.notification.model_dump(mode="json")
        assert row["context"] == sample.context.model_dump(mode="json")
        assert row["gold"] == sample.label.model_dump()
        for task in ("urgency","relevance"):
            probs = row["conditional_probabilities"][task]
            assert len(probs)==5 and all(0<=p<=1 for p in probs) and abs(sum(probs)-1)<1e-5
            assert row["prediction"][task+"_score"] == max(range(5),key=probs.__getitem__)+1
    metrics = score_metrics(samples,[r["prediction"] for r in rows])
    assert metrics == report["metrics"] and metrics["fully_invariant_groups"] == 8
    return report,rows,metrics


def compare_digit(root, check_only=False):
    if not check_only and (root/"comparison.json").exists():
        raise ValueError("preserve previous comparison")
    samples = load_split_samples(SOURCE/"dataset.jsonl",SOURCE/"prepared","validation")
    results = {}
    for mode in ("natural","balanced"):
        folder = root/mode
        report, rows, metrics = verify_report(folder, samples)
        configs = {}
        for task,count in (("urgency",50),("relevance",150)):
            path = folder/task/"run_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            assert report["run_config_hashes"][task] == digest(path)
            assert config["dataset_sha256"] == digest(SOURCE/"dataset.jsonl")
            assert config["source_manifest_sha256"] == digest(SOURCE/"prepared/manifest.json")
            assert config["prepared_manifest_sha256"] == digest(PREPARED/task/"manifest.json")
            assert config["prompt_sha256"] == prompt_digest(task)
            assert config["task"] == task and config["mode"] == mode and config["max_steps"] == count
            records = [json.loads(v) for v in (PREPARED/task/"train.jsonl").read_text(encoding="utf-8").splitlines()]
            scores = [parse_scores(r["messages"][-1]["content"],task)[task+"_score"] for r in records]
            indices = balanced_indices(scores) if mode == "balanced" else list(range(count))
            assert config["selected_prepared_indices"] == indices
            assert config["selected_score_counts"] == {str(k):v for k,v in Counter(scores[i] for i in indices).items()}
            configs[task] = {"training_loss":config["training_loss"],"training_seconds":config["training_metrics"]["train_runtime"],
                             "selected_unique_records":config["selected_unique_records"]}
        results[mode] = {"metrics":metrics,"mean_seconds":report["mean_seconds"],"training":configs,
            "prediction_counts":{task:dict(Counter(r["prediction"][task+"_score"] for r in rows)) for task in ("urgency","relevance")}}
    save_comparison(root/"comparison.json", check_only,{"purpose":"verified fixed-data conditional digit comparison", "created_date":"2026-10-04",
        "verification":"passed","test_used":False,"supplement_used":False,"results":results})
    print(json.dumps(results,ensure_ascii=False,indent=2))



def compare_supplement(root, check_only=False):
    if not check_only and (root/"comparison.json").exists():
        raise ValueError("preserve previous comparison")
    samples = load_split_samples(SOURCE/"dataset.jsonl",SOURCE/"prepared","validation")
    results, predictions = {}, {}
    settings = (("previous",Path("filtering_training/outputs/runs/2026-10-04-qwen06-digit-01/natural"),PREPARED,(50,150)),
                ("matched_steps",root/"matched_steps",PREPARED,(58,174)),("supplement",root/"supplement",OUTPUT,(58,174)))
    for name,folder,prepared,steps in settings:
        report, rows, metrics = verify_report(folder, samples)
        training = {}
        for task,max_steps in zip(("urgency","relevance"),steps):
            path = folder/task/"run_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            manifest = json.loads((prepared/task/"manifest.json").read_text(encoding="utf-8"))
            assert report["run_config_hashes"][task] == digest(path)
            assert config["prompt_sha256"] == prompt_digest(task)
            assert config["dataset_sha256"] == digest(SOURCE/"dataset.jsonl")
            assert config["source_manifest_sha256"] == digest(SOURCE/"prepared/manifest.json")
            assert config["prepared_manifest_sha256"] == digest(prepared/task/"manifest.json")
            assert config["mode"] == "natural" and config["task"] == task and config["max_steps"] == max_steps
            assert config["log_history"][-1]["step"] == max_steps
            assert config["selected_prepared_indices"] == list(range(manifest["splits"]["train"]["count"]))
            assert digest(prepared/task/"validation.jsonl") == digest(PREPARED/task/"validation.jsonl")
            assert digest(prepared/task/"train.jsonl") == manifest["splits"]["train"]["file_sha256"]
            if name == "supplement":
                supplement = config["supplement"]
                assert supplement == manifest["supplement"]
                assert supplement["dataset_sha256"] == digest(APPROVED/"dataset.jsonl")
                assert supplement["approval_sha256"] == digest(APPROVED/"approval.json")
                assert supplement["guideline_override_ids"] == ["sep02_07_related"]
                old = (PREPARED/task/"train.jsonl").read_text(encoding="utf-8").splitlines()
                new = (prepared/task/"train.jsonl").read_text(encoding="utf-8").splitlines()
                assert [json.loads(v) for v in old] == [json.loads(v) for v in new[:len(old)]]
            training[task] = {"max_steps":max_steps,"records":manifest["splits"]["train"]["count"],
                "loss":config["training_loss"],"seconds":config["training_metrics"]["train_runtime"]}
        predictions[name] = rows
        results[name] = {"metrics":metrics,"mean_seconds":report["mean_seconds"],"training":training,
            "prediction_counts":{t:dict(Counter(r["prediction"][t+"_score"] for r in rows)) for t in ("urgency","relevance")}}
    changes = []
    for sample,before,after in zip(samples,predictions["matched_steps"],predictions["supplement"]):
        if before["prediction"] != after["prediction"]:
            changes.append({"notification_id":sample.notification.id,"gold":sample.label.model_dump(),
                "before":before["prediction"],"after":after["prediction"],
                "gold_pass":should_pass(sample.label.urgency_score,sample.label.relevance_score)})
    save_comparison(root/"comparison.json", check_only,{"purpose":"fixed-validation approved supplement effect with equal-step control",
        "created_date":"2026-10-04","verification":"passed","test_used":False,"results":results,"changed_predictions":changes})
    print(json.dumps(results,ensure_ascii=False,indent=2))


def compare_threshold(root, check_only=False, baseline=None, experiment="threshold"):
    if not check_only and (root/"comparison.json").exists():
        raise ValueError("preserve previous comparison")
    baseline = baseline or Path("filtering_training/outputs/runs/2026-10-04-qwen06-digit-supplement-01/supplement")
    samples = load_split_samples(SOURCE/"dataset.jsonl",SOURCE/"prepared","validation")
    results = {}
    configs = {}
    for name,folder in (("baseline",baseline),(experiment,root)):
        report, rows, metrics = verify_report(folder, samples)
        for task,run_dir in (("urgency",folder/"urgency"),("relevance",baseline/"relevance")):
            assert report["run_config_hashes"][task] == digest(run_dir/"run_config.json")
            config = json.loads((run_dir/"run_config.json").read_text(encoding="utf-8"))
            assert config["prepared_manifest_sha256"] == digest(OUTPUT/task/"manifest.json")
            assert config["dataset_sha256"] == digest(SOURCE/"dataset.jsonl")
            assert config["source_manifest_sha256"] == digest(SOURCE/"prepared/manifest.json")
            assert config["prompt_sha256"] == prompt_digest(task)
            if task == "urgency":
                configs[name] = config
                expected_steps = 174 if experiment=="epochs3" and name==experiment else 58
                assert config["max_steps"] == config["log_history"][-1]["step"] == expected_steps
                if experiment=="epochs3":
                    assert config["log_history"][-1]["epoch"] == (3 if name==experiment else 1)
        fit_path = folder/"train_fit_urgency.json"
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        fit_pred = fit_path.with_suffix(".predictions.jsonl")
        assert fit["run_config_sha256"] == digest(folder/"urgency/run_config.json")
        assert fit["train_sha256"] == digest(OUTPUT/"urgency/train.jsonl")
        assert fit["predictions_sha256"] == digest(fit_pred)
        fit_rows = [json.loads(v) for v in fit_pred.read_text(encoding="utf-8").splitlines()]
        for group,selected in (("all_train",fit_rows),("approved_supplement_train",[r for r in fit_rows if r["notification_id"].startswith("sep02_")])):
            assert fit[group]["count"] == len(selected)
            assert fit[group]["exact_correct"] == sum(r["gold"]==r["prediction"] for r in selected)
            assert fit[group]["urgent_false_blocks"] == sum(r["gold"]>=4 and r["prediction"]<4 for r in selected)
        results[name] = {"metrics":metrics,"mean_seconds":report["mean_seconds"],"train_fit":fit["all_train"],
            "supplement_train_fit":fit["approved_supplement_train"],
            "urgency_prediction_counts":dict(Counter(r["prediction"]["urgency_score"] for r in rows)),
            "training_loss":configs[name]["training_loss"],"training_seconds":configs[name]["training_metrics"]["train_runtime"]}
    stable = ("model","task","mode","objective","dataset_sha256","source_manifest_sha256","prepared_manifest_sha256",
              "prompt_sha256","digit_token_ids","selected_prepared_indices","original_score_counts","selected_score_counts",
              "max_steps","seed","learning_rate","max_length","batch_size","lora","precision","quantization","supplement")
    for key in stable:
        if key=="max_steps" and experiment=="epochs3":
            continue
        assert configs["baseline"][key] == configs[experiment][key], key
    assert configs["baseline"].get("threshold_weight",0) == 0
    assert configs[experiment]["threshold_weight"] == (0 if experiment=="epochs3" else 1)
    purpose = "urgency three versus one epoch with fixed relevance adapter" if experiment=="epochs3" else "urgency threshold auxiliary loss with fixed relevance adapter and steps"
    save_comparison(root/"comparison.json", check_only,{"purpose":purpose,
        "created_date":"2026-10-04","verification":"passed","test_used":False,"results":results})
    print(json.dumps(results,ensure_ascii=False,indent=2))



def compare_continual(root, experiment="continual"):
    """Verify same approved validation before/after and append the existing log."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from filtering_training.modeling.digit_score_experiment import load_validation_for_evaluation, VALIDATION_POOL
    if (root/"comparison.json").exists():
        raise ValueError("preserve previous comparison")
    samples,_,_=load_validation_for_evaluation(VALIDATION_POOL)
    reports={}; predictions={}; source_reports={}
    for name in ("before","after"):
        report=json.loads((root/f"{name}.json").read_text(encoding="utf-8"))
        path=root/f"{name}.predictions.jsonl"
        if report["predictions_sha256"]!=digest(path):
            raise ValueError("validation predictions changed")
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows)!=len(samples):
            raise ValueError("validation counts differ")
        for sample,row in zip(samples,rows,strict=True):
            if (row["notification_id"]!=sample.notification.id or row["notification"]!=sample.notification.model_dump(mode="json")
                or row["context"]!=sample.context.model_dump(mode="json") or row["gold"]!=sample.label.model_dump()):
                raise ValueError("validation input or label differs")
            for task in ("urgency","relevance"):
                probs=row["conditional_probabilities"][task]
                if len(probs)!=5 or any(not 0<=p<=1 for p in probs) or abs(sum(probs)-1)>1e-5 or row["prediction"][task+"_score"]!=max(range(5),key=probs.__getitem__)+1:
                    raise ValueError("score differs from digit probabilities")
        calculated=score_metrics(samples,[row["prediction"] for row in rows])
        if calculated!=report["metrics"]:
            raise ValueError("validation metrics differ")
        for task,run in report["task_runs"].items():
            if digest(Path(run)/"run_config.json")!=report["run_config_hashes"][task]:
                raise ValueError("evaluated adapter configuration changed")
        reports[name]=calculated; predictions[name]=rows; source_reports[name]=report
    methods={}
    for task in ("urgency","relevance"):
        before_config=json.loads((Path(source_reports["before"]["task_runs"][task])/"run_config.json").read_text(encoding="utf-8"))
        after_config=json.loads((root/task/"run_config.json").read_text(encoding="utf-8"))
        if not after_config.get("training_metrics") or not after_config.get("continuation_parent"):
            raise ValueError("completed continuation training record required")
        if experiment=="accumulated":
            for key in ("prepared_root","prepared_manifest_sha256","selected_prepared_indices","selected_notification_ids",
                        "prompt_sha256","digit_prompt_hashes","lora","learning_rate","max_length","precision","quantization"):
                if before_config[key]!=after_config[key]:
                    raise ValueError("fixed-data method comparison differs: "+key)
            schedule=after_config["training_schedule"]
            if (schedule["gradient_accumulation_steps"]!=8 or schedule["num_train_epochs"]!=1
                or schedule["warmup_ratio"]!=.05 or after_config["actual_training_examples"]!=after_config["selected_unique_records"]
                or after_config["actual_optimizer_steps"]!=schedule["expected_optimizer_steps"]):
                raise ValueError("accumulated run did not match agreed training schedule")
            if after_config["continuation_parent"]["run_config_sha256"]!=source_reports["before"]["run_config_hashes"][task]:
                raise ValueError("new weights did not continue the evaluated baseline")
        methods[task]={name:{"records":config["selected_unique_records"],"optimizer_steps":config["max_steps"],
                            "gradient_accumulation_steps":config.get("training_schedule",{}).get("gradient_accumulation_steps",1),
                            "warmup_ratio":config.get("training_schedule",{}).get("warmup_ratio",0),
                            "training_loss":config["training_loss"],"training_seconds":config["training_metrics"]["train_runtime"]}
                       for name,config in (("before",before_config),("after",after_config))}
    created_date=datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    write_json(root/"comparison.json",{"purpose":"same approved validation before and after adapter continuation",
               "created_date":created_date,"experiment":experiment,"results":reports,"training_methods":methods,
               "metrics_by_set":{name:source_reports[name].get("metrics_by_set",{}) for name in ("before","after")},
               "validation_rows":len(samples),"test_used":False,
               "interpretation":"before/after continuation also adds one data exposure; not an isolated causal comparison of training settings"})
    before,after=reports["before"],reports["after"]
    heading="누적 배치 8·warmup 학습과 직전 결과 비교" if experiment=="accumulated" else "어댑터 누적 학습·검증 비교"
    lines=[f"\n\n## {created_date} — {heading}\n",
           "| 항목 | 학습 전 | 학습 후 |","| --- | --- | --- |",
           f"| PASS/BLOCK 정답 | {before['policy_correct']}/{len(samples)} | {after['policy_correct']}/{len(samples)} |",
           f"| 긴급 알림 차단 | {before['urgent_false_blocks']}/{before['urgent_count']} | {after['urgent_false_blocks']}/{after['urgent_count']} |",
           f"| 불필요한 통과 | {before['false_passes']} | {after['false_passes']} |"]
    if experiment=="accumulated":
        sets={name:source_reports[name]["metrics_by_set"] for name in ("before","after")}
        lines += [f"| 기존 / 추가 검증 정답 | {sets['before']['legacy_24']['policy_correct']}/24 / {sets['before']['approved_expansion']['policy_correct']}/50 | {sets['after']['legacy_24']['policy_correct']}/24 / {sets['after']['approved_expansion']['policy_correct']}/50 |"]
    if experiment=="accumulated":
        lines += [f"| 긴급도 / 관련도 점수 정답 | {round(before['urgency_score']['exact_accuracy']*len(samples))}/{len(samples)} / {round(before['relevance_score']['exact_accuracy']*len(samples))}/{len(samples)} | {round(after['urgency_score']['exact_accuracy']*len(samples))}/{len(samples)} / {round(after['relevance_score']['exact_accuracy']*len(samples))}/{len(samples)} |"]
        lines += ["\n| 학습 방식 | 직전 학습 | 이번 학습 |", "| --- | --- | --- |",
                  "| 배치 / gradient accumulation | 1 / 1 | 1 / 8 |",
                  "| 학습량 | 고정 step으로 전체 1회 | 실제 1 epoch, 샘플 처리 수 검증 |",
                  "| Warmup | 없음 | optimizer step의 5% |",
                  "| Loss 평균 | 샘플별 업데이트 | microbatch 평균을 누적, 마지막 부분 묶음 포함 |",
                  "| 유지 | 학습률 5e-5, linear, LoRA r8·q/v | 동일, 데이터·라벨·프롬프트도 동일 |"]
    for task in ("urgency","relevance"):
        config=json.loads((root/task/"run_config.json").read_text(encoding="utf-8"))
        if not config.get("training_metrics") or not config.get("continuation_parent"):
            raise ValueError("completed continuation training record required")
        title="긴급도" if task=="urgency" else "관련도"
        previous=methods[task]["before"]
        lines += [f"| {title} 학습 건수·손실·시간 | {previous['records']}건 / {previous['training_loss']:.4f} / {previous['training_seconds']/60:.1f}분 | {config['selected_unique_records']}건 / "
                  f"{config['training_loss']:.4f} / {config['training_metrics']['train_runtime']/60:.1f}분 |"]
        if experiment=="accumulated":
            lines += [f"| {title} optimizer 업데이트 | {previous['optimizer_steps']} | {config['actual_optimizer_steps']} |"]
    changed=[i for i in range(len(samples)) if predictions['before'][i]['prediction']!=predictions['after'][i]['prediction']]
    urgent_changed=[i for i in changed if samples[i].label.urgency_score>=4]
    history_changed=[i for i in changed if getattr(samples[i].context,"recent_windows",[])]
    remaining_errors=[i for i,sample in enumerate(samples)
                      if should_pass(sample.label.urgency_score,sample.label.relevance_score)!=should_pass(**predictions['after'][i]['prediction'])]
    indices=[]; seen_bodies=set()
    for candidates in (urgent_changed,history_changed,remaining_errors,changed,list(range(len(samples)))):
        for index in candidates:
            body=samples[index].notification.body
            if body not in seen_bodies:
                indices.append(index);seen_bodies.add(body)
                break
        if len(indices)==3:
            break
    for index,sample in enumerate(samples):
        if len(indices)==3:
            break
        if sample.notification.body not in seen_bodies:
            indices.append(index)
            seen_bodies.add(sample.notification.body)
    for number,index in enumerate(indices,1):
        sample=samples[index]
        history=getattr(sample.context,"recent_windows",[])
        lines += [f"\n**대표 입출력 {number}** ({sample.notification.id})",
                  f"- 알림: {sample.notification.body}",f"- 현재 창: {sample.context.window_title or '(빈 창)'}",
                  "- 최근 창: "+(" / ".join(f"{w.app_name}: {w.window_title}" for w in history) or "없음"),
                  f"- 정답 긴급도/관련도: {sample.label.urgency_score}/{sample.label.relevance_score}; "
                  f"학습 전: {predictions['before'][index]['prediction']}; 학습 후: {predictions['after'][index]['prediction']}"]
    lines += [f"\n- 결과: `{root}`. 동일한 검증 {len(samples)}행 사용, 최종 테스트 미사용."]
    if experiment=="accumulated":
        lines += ["- 직전 가중치에 같은 데이터를 한 번 더 학습한 결과이며 추가 학습 효과도 포함. 설정 변경만의 효과로 단정하지 않음. 새 2천 원문 후보와 현재 창 우선 프롬프트는 이번 학습에 넣지 않음."]
    with Path("docs/filtering-experiment-log.md").open("a",encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",choices=("digit","supplement","threshold","epochs3","continual","accumulated"),default="digit")
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--baseline",type=Path)
    parser.add_argument("--check-only",action="store_true")
    args = parser.parse_args()
    if args.experiment in ("continual","accumulated"):
        if args.check_only: parser.error("continual comparison cannot rewrite an existing result")
        print(json.dumps(compare_continual(args.runs,args.experiment),ensure_ascii=False))
    elif args.experiment in ("threshold","epochs3"):
        compare_threshold(args.runs,args.check_only,args.baseline,args.experiment)
    elif args.experiment == "supplement":
        compare_supplement(args.runs,args.check_only)
    else:
        compare_digit(args.runs,args.check_only)


if __name__ == "__main__":
    main()
