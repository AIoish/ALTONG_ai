"""용도: 기존 분리 기준선과 세 대조의 원시 출력·분할·학습 변경을 검증해 비교한다.
생성일: 2026-10-04
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from filtering_training.common.score_tasks import parse_scores, prompt_digest, score_metrics
from filtering_training.modeling.evaluate import load_split_samples
from filtering_training.preparation.prepare_score_experiment import digest, write_json
from src.filtering.policy import POLICY_VERSION


def compare(source, prepared, baseline, controls, output):
    if output.exists():
        raise ValueError("preserve the earlier comparison")
    samples=load_split_samples(source/"dataset.jsonl",source/"prepared","validation")
    ids=[s.notification.id for s in samples]
    groups={"original":baseline,**{mode:controls/mode for mode in ("standard_nll","numeric_weight","balanced")}}
    result={"purpose":"fixed-data, bounded-step controls; not final model selection",
            "created_date":"2026-10-04","source_sha256":digest(source/"dataset.jsonl"),"variants":{}}
    for name,folder in groups.items():
        report=json.loads((folder/"C.json").read_text(encoding="utf-8"))
        if (report["split"]!="validation" or report["sampling"] is not None
                or report["dataset_sha256"]!=digest(source/"dataset.jsonl")
                or report["source_manifest_sha256"]!=digest(source/"prepared/manifest.json")
                or report["evaluated_ids"]!=ids or report["policy_version"]!=POLICY_VERSION
                or report["prompt_hashes"]!={task:prompt_digest(task) for task in ("urgency","relevance")}):
            raise ValueError("report source, policy, prompt or coverage mismatch")
        records_path=folder/"C.predictions.jsonl"
        if report["predictions_sha256"]!=digest(records_path):
            raise ValueError("prediction records changed")
        rows=[json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
        if [row["notification_id"] for row in rows]!=ids:
            raise ValueError("incomplete or duplicate predictions")
        predictions=[]
        for sample,row in zip(samples,rows):
            if (row["notification"]!=sample.notification.model_dump(mode="json")
                    or row["context"]!=sample.context.model_dump(mode="json")
                    or row["gold"]!=sample.label.model_dump()):
                raise ValueError("input or gold changed")
            parsed,invalid={},False
            if set(row["responses"])!={"urgency","relevance"}:
                raise ValueError("missing task response")
            for task,response in row["responses"].items():
                try:
                    value=parse_scores(response["raw"],task)
                except ValueError:
                    if not response["error"]:
                        raise ValueError("invalid raw response accepted")
                    invalid=True
                else:
                    if response["error"]:
                        if response["error"]!="generation reached token limit without EOS":
                            raise ValueError("unexplained response error")
                        invalid=True
                    else:
                        parsed.update(value)
            predicted=None if invalid else parsed
            if row["prediction"]!=predicted:
                raise ValueError("prediction differs from raw responses")
            predictions.append(predicted)
        metrics=score_metrics(samples,predictions)
        if metrics!=report["metrics"]:
            raise ValueError("reported metrics differ from recomputation")
        training={}
        for task in ("urgency","relevance"):
            config_path=folder/task/"run_config.json"
            config=json.loads(config_path.read_text(encoding="utf-8"))
            m=json.loads((prepared/task/"manifest.json").read_text(encoding="utf-8"))
            if (report["run_config_hashes"][task]!=digest(config_path)
                    or config["prepared_manifest_sha256"]!=digest(prepared/task/"manifest.json")
                    or config["dataset_sha256"]!=digest(source/"dataset.jsonl")
                    or config["source_manifest_sha256"]!=digest(source/"prepared/manifest.json")
                    or config["prompt_sha256"]!=prompt_digest(task)
                    or config["model"]!="Qwen/Qwen3-0.6B" or config["score_task"]!=task
                    or config["seed"]!=42 or config["max_length"]!=768
                    or config["max_steps"]!=m["splits"]["train"]["count"]
                    or config["learning_rate"]!=0.0002
                    or config["lora"]!={"r":8,"alpha":16,"dropout":0.05,"target_modules":["q_proj","v_proj"]}):
                raise ValueError("training controls differ unexpectedly")
            if name!="original":
                raw_records=[json.loads(line) for line in (prepared/task/"train.jsonl").read_text(encoding="utf-8").splitlines()]
                selected=config["selected_prepared_indices"]
                if any(type(i) is not int or not 0 <= i < len(raw_records) for i in selected):
                    raise ValueError("sampler references an unknown training row")
                values=[parse_scores(raw_records[i]["messages"][-1]["content"],task)[task+"_score"] for i in selected]
                counts={str(k):v for k,v in Counter(values).items()}
                if config["score_control"]!=name or counts!=config["selected_score_counts"] or len(selected)!=config["max_steps"]:
                    raise ValueError("sampler provenance differs")
                if name!="balanced" and selected!=list(range(len(raw_records))):
                    raise ValueError("data order changed in loss-only control")
                if config["numeric_weight"]!=(7 if name=="numeric_weight" else 1):
                    raise ValueError("unexpected numeric weighting")
            training[task]={"steps":config["max_steps"],"loss":config["training_loss"],
                "loss_engine":config.get("loss_engine","chunked_nll"),
                "numeric_weight":config.get("numeric_weight",1),
                "selected_score_counts":config.get("selected_score_counts"),
                "selected_unique_records":config.get("selected_unique_records"),
                "runtime":config["training_metrics"]["train_runtime"]}
        diagnosis_path=folder/"learning_diagnosis.json"
        diagnosis=json.loads(diagnosis_path.read_text(encoding="utf-8"))
        probes={}
        for task,data in diagnosis["tasks"].items():
            train_rows=[r for r in data["rows"] if r["split"]=="train"]
            probes[task]={"selected_train_count":len(train_rows),
                "selected_train_exact":sum(r["gold"]==r["free_prediction"] for r in train_rows),
                "selected_train_predictions":dict(Counter(r["free_prediction"] for r in train_rows)),
                "numeric_nll_mean":sum(r["numeric_nll"] for r in train_rows)/len(train_rows),
                "standalone_matches_multi_adapter":data["validation_matches_multi_adapter"]}
        result["variants"][name]={"report_sha256":digest(folder/"C.json"),"metrics":metrics,
            "performance":report["performance"],"training":training,"training_probes":probes}
    write_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("filtering_training/outputs/v3_reviewed_01"))
    parser.add_argument("--prepared-root",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--controls",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.source,args.prepared_root,args.baseline,args.controls,args.output)
    for name,r in result["variants"].items():
        print(name,r["metrics"]["policy_correct"],"/24 urgent failures",r["metrics"]["urgent_failures"],
              "false passes",r["metrics"]["false_passes"])


if __name__=="__main__":
    main()
