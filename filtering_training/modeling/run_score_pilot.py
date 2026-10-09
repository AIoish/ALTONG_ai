"""용도: 필터링 점수 실험의 학습·검증·비교를 공통 실행기로 순차 실행한다.
생성일: 2026-10-03
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from filtering_training.common.score_tasks import TASKS
from filtering_training.modeling.digit_score_experiment import PREPARED, preflight_training
from filtering_training.preparation.prepare_digit_supplement import OUTPUT, prepare

PREFIX = "filtering_training."
BASELINE = Path("filtering_training/outputs/runs/2026-10-04-qwen06-digit-supplement-01/supplement")


def build_plan(experiment, prepared, output, baseline=BASELINE):
    """Return commands without creating files or loading models."""
    plan = []
    def add(log, module, arguments):
        plan.append((output/log, PREFIX+module, list(map(str, arguments))))
    if experiment == "pilot":
        for task in TASKS:
            add(f"train_{task}.log", "modeling.train_score_task",
                ["--prepared-dir",prepared/task,"--run-dir",output/task])
        for variant in ("A","B","C"):
            add(f"evaluate_{variant}.log", "modeling.evaluate_score_experiment",
                ["--prepared-root",prepared,"--runs",output,"--variant",variant,"--output",output/f"{variant}.json"])
        add("comparison.log","modeling.compare_score_experiment",
            ["--reports",output/"A.json",output/"B.json",output/"C.json","--output",output/"comparison.json"])
    elif experiment == "controls":
        for mode in ("standard_nll","numeric_weight","balanced"):
            folder = output/mode
            for task in ("urgency","relevance"):
                add(f"{mode}/train_{task}.log","modeling.train_score_control",
                    ["--prepared-dir",prepared/task,"--run-dir",folder/task,"--mode",mode])
            add(f"{mode}/evaluate.log","modeling.evaluate_score_experiment",
                ["--prepared-root",prepared,"--runs",folder,"--variant","C","--output",folder/"C.json"])
            add(f"{mode}/diagnosis.log","quality.audit_score_learning",
                ["--prepared-root",prepared,"--runs",folder,"--output",folder/"learning_diagnosis.json"])
    elif experiment in ("digit","supplement"):
        settings = (("natural",prepared),("balanced",prepared)) if experiment == "digit" else (("matched_steps",prepared),("supplement",OUTPUT))
        for mode,task_prepared in settings:
            folder = output/mode
            for task,steps in (("urgency",58),("relevance",174)):
                arguments = ["train","--task",task,"--mode",mode if experiment=="digit" else "natural",
                    "--prepared-root",task_prepared,"--output",folder/task]
                if experiment == "supplement":
                    arguments += ["--max-steps",steps]
                add(f"{mode}/train_{task}.log","modeling.digit_score_experiment",arguments)
            add(f"{mode}/evaluate.log","modeling.digit_score_experiment",
                ["evaluate","--runs",folder,"--prepared-root",task_prepared,"--output",folder/"validation.json"])
    elif experiment == "fit":
        add("train_urgency.log","modeling.digit_score_experiment",
            ["train","--task","urgency","--mode","natural","--max-steps",60,"--train-indices","7,4,0",
             "--prepared-root",OUTPUT,"--output",output/"urgency"])
        add("train_fit.log","quality.audit_digit_training",["--runs",output,"--task","urgency"])
    elif experiment == "pool":
        for task in ("urgency","relevance"):
            add(f"train_{task}.log","modeling.digit_score_experiment",
                ["train","--task",task,"--mode","natural","--prepared-root",prepared,"--output",output/task])
        add("evaluate.log","modeling.digit_score_experiment",
            ["evaluate","--runs",output,"--prepared-root",prepared,"--output",output/"validation.json"])
    elif experiment == "continual":
        add("evaluate_before.log","modeling.digit_score_experiment",
            ["evaluate","--runs",baseline,"--relevance-run",baseline/"relevance_id_clean",
             "--prepared-root",Path("filtering_training/outputs/v4_training_pool_01/prepared"),"--output",output/"before.json"])
        for task,parent in (("urgency","urgency"),("relevance","relevance_id_clean")):
            add(f"train_{task}.log","modeling.digit_score_experiment",
                ["train","--task",task,"--mode","natural","--prepared-root",prepared,
                 "--continue-from",baseline/parent,"--learning-rate",.00005,"--output",output/task])
        add("evaluate_after.log","modeling.digit_score_experiment",
            ["evaluate","--runs",output,"--prepared-root",prepared,"--output",output/"after.json"])
        add("comparison.log","modeling.compare_digit_scores",["--experiment","continual","--runs",output])
    elif experiment == "accumulated":
        add("evaluate_before.log","modeling.digit_score_experiment",
            ["evaluate","--runs",baseline,"--prepared-root",prepared,"--output",output/"before.json"])
        for task in ("urgency","relevance"):
            add(f"train_{task}.log","modeling.digit_score_experiment",
                ["train","--task",task,"--mode","natural","--prepared-root",prepared,
                 "--continue-from",baseline/task,"--learning-rate",.00005,
                 "--gradient-accumulation-steps",8,"--epochs",1,"--warmup",.05,"--output",output/task])
        add("evaluate_after.log","modeling.digit_score_experiment",
            ["evaluate","--runs",output,"--prepared-root",prepared,"--output",output/"after.json"])
        add("comparison.log","modeling.compare_digit_scores",["--experiment","accumulated","--runs",output])
    elif experiment in ("threshold","epochs3"):
        add("train_urgency.log","modeling.digit_score_experiment",
            ["train","--task","urgency","--mode","natural","--max-steps",174 if experiment=="epochs3" else 58,
             "--threshold-weight",0 if experiment=="epochs3" else 1,
             "--prepared-root",OUTPUT,"--output",output/"urgency"])
        add("evaluate.log","modeling.digit_score_experiment",
            ["evaluate","--runs",output,"--prepared-root",OUTPUT,"--relevance-run",baseline/"relevance",
             "--output",output/"validation.json"])
        add("train_fit.log","quality.audit_digit_training",["--runs",output,"--task","urgency"])
        if experiment == "epochs3":
            add("comparison.log","modeling.compare_digit_scores",
                ["--experiment","epochs3","--runs",output,"--baseline",baseline])
    else:
        raise ValueError("unknown experiment")
    return plan


def run(prepared, output, experiment="pilot", baseline=BASELINE):
    if output.exists():
        raise ValueError("use a fresh pilot directory")
    if experiment=="continual":
        for task,parent in (("urgency","urgency"),("relevance","relevance_id_clean")):
            preflight_training(output/task,task,prepared,baseline/parent,.00005)
    if experiment=="accumulated":
        for task in ("urgency","relevance"):
            plan=preflight_training(output/task,task,prepared,baseline/task,.00005,accumulation=8,epochs=1,warmup=.05)
            print("Preflight "+json.dumps(plan,ensure_ascii=False),flush=True)
    if experiment == "pool":
        for task in ("urgency","relevance"):
            manifest = json.loads((prepared/task/"manifest.json").read_text(encoding="utf-8"))
            if not manifest.get("training_ready") or not manifest.get("pool") or manifest.get("sampling",{}).get("required_mode") != "natural":
                raise ValueError("pool run requires fully checked large-pool preparation")
    if experiment == "supplement" and not OUTPUT.exists():
        prepare()
    if experiment in ("threshold","epochs3") and not (baseline/"relevance/run_config.json").is_file():
        raise ValueError("threshold requires an existing relevance baseline")
    plan = build_plan(experiment,prepared,output,baseline)
    output.mkdir(parents=True)
    (output/"README.md").write_text(
        "<!-- 용도: 0.6B 점수 실험의 가중치·학습 로그·검증 보고서.\n"
        f"생성일: {datetime.now(ZoneInfo('Asia/Seoul')).date().isoformat()} (Asia/Seoul) -->\n\n"
        "각 작업은 독립 프로세스로 순차 실행합니다. 로그는 UTF-8입니다.\n"
        f"실험 종류: {experiment}. 모델은 Qwen/Qwen3-0.6B이며 최종 테스트는 추론하지 않습니다.\n",
        encoding="utf-8")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1", "PYTHONUNBUFFERED":"1"}
    for log,module,arguments in plan:
        log.parent.mkdir(parents=True,exist_ok=True)
        print(f"Starting {log.relative_to(output)}",flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, "-X", "utf8", "-m", module, *map(str, arguments)],
                                    stdout=stream, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f"{module} failed; inspect {log}")
        print(f"Completed {log.relative_to(output)}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",choices=("pilot","controls","digit","supplement","threshold","fit","epochs3","pool","continual","accumulated"),default="pilot")
    parser.add_argument("--prepared-root", type=Path, default=PREPARED)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline",type=Path,default=BASELINE)
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps([{ "log":str(log),"module":module,"arguments":arguments}
            for log,module,arguments in build_plan(args.experiment,args.prepared_root,args.run_dir,args.baseline)],ensure_ascii=False,indent=2))
    else:
        run(args.prepared_root,args.run_dir,args.experiment,args.baseline)


if __name__ == "__main__":
    main()
