"""용도: JSON 생성 없이 다섯 숫자 logit으로 긴급도·관련도를 학습하고 검증한다.
생성일: 2026-10-04
"""

import argparse
from collections import Counter
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import time

from filtering_training.common.digit_scores import digit_loss, prompt_digest, replace_prompt
from filtering_training.common.score_loss import balanced_indices
from filtering_training.common.score_tasks import messages, parse_scores, score_metrics, load_score_samples, unique_urgency_samples
from filtering_training.modeling.evaluate import load_split_samples
from filtering_training.preparation.prepare_score_experiment import digest, write_json, write_lines

MODEL = "Qwen/Qwen3-0.6B"
SOURCE = Path("filtering_training/outputs/v3_reviewed_01")
PREPARED = Path("filtering_training/outputs/v3_score_separation_01/prepared")
VALIDATION_POOL = Path("filtering_training/outputs/v5_validation_pool_01")
INPUT_POLICY = "no notification variant id in either score input; urgency has no context"


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=dtype,
        device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
    if any(len(v) != 1 for v in ids):
        raise ValueError("digits must each use one token")
    return tokenizer, model, dtype, [v[0] for v in ids]


def encode(tokenizer, msgs):
    result = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
        tokenize=True, return_dict=True)
    if len(result["input_ids"]) > 768:
        raise ValueError("prompt exceeds fixed length")
    return dict(result)


def select_indices(scores, mode, subset=None):
    if subset is None:
        return balanced_indices(scores) if mode == "balanced" else list(range(len(scores)))
    if mode != "natural" or not subset or len(set(subset)) != len(subset):
        raise ValueError("subset requires natural mode and distinct indices")
    if any(type(i) is not int or not 0 <= i < len(scores) for i in subset):
        raise ValueError("subset indices must refer only to prepared train rows")
    return list(subset)


def continuation_snapshot(run, task):
    """Validate an existing adapter without loading a model or optimizer."""
    config=json.loads((run/"run_config.json").read_text(encoding="utf-8"))
    adapter=run/"adapter"
    settings=json.loads((adapter/"adapter_config.json").read_text(encoding="utf-8"))
    if config.get("task")!=task or config.get("model")!=MODEL or not config.get("training_metrics"):
        raise ValueError("continuation requires a completed matching score adapter")
    if config.get("prompt_sha256")!=prompt_digest(task) or config.get("dataset_sha256")!=digest(SOURCE/"dataset.jsonl"):
        raise ValueError("parent adapter prompt or source mismatch")
    if task=="relevance" and config.get("input_policy")!=INPUT_POLICY:
        raise ValueError("parent relevance adapter leaks notification ids")
    if config.get("source_manifest_sha256")!=digest(SOURCE/"prepared/manifest.json"):
        raise ValueError("parent split changed")
    if digest(Path(config["prepared_root"])/task/"manifest.json")!=config["prepared_manifest_sha256"]:
        raise ValueError("parent prepared provenance changed")
    if (settings.get("base_model_name_or_path")!=MODEL or settings.get("peft_type")!="LORA"
        or settings.get("r")!=8 or settings.get("lora_alpha")!=16 or settings.get("lora_dropout")!=.05
        or set(settings.get("target_modules",[]))!={"q_proj","v_proj"} or settings.get("bias")!="none"):
        raise ValueError("parent adapter architecture mismatch")
    from safetensors import safe_open
    weights=adapter/"adapter_model.safetensors"
    with safe_open(str(weights),framework="pt",device="cpu") as tensors:
        keys=list(tensors.keys())
        if not keys or any("lora_" not in key for key in keys):
            raise ValueError("parent weights must contain only LoRA tensors")
    return {"run":str(run),"run_config_sha256":digest(run/"run_config.json"),
            "adapter_config_sha256":digest(adapter/"adapter_config.json"),"adapter_sha256":digest(weights),
            "model_revision":config["model_revision"],"digit_token_ids":config["digit_token_ids"],
            "optimizer_state":"new optimizer; continue adapter weights only"}


def training_schedule(count, accumulation=1, epochs=None, max_steps=None, warmup=0.0):
    """Distinguish optimizer updates from examples, including the final partial batch."""
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("accumulation must be a positive integer")
    if not math.isfinite(warmup) or not 0 <= warmup < 1:
        raise ValueError("warmup must be a ratio in [0,1)")
    if epochs is not None:
        if type(epochs) is not int or epochs < 1 or max_steps is not None:
            raise ValueError("epochs require a positive integer and no max_steps override")
        updates = math.ceil(count / accumulation) * epochs
        return {"mode":"epochs", "num_train_epochs":epochs, "trainer_max_steps":-1,
                "expected_optimizer_steps":updates, "expected_examples":count*epochs,
                "gradient_accumulation_steps":accumulation, "effective_batch_size":accumulation,
                "warmup_ratio":warmup, "warmup_steps":math.ceil(updates*warmup)}
    steps = count if max_steps is None else max_steps
    if type(steps) is not int or steps < 1 or accumulation != 1 or warmup:
        raise ValueError("step-based historical experiments require positive steps and accumulation one without warmup")
    return {"mode":"steps", "num_train_epochs":None, "trainer_max_steps":steps,
            "expected_optimizer_steps":steps, "expected_examples":steps,
            "gradient_accumulation_steps":1, "effective_batch_size":1, "warmup_ratio":0.0,"warmup_steps":0}


def preflight_training(output, task, prepared_root=PREPARED, continue_from=None, learning_rate=None, max_steps=None,
                       accumulation=1, epochs=None, warmup=0.0):
    """Read-only readiness check; no model load, training, or output creation."""
    if output.exists():
        raise ValueError("choose a fresh run directory")
    prepared=prepared_root/task
    manifest=json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_ready") is False or manifest.get("quality_gate",{}).get("pending"):
        raise ValueError("prepared training draft has unfinished quality gates")
    if manifest.get("continuation_data") and continue_from is None:
        raise ValueError("continuation data requires an existing adapter")
    for split in ("train","validation"):
        if digest(prepared/f"{split}.jsonl")!=manifest["splits"][split]["file_sha256"]:
            raise ValueError("prepared split changed")
    if manifest["source_sha256"]!=digest(SOURCE/"dataset.jsonl"):
        raise ValueError("source changed")
    if manifest.get("source_manifest_sha256")!=digest(SOURCE/"prepared/manifest.json"):
        raise ValueError("source split changed")
    for kind,expected in manifest.get("digit_prompt_hashes",{}).items():
        if expected!=prompt_digest(task,history=kind=="history"):
            raise ValueError("prepared prompt changed")
    parent=continuation_snapshot(continue_from,task) if continue_from else None
    if manifest.get("continuation_parent") and parent!=manifest["continuation_parent"]:
        parent_config = json.loads((continue_from/"run_config.json").read_text(encoding="utf-8")) if parent else {}
        if (not parent or Path(parent_config.get("prepared_root", "")).resolve()!=prepared_root.resolve()
            or parent_config.get("prepared_manifest_sha256")!=digest(prepared/"manifest.json")
            or parent_config.get("selected_prepared_indices")!=list(range(manifest["splits"]["train"]["count"]))):
            raise ValueError("planned continuation adapter changed")
    rate=learning_rate if learning_rate is not None else (.00005 if parent else .0002)
    if not math.isfinite(rate) or rate<=0:
        raise ValueError("learning rate must be finite and positive")
    schedule=training_schedule(manifest["splits"]["train"]["count"],accumulation,epochs,max_steps,warmup)
    if manifest.get("validation_pool"):
        validation=Path(manifest["validation_pool"]["folder"])
        if digest(validation/"dataset.jsonl")!=manifest["validation_pool"]["dataset_sha256"]:
            raise ValueError("fixed expanded validation changed")
        samples,_,_=load_validation_for_evaluation(validation)
        validation_rows=len(samples)
    else:
        validation_rows=len(load_validation_for_evaluation()[0])
    records=[json.loads(line) for line in (prepared/"train.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records)!=manifest["splits"]["train"]["count"]:
        raise ValueError("prepared count differs")
    for record in records:
        parse_scores(record["messages"][-1]["content"],task)
        effective=replace_prompt(record["messages"],task)
        payload=json.loads(effective[1]["content"])
        if "id" in payload["notification"] or (task=="urgency" and "context" in payload):
            raise ValueError("training input leaks context or id")
    return {"purpose":"read-only continuation preflight; training not executed","task":task,
            "train_rows":len(records),"max_steps":schedule["expected_optimizer_steps"],"learning_rate":rate,"validation_rows":validation_rows,
            "schedule":schedule,"train_file_sha256":digest(prepared/"train.jsonl"),
            "parent":parent,"prepared_manifest_sha256":digest(prepared/"manifest.json"),
            "training_executed":False,"output_created":False}


def attach_training_adapter(model, parent=None):
    """Restore trainable weights for continuation instead of initializing LoRA."""
    from peft import PeftModel, LoraConfig, get_peft_model
    if parent:
        return PeftModel.from_pretrained(model,Path(parent["run"])/"adapter",is_trainable=True)
    return get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"]))


def digit_trainer_class(base_class, ids, threshold_weight=0.0):
    """Let Trainer average microbatch losses; this objective does not use token counts."""
    class DigitTrainer(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            self.training_examples_seen = 0

        def training_step(self, model, inputs, num_items_in_batch=None):
            self.training_examples_seen += len(inputs["labels"])
            return super().training_step(model, inputs, num_items_in_batch)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs, logits_to_keep=1, use_cache=False)
            loss = digit_loss(outputs.logits, labels, ids, threshold_weight)
            return (loss, outputs) if return_outputs else loss
    return DigitTrainer


def train(output, task, mode, smoke, prepared_root=PREPARED, max_steps=None, threshold_weight=0.0, subset=None,
          continue_from=None, learning_rate=None, dry_run=False, accumulation=1, epochs=None, warmup=0.0):
    if output.exists():
        raise ValueError("choose a fresh run directory")
    if threshold_weight and task != "urgency":
        raise ValueError("threshold auxiliary objective is restricted to urgency")
    if epochs is not None and (smoke or subset is not None or mode != "natural"):
        raise ValueError("epoch training uses the complete natural dataset")
    plan=preflight_training(output,task,prepared_root,continue_from,learning_rate,2 if smoke else max_steps,
                            accumulation,epochs,warmup)
    if continue_from and (mode!="natural" or subset is not None):
        raise ValueError("continuation uses the complete natural prepared dataset")
    if dry_run:
        return plan
    import torch
    import transformers
    import peft
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(42)
    prepared = prepared_root/task
    manifest = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_ready") is False:
        raise ValueError("prepared training draft has unfinished quality gates")
    if manifest.get("sampling",{}).get("required_mode") == "natural" and (mode != "natural" or subset is not None):
        raise ValueError("prepared sampling cap requires natural mode and the complete prepared train subset")
    if manifest.get("digit_prompt_sha256") and manifest["digit_prompt_sha256"] != prompt_digest(task):
        raise ValueError("digit prompt changed after training preparation")
    for split in ("train", "validation"):
        if digest(prepared/f"{split}.jsonl") != manifest["splits"][split]["file_sha256"]:
            raise ValueError("prepared split changed")
    if manifest["source_sha256"] != digest(SOURCE/"dataset.jsonl"):
        raise ValueError("source changed")
    tokenizer, model, dtype, ids = load_model()
    records = [json.loads(v) for v in (prepared/"train.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = [parse_scores(v["messages"][-1]["content"], task)[task+"_score"] for v in records]
    indices = select_indices(scores,mode,subset)
    rows = [{**encode(tokenizer, replace_prompt(records[i]["messages"], task)), "labels": scores[i]-1} for i in indices]
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if plan["parent"] and (ids!=plan["parent"]["digit_token_ids"] or getattr(model.config,"_commit_hash",None)!=plan["parent"]["model_revision"]):
        raise ValueError("continuation base revision or tokenizer mismatch")
    if plan["parent"] and continuation_snapshot(continue_from,task)!=plan["parent"]:
        raise ValueError("parent adapter changed during initialization")
    model = attach_training_adapter(model,plan["parent"])
    if any(p.requires_grad and "lora_" not in n for n,p in model.named_parameters()):
        raise RuntimeError("only adapters may train")
    initial_adapters = {n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad} if subset else None
    DigitTrainer = digit_trainer_class(Trainer, ids, threshold_weight)
    def collate(batch):
        if len(batch) != 1:
            raise ValueError("this fixed experiment requires batch size one")
        row = batch[0]
        return {k: torch.tensor([v], dtype=torch.long) for k,v in row.items()}
    output.mkdir(parents=True)
    config = {"purpose":"five-class conditional digit objective; no final-test usage", "created_date":datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),
        "model": MODEL, "task":task, "mode":mode, "objective":"CE over digit logits only; prompt-only input; no JSON/EOS targets",
        "threshold_weight":threshold_weight,"threshold_objective":"CE on summed probabilities of scores 1..3 versus 4..5",
        "dataset_sha256":digest(SOURCE/"dataset.jsonl"), "source_manifest_sha256":digest(SOURCE/"prepared/manifest.json"),
        "prepared_manifest_sha256":digest(prepared/"manifest.json"), "prompt_sha256":prompt_digest(task),
        "digit_token_ids":ids, "selected_prepared_indices":indices, "selected_unique_records":len(set(indices)),
        "original_score_counts":dict(Counter(scores)), "selected_score_counts":dict(Counter(scores[i] for i in indices)),
        "prepared_root":str(prepared_root), "supplement":manifest.get("supplement"),"pool":manifest.get("pool"),"sampling":manifest.get("sampling"),
        "diagnostic_subset":subset is not None,
        "selected_notification_ids":[manifest["splits"]["train"]["notification_ids"][i] for i in indices],
        "max_steps":plan["max_steps"], "seed":42,"learning_rate":plan["learning_rate"],"max_length":768,
        "training_schedule":plan["schedule"],"train_file_sha256":plan["train_file_sha256"],
        "continuation_parent":plan["parent"],"optimizer_resumed":False,"digit_prompt_hashes":manifest.get("digit_prompt_hashes"),
        "batch_size":1,"lora":{"r":8,"alpha":16,"dropout":.05,"target_modules":["q_proj","v_proj"]},
        "precision":str(dtype),"quantization":model.config.quantization_config.to_dict(),
        "model_revision":getattr(model.config,"_commit_hash",None),
        "versions":{"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__},
        "input_policy":INPUT_POLICY,
        "max_prompt_tokens":max(len(row["input_ids"]) for row in rows),
        "code_sha256":{"trainer":digest(Path(__file__)),"loss":digest(Path(__file__).parents[1]/"common/digit_scores.py")}}
    write_json(output/"run_config.json",config)
    schedule=plan["schedule"]
    args = TrainingArguments(output_dir=str(output/"trainer"),max_steps=schedule["trainer_max_steps"],
        num_train_epochs=schedule["num_train_epochs"] or 3,
        per_device_train_batch_size=1,gradient_accumulation_steps=accumulation,learning_rate=plan["learning_rate"],
        warmup_steps=schedule["warmup_steps"],lr_scheduler_type="linear",weight_decay=0.0,max_grad_norm=1.0,
        optim="adamw_torch",bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,
        gradient_checkpointing=True,save_strategy="no",eval_strategy="no",logging_steps=1,
        report_to="none",disable_tqdm=True,seed=42,data_seed=42,remove_unused_columns=False)
    print("Initializing trainer",flush=True)
    trainer = DigitTrainer(model=model,args=args,train_dataset=Dataset.from_list(rows),data_collator=collate)
    print("Final training configuration "+json.dumps({"task":task,"rows":len(rows),"learning_rate":plan["learning_rate"],
        "schedule":schedule,"lora":config["lora"],"precision":str(dtype),"max_length":768,
        "loss_averaging":"Trainer averages microbatch means; model_accepts_loss_kwargs=False"},ensure_ascii=False),flush=True)
    print("Starting optimizer steps",flush=True)
    result = trainer.train()
    if epochs is not None and (trainer.state.global_step!=schedule["expected_optimizer_steps"]
        or trainer.training_examples_seen!=schedule["expected_examples"]
        or abs(trainer.state.epoch-epochs)>1e-6):
        raise RuntimeError("actual examples, optimizer updates or epochs differ from planned training")
    if initial_adapters is not None:
        deltas = [(p.detach().float().cpu()-initial_adapters[n]) for n,p in model.named_parameters() if n in initial_adapters]
        config["adapter_updates"] = {"changed_tensors":sum(bool(d.count_nonzero()) for d in deltas),
            "tensor_count":len(deltas),"delta_l2":sum(float(d.square().sum()) for d in deltas)**.5}
    model.save_pretrained(output/"adapter")
    tokenizer.save_pretrained(output/"adapter")
    config.update(training_loss=result.training_loss,training_metrics=result.metrics,log_history=trainer.state.log_history,
                  actual_optimizer_steps=trainer.state.global_step,actual_training_examples=trainer.training_examples_seen,
                  loss_averaging="Trainer averages microbatch means; model_accepts_loss_kwargs=False",
                  trainer_settings={"gradient_accumulation_steps":accumulation,"warmup_steps":schedule["warmup_steps"],
                                    "lr_scheduler_type":"linear","weight_decay":0.0,"max_grad_norm":1.0})
    write_json(output/"run_config.json",config)


def load_validation_for_evaluation(validation_pool=None):
    """Keep legacy comparison and add only previously approved validation rows."""
    legacy=load_split_samples(SOURCE/"dataset.jsonl",SOURCE/"prepared","validation")
    groups={"legacy_24":[s.notification.id for s in legacy]}
    if validation_pool is None:
        return legacy,groups,None
    manifest=json.loads((validation_pool/"manifest.json").read_text(encoding="utf-8"))
    subset=manifest.get("evaluation_set",{})
    if manifest.get("intended_split")!="validation" or not subset or subset.get("training_used"):
        raise ValueError("approved validation subset required")
    if digest(validation_pool/subset["dataset_file"])!=subset["dataset_sha256"] or digest(validation_pool/"approval.json")!=subset["preview_approval_sha256"]:
        raise ValueError("approved validation subset changed")
    approved={}
    for source in subset["seed_sources"]:
        folder=Path(source["folder"])
        if digest(folder/"dataset.jsonl")!=source["dataset_sha256"] or digest(folder/"approval.json")!=source["approval_sha256"]:
            raise ValueError("validation approval evidence changed")
        for sample in load_score_samples(folder/"dataset.jsonl"):
            approved[sample.notification.id]=sample.model_dump(mode="json")
    preview=json.loads((validation_pool/"approval.json").read_text(encoding="utf-8"))
    for item in preview["approved_cases"]:
        approved[item["notification"]["id"]]=item
    extra=load_score_samples(validation_pool/subset["dataset_file"])
    if {s.notification.id for s in extra}!=set(subset["approved_ids"]) or {s.notification.id for s in extra}!=set(approved):
        raise ValueError("unreviewed validation rows in evaluation subset")
    for sample in extra:
        if sample.model_dump(mode="json")!=approved[sample.notification.id]:
            raise ValueError("validation label differs from approval")
    if set(groups["legacy_24"]) & set(approved):
        raise ValueError("duplicate evaluation ids")
    groups["approved_expansion"]=[s.notification.id for s in extra]
    return legacy+extra,groups,subset


def evaluate(output, runs, prepared_root=PREPARED, urgency_run=None, relevance_run=None, validation_pool=VALIDATION_POOL):
    if output.exists() or output.with_suffix(".predictions.jsonl").exists():
        raise ValueError("preserve previous evaluation")
    samples,groups,validation_subset=load_validation_for_evaluation(validation_pool)
    import torch
    from peft import PeftModel
    configs = {}
    task_runs = {"urgency": urgency_run or runs/"urgency", "relevance": relevance_run or runs/"relevance"}
    for task in ("urgency","relevance"):
        config = json.loads((task_runs[task]/"run_config.json").read_text(encoding="utf-8"))
        if task == "relevance" and config.get("input_policy") != INPUT_POLICY:
            raise ValueError("relevance adapter was trained with a different input policy")
        if (config["task"] != task or config["prompt_sha256"] != prompt_digest(task)
                or config["dataset_sha256"] != digest(SOURCE/"dataset.jsonl")
                or config["source_manifest_sha256"] != digest(SOURCE/"prepared/manifest.json")
                or config["prepared_manifest_sha256"] != digest(prepared_root/task/"manifest.json")):
            raise ValueError("adapter provenance mismatch")
        configs[task] = config
    tokenizer, model, _, ids = load_model()
    model = PeftModel.from_pretrained(model,task_runs["urgency"]/"adapter",adapter_name="urgency")
    model.load_adapter(task_runs["relevance"]/"adapter",adapter_name="relevance")
    model.eval()
    def predict(sample):
        prediction, probabilities = {}, {}
        for task in ("urgency","relevance"):
            if configs[task]["digit_token_ids"] != ids:
                raise ValueError("tokenizer mismatch")
            model.set_adapter(task)
            inputs = {k:torch.tensor([v],device=model.device) for k,v in encode(tokenizer,replace_prompt(messages(sample,task),task)).items()}
            with torch.inference_mode():
                logits = model(**inputs,logits_to_keep=1,use_cache=False).logits[0,-1,ids].float()
            prediction[task+"_score"] = int(logits.argmax())+1
            probabilities[task] = logits.softmax(-1).tolist()
        return prediction, probabilities
    predict(samples[0])
    rows, predictions, times = [], [], []
    for sample in samples:
        torch.cuda.synchronize()
        start = time.perf_counter()
        pred, probs = predict(sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        times.append(elapsed)
        predictions.append(pred)
        rows.append({"notification_id":sample.notification.id,"notification":sample.notification.model_dump(mode="json"),
            "context":sample.context.model_dump(mode="json"),"gold":sample.label.model_dump(),
            "prediction":pred,"conditional_probabilities":probs,"end_to_end_seconds":elapsed})
    write_lines(output.with_suffix(".predictions.jsonl"),rows)
    write_json(output,{"purpose":"validation-only conditional classification; output format guaranteed by code",
        "created_date":datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat(),"dataset_sha256":digest(SOURCE/"dataset.jsonl"),"split":"validation",
        "task_runs":{t:str(p) for t,p in task_runs.items()},
        "run_config_hashes":{t:digest(task_runs[t]/"run_config.json") for t in configs},
        "predictions_sha256":digest(output.with_suffix(".predictions.jsonl")),
        "metrics":score_metrics(samples,predictions),
        "validation_subset":validation_subset,"evaluated_rows":len(samples),"evaluated_originals":len(unique_urgency_samples(samples)),
        "metrics_by_set":{name:score_metrics([s for s in samples if s.notification.id in set(ids_)],
                               [pred for s,pred in zip(samples,predictions) if s.notification.id in set(ids_)])
                          for name,ids_ in groups.items()},
        "mean_seconds":sum(times)/len(times)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("train","evaluate"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--task",choices=("urgency","relevance"))
    parser.add_argument("--mode",choices=("natural","balanced"),default="natural")
    parser.add_argument("--runs",type=Path)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--prepared-root",type=Path,default=PREPARED)
    parser.add_argument("--max-steps",type=int)
    parser.add_argument("--threshold-weight",type=float,default=0.0)
    parser.add_argument("--urgency-run",type=Path)
    parser.add_argument("--relevance-run",type=Path)
    parser.add_argument("--validation-pool",type=Path,default=VALIDATION_POOL,help="add the frozen approved subset to legacy validation")
    parser.add_argument("--legacy-validation-only",action="store_true",help="evaluate only the original 24 context rows")
    parser.add_argument("--train-indices",help="comma-separated prepared train indices for fit diagnostics")
    parser.add_argument("--continue-from",type=Path,help="completed task run whose adapter weights are continued")
    parser.add_argument("--learning-rate",type=float)
    parser.add_argument("--gradient-accumulation-steps",type=int,default=1)
    parser.add_argument("--epochs",type=int)
    parser.add_argument("--warmup",type=float,default=0.0,help="fraction of planned optimizer updates")
    parser.add_argument("--dry-run",action="store_true",help="validate without loading models, creating output, or training")
    args = parser.parse_args()
    if args.action == "train":
        if not args.task: parser.error("train requires --task")
        if args.max_steps is not None and args.max_steps <= 0: parser.error("max-steps must be positive")
        subset = [int(v) for v in args.train_indices.split(",")] if args.train_indices is not None else None
        result=train(args.output,args.task,args.mode,args.smoke,args.prepared_root,args.max_steps,args.threshold_weight,subset,
                     args.continue_from,args.learning_rate,args.dry_run,args.gradient_accumulation_steps,args.epochs,args.warmup)
        if args.dry_run: print(json.dumps(result,ensure_ascii=False))
    else:
        if not args.runs: parser.error("evaluate requires --runs")
        evaluate(args.output,args.runs,args.prepared_root,args.urgency_run,args.relevance_run,None if args.legacy_validation_only else args.validation_pool)


if __name__ == "__main__":
    main()
