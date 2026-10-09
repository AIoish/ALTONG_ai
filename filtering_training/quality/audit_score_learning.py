"""용도: 분리 점수 모델의 정답 분포·숫자 토큰 학습·단독 로딩 출력을 진단한다.
생성일: 2026-10-03
"""

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import re

from filtering_training.common.score_tasks import parse_scores, prompt_digest
from filtering_training.preparation.prepare_score_experiment import digest, write_json


def numeric_spans(completion, field):
    """Return character spans of score values, excluding field names and syntax."""
    return [m.span(1) for m in re.finditer(r'"'+re.escape(field)+r'"\s*:\s*([1-5])(?=\s*[,}])', completion)]


def diagnose(prepared_root, runs, output, created_date="2026-10-03"):
    if output.exists():
        raise ValueError("preserve earlier diagnosis")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("this local diagnostic requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    number_ids = {str(i): tokenizer.encode(str(i), add_special_tokens=False) for i in range(1,6)}
    if any(len(ids) != 1 for ids in number_ids.values()):
        raise ValueError("diagnostic expects one token per digit")
    report = {"purpose": "training-fit and token diagnostic, not unseen final evaluation",
              "created_date": created_date, "tasks": {},
              "environment": {"torch": torch.__version__,"transformers":transformers.__version__}}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for task in ("urgency", "relevance"):
        prepared = prepared_root/task
        config = json.loads((runs/task/"run_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
        if (config["prompt_sha256"] != prompt_digest(task)
                or config["prepared_manifest_sha256"] != digest(prepared/"manifest.json")
                or config["score_task"] != task):
            raise ValueError("adapter task mismatch")
        records = {}
        for split in ("train","validation"):
            if digest(prepared/f"{split}.jsonl") != manifest["splits"][split]["file_sha256"]:
                raise ValueError("prepared file changed")
            records[split] = [json.loads(line) for line in (prepared/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        field = task+"_score"
        counts = {split: dict(sorted(Counter(json.loads(r["messages"][-1]["content"])[field]
                                             for r in values).items())) for split,values in records.items()}
        chosen = []
        per_label = Counter()
        for index, record in enumerate(records["train"]):
            gold = json.loads(record["messages"][-1]["content"])[field]
            if per_label[gold] < 2:
                chosen.append(("train", index, record))
                per_label[gold] += 1
        chosen += [("validation", i, r) for i,r in enumerate(records["validation"])]
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True,
                    dtype=dtype, device_map={"":0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
        # Only one adapter is loaded: compare against the earlier two-adapter inference.
        model = PeftModel.from_pretrained(model,runs/task/"adapter")
        model.eval()
        rows = []
        for split,index,record in chosen:
            prompt = tokenizer.apply_chat_template(record["messages"][:2],tokenize=False,
                            add_generation_prompt=True,enable_thinking=False)
            completion = record["messages"][-1]["content"]+tokenizer.eos_token
            combined = prompt+completion
            encoded = tokenizer(combined,add_special_tokens=False,return_offsets_mapping=True)
            spans = [(len(prompt)+a,len(prompt)+b) for a,b in numeric_spans(completion,field)]
            numeric = [i for i,(a,b) in enumerate(encoded["offset_mapping"])
                       if any(a < end and b > start for start,end in spans)]
            if len(numeric) != 1:
                raise ValueError("expected exactly one numeric score token")
            completion_positions = [i for i,(a,b) in enumerate(encoded["offset_mapping"])
                                    if b > len(prompt)]
            input_ids = torch.tensor([encoded["input_ids"]],device=model.device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits[0].float()
                loss = torch.nn.functional.cross_entropy(logits[:-1],input_ids[0,1:],reduction="none")
                num_index = numeric[0]
                probabilities = logits[num_index-1].softmax(-1)
                digit_probabilities = {n: float(probabilities[ids[0]].item()) for n,ids in number_ids.items()}
                best_token = int(logits[num_index-1].argmax().item())
                free_ids = tokenizer(prompt,add_special_tokens=False,return_tensors="pt").to(model.device)
                generated = model.generate(**free_ids,max_new_tokens=64,do_sample=False,
                                            pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(generated[0][free_ids["input_ids"].shape[-1]:],skip_special_tokens=True)
            try:
                predicted = parse_scores(raw,task)[field]
            except ValueError:
                predicted = None
            syntax_positions = [i for i in completion_positions if i not in numeric and i > 0]
            rows.append({"split":split,"prepared_index":index,
                         "notification_id":manifest["splits"][split]["notification_ids"][index],
                         "gold":json.loads(record["messages"][-1]["content"])[field],
                         "free_prediction":predicted,"raw":raw,
                         "numeric_teacher_forced_greedy":tokenizer.decode([best_token]),
                         "digit_probabilities":digit_probabilities,
                         "numeric_nll":float(loss[num_index-1].item()),
                         "syntax_eos_mean_nll":float(loss[[i-1 for i in syntax_positions]].mean().item()),
                         "numeric_tokens":len(numeric),"completion_tokens":len(completion_positions)})
            print(task,split,index,'gold',rows[-1]['gold'],'predicted',predicted,flush=True)
            del logits, loss, probabilities, input_ids, generated, free_ids
        task_report = {"counts":counts,"train_probe_rule":"first two records per represented gold score; not a full train accuracy",
                       "adapter_sha256":digest(runs/task/"adapter/adapter_model.safetensors"),
                       "prompt_sha256":prompt_digest(task),"prepared_manifest_sha256":digest(prepared/"manifest.json"),
                       "rows":rows,"validation_matches_multi_adapter":None}
        original = [json.loads(line) for line in (runs/"C.predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        previous_by_id = {r["notification_id"]:r for r in original}
        task_report["validation_matches_multi_adapter"] = all(
            row["raw"] == previous_by_id[row["notification_id"]]["responses"][task]["raw"]
            for row in rows if row["split"] == "validation")
        report["tasks"][task] = task_report
        del model
        gc.collect()
        torch.cuda.empty_cache()
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root",type=Path,required=True)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--created-date",default="2026-10-04")
    args=parser.parse_args()
    diagnose(args.prepared_root,args.runs,args.output,args.created_date)


if __name__=="__main__":
    main()
