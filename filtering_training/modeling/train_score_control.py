"""용도: 기존 분리 모델에서 숫자 loss 또는 정답 샘플링 하나만 바꾼 대조 학습.
생성일: 2026-10-04
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from filtering_training.common.score_loss import balanced_indices, weighted_score_loss
from filtering_training.common.score_tasks import parse_scores, prompt_digest
from filtering_training.modeling.train import load_training_pairs
from filtering_training.preparation.prepare_score_experiment import digest, write_json


def run(prepared, output, mode, smoke=False):
    if output.exists():
        raise ValueError("preserve previous run; choose a fresh directory")
    manifest = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    task = manifest["task"]
    if task not in ("urgency", "relevance") or manifest["prompt_sha256"] != prompt_digest(task):
        raise ValueError("this control requires a fixed single-score task")
    if mode not in ("numeric_weight", "balanced", "standard_nll"):
        raise ValueError("unknown control")
    for split in ("train", "validation"):
        if digest(prepared/f"{split}.jsonl") != manifest["splits"][split]["file_sha256"]:
            raise ValueError("prepared data changed")
    import torch
    import transformers
    import trl
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA is required")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    pairs = load_training_pairs(prepared/"train.jsonl", tokenizer, 768)
    records = [json.loads(line) for line in (prepared/"train.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = [parse_scores(r["messages"][-1]["content"], task)[task+"_score"] for r in records]
    indexes = balanced_indices(scores) if mode == "balanced" else list(range(len(scores)))
    selected = [pairs[i] for i in indexes]
    ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
    if any(len(value) != 1 for value in ids):
        raise ValueError("each score digit must be a single tokenizer token")
    digit_ids = [value[0] for value in ids]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True,
        dtype=dtype, device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    quantization = model.config.quantization_config
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    output.mkdir(parents=True)
    config = {"purpose": "one-variable fixed-data control; no final-test usage",
        "created_date": "2026-10-04", "model": "Qwen/Qwen3-0.6B", "score_task": task,
        "score_control": mode, "numeric_weight": 7 if mode == "numeric_weight" else 1,
        "loss_engine": "chunked_nll" if mode == "balanced" else "nll",
        "dataset_sha256": manifest["source_sha256"], "prompt_sha256": prompt_digest(task),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "prepared_manifest_sha256": digest(prepared/"manifest.json"),
        "train_count": len(selected), "max_steps": 2 if smoke else len(selected),
        "max_length": 768, "seed": 42, "learning_rate": 0.0002,
        "lora": {"r": 8,"alpha": 16,"dropout": 0.05,"target_modules": ["q_proj","v_proj"]},
        "method": "QLoRA", "quantization": quantization, "precision": str(dtype),
        "original_score_counts": dict(Counter(scores)),
        "selected_score_counts": dict(Counter(scores[i] for i in indexes)),
        "selected_prepared_indices": indexes, "selected_unique_records": len(set(indexes)),
        "selected_record_repeats": dict(Counter(indexes)), "digit_token_ids": digit_ids,
        "training_rows_tokens_one_pass": sum(len(tokenizer(p["prompt"]+p["completion"],add_special_tokens=False)["input_ids"]) for p in selected),
        "versions": {"torch":torch.__version__,"transformers":transformers.__version__,"trl":trl.__version__,"peft":peft.__version__},
        "code_sha256": {"trainer":digest(Path(__file__)),"loss":digest(Path(__file__).parents[1]/"common/score_loss.py")}}
    write_json(output/"run_config.json", config)
    args = SFTConfig(output_dir=str(output/"trainer"),max_steps=config["max_steps"],max_length=768,
        per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=0.0002,
        optim="adamw_torch",bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,
        gradient_checkpointing=True,save_strategy="no",eval_strategy="no",logging_steps=1,
        report_to="none",disable_tqdm=True,seed=42,data_seed=42,completion_only_loss=True,
        assistant_only_loss=False,eos_token=tokenizer.eos_token,
        loss_type="chunked_nll" if mode == "balanced" else "nll")
    def loss_func(outputs, labels, num_items_in_batch=None):
        return weighted_score_loss(outputs.logits, labels, digit_ids, numeric_weight=config["numeric_weight"])
    trainer = SFTTrainer(model=model,args=args,train_dataset=Dataset.from_list(selected),
        processing_class=tokenizer,peft_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
        bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"]),
        compute_loss_func=loss_func if mode!="balanced" else None)
    # Verify actual post-collation-style labels, not just the authored JSON targets.
    for record in trainer.train_dataset:
        labels = record["labels"]
        supervised = [v for v in labels if v != -100]
        if labels[0] != -100 or len(supervised) != 8 or sum(v in digit_ids for v in supervised) != 1:
            raise RuntimeError("expected masked prompt and exactly one digit in eight supervised tokens")
    config["verified_completion_tokens"] = 8
    config["numeric_loss_share"] = .5 if mode == "numeric_weight" else .125
    if any(p.requires_grad and "lora_" not in name for name,p in trainer.model.named_parameters()):
        raise RuntimeError("only adapters may be trained")
    torch.cuda.reset_peak_memory_stats()
    result=trainer.train()
    trainer.model.save_pretrained(output/"adapter")
    tokenizer.save_pretrained(output/"adapter")
    config.update({"training_loss":result.training_loss,"training_metrics":result.metrics,
        "log_history":trainer.state.log_history,"peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated()})
    write_json(output/"run_config.json",config)
    return {"task":task,"control":mode,"steps":config["max_steps"],"loss":config["training_loss"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--mode",choices=("numeric_weight","balanced","standard_nll"),required=True)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run(args.prepared_dir,args.run_dir,args.mode,args.smoke)))


if __name__=="__main__":
    main()
