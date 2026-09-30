"""Run a Qwen3-1.7B LoRA SFT training experiment."""

import argparse
import hashlib
import json
from pathlib import Path

from filtering_training import TRAINING_ROOT


PREPARED_DIR = TRAINING_ROOT / "outputs" / "prepared"
RUN_DIR = TRAINING_ROOT / "outputs" / "lora-smoke"
MODEL_NAME = "Qwen/Qwen3-1.7B"


def load_training_pairs(path: Path, tokenizer, max_length: int) -> list[dict[str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            messages = record["messages"]
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError(f"invalid message roles at line {line_number}")
            prompt = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            completion = messages[2]["content"] + tokenizer.eos_token
            length = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
            if length > max_length:
                raise ValueError(
                    f"sample at line {line_number} has {length} tokens; "
                    f"increase --max-length above {max_length} to avoid truncation"
                )
            pairs.append({"prompt": prompt, "completion": completion})
    if not pairs:
        raise ValueError("training split is empty")
    return pairs


def train(prepared_dir: Path, run_dir: Path, max_steps: int,
          max_length: int, seed: int, model_name: str, qlora: bool = False,
          learning_rate: float = 2e-4, save_steps: int = 0, eval_steps: int = 0,
          lora_targets: str = "q_proj,v_proj",
          init_adapter: Path | None = None) -> dict:
    import torch
    import transformers
    import trl
    import peft
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local LoRA training run")
    if max_steps < 1 or max_length < 1 or learning_rate <= 0 or save_steps < 0 or eval_steps < 0:
        raise ValueError("max_steps and max_length must be positive")
    if init_adapter is not None:
        if not qlora:
            raise ValueError("continuing this adapter requires --qlora")
        if not (init_adapter / "adapter_config.json").is_file():
            raise ValueError("initial adapter is missing")
    set_seed(seed)
    use_bf16 = torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pairs = load_training_pairs(prepared_dir / "train.jsonl", tokenizer, max_length)
    validation_pairs = load_training_pairs(prepared_dir / "validation.jsonl", tokenizer, max_length) if eval_steps else None
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    targets = "all-linear" if lora_targets == "all-linear" else lora_targets.split(",")
    source_config = None
    if init_adapter is not None:
        source_path = init_adapter.parent / "run_config.json"
        if not source_path.is_file():
            raise ValueError("initial adapter run configuration is missing")
        source_config = json.loads(source_path.read_text(encoding="utf-8"))
        if (source_config["model"] != model_name
                or source_config["dataset_sha256"] != manifest["source_sha256"]
                or source_config["max_length"] != max_length
                or source_config["lora"]["target_modules"] != targets
                or source_config.get("method") != "QLoRA"):
            raise ValueError("initial adapter does not match model, data, length, or LoRA configuration")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": model_name,
        "dataset_sha256": manifest["source_sha256"],
        "prepared_manifest_sha256": hashlib.sha256(
            (prepared_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "train_count": len(pairs),
        "max_steps": max_steps,
        "max_length": max_length,
        "seed": seed,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05,
                 "target_modules": targets},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "trl": trl.__version__, "peft": peft.__version__},
        "device": torch.cuda.get_device_name(0),
        "precision": "bf16" if use_bf16 else "fp16",
        "method": "QLoRA" if qlora else "LoRA",
        "learning_rate": learning_rate,
        "save_steps": save_steps, "eval_steps": eval_steps,
        "validation_count": len(validation_pairs) if validation_pairs else 0,
        "purpose": "local LoRA SFT experiment; quality requires separate evaluation",
        "initial_adapter": str(init_adapter) if init_adapter else None,
        "initial_adapter_run_config_sha256": hashlib.sha256(
            (init_adapter.parent / "run_config.json").read_bytes()
        ).hexdigest() if init_adapter else None,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model_kwargs = {"dtype": torch.bfloat16 if use_bf16 else torch.float16}
    if qlora:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        model_kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        ))
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if qlora:
        model = prepare_model_for_kbit_training(model)
    run_config["quantization"] = getattr(model.config, "quantization_config", None)
    if hasattr(run_config["quantization"], "to_dict"):
        run_config["quantization"] = run_config["quantization"].to_dict()
    provenance_path = Path(model_name) / "download_provenance.json"
    run_config["checkpoint_provenance"] = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else None
    model.config.use_cache = False
    if init_adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    args = SFTConfig(
        output_dir=str(run_dir / "trainer"),
        max_steps=max_steps,
        max_length=max_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        optim="adamw_torch",
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_total_limit=2,
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps or 500,
        per_device_eval_batch_size=1,
        logging_steps=1,
        report_to="none",
        disable_tqdm=True,
        seed=seed,
        data_seed=seed,
        completion_only_loss=True,
        assistant_only_loss=False,
        eos_token=tokenizer.eos_token,
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(pairs),
        eval_dataset=Dataset.from_list(validation_pairs) if validation_pairs else None,
        processing_class=tokenizer,
        peft_config=None if init_adapter is not None else lora,
    )
    prepared_sample = trainer.train_dataset[0]
    labels = prepared_sample.get("labels")
    if (labels is None or labels[0] != -100
            or not any(value == -100 for value in labels)
            or not any(value != -100 for value in labels)):
        raise RuntimeError("training data does not mask the prompt from loss")
    run_config["trainable_parameters"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if qlora and any(p.requires_grad and "lora_" not in name for name, p in trainer.model.named_parameters()):
        raise RuntimeError("QLoRA must only train adapter parameters")
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    run_config["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    run_config["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    run_config["log_history"] = trainer.state.log_history
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    run_config["training_loss"] = result.training_loss
    run_config["training_metrics"] = result.metrics
    run_config["adapter_dir"] = str(adapter_dir)
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--lora-targets", default="q_proj,v_proj")
    parser.add_argument("--init-adapter", type=Path)
    args = parser.parse_args()
    config = train(args.prepared_dir, args.run_dir, args.max_steps,
                   args.max_length, args.seed, args.model, args.qlora,
                   args.learning_rate, args.save_steps, args.eval_steps, args.lora_targets,
                   args.init_adapter)
    print(json.dumps({"training_loss": config["training_loss"],
                      "adapter_dir": config["adapter_dir"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
