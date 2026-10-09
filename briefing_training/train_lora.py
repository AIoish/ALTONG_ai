"""Train a QLoRA adapter for structured briefing summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .prepare_dataset import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALIDATION_PATH,
    training_messages,
    validate_records,
)
from .prompts import MODEL_NAME
from .prepare_category_dataset import (
    TRAIN_PATH as CATEGORY_TRAIN_PATH, VALIDATION_PATH as CATEGORY_VALIDATION_PATH,
    training_messages as category_training_messages,
    validate_records as validate_category_records,
)


DEFAULT_OUTPUT_DIRECTORY = Path("outputs") / "briefing-qwen-lora"


def load_records(path: str | Path, *, expected_split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number} in {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record at line {line_number} in {path} must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"{path} must contain at least one record")
    modes = {"primary_category" in record.get("target", {}) for record in records}
    if len(modes) != 1:
        raise ValueError("cannot mix summary-only and category+summary training records")
    validator = validate_category_records if True in modes else validate_records
    validator(records, expected_split=expected_split)
    return records


def model_input_fingerprint(record: Mapping[str, Any]) -> str:
    group = record.get("input")
    if not isinstance(group, Mapping):
        raise ValueError("record must contain an input object")
    visible_group = dict(group)
    notifications = visible_group.get("notifications")
    if isinstance(notifications, Sequence) and not isinstance(
        notifications, (str, bytes)
    ):
        visible_group["notifications"] = [
            {key: value for key, value in notification.items() if key != "id"}
            if isinstance(notification, Mapping)
            else notification
            for notification in notifications
        ]
    return json.dumps(visible_group, ensure_ascii=False, sort_keys=True)


def validate_split_separation(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> None:
    train_inputs = {model_input_fingerprint(record) for record in train_records}
    validation_inputs = {
        model_input_fingerprint(record) for record in validation_records
    }
    overlap = train_inputs & validation_inputs
    if overlap:
        raise ValueError(
            f"training and validation data contain {len(overlap)} duplicate inputs"
        )


def validate_resume_checkpoint(path: Path | None) -> Path | None:
    """Validate a Trainer checkpoint before loading the model and datasets."""
    if path is None:
        return None
    checkpoint = path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"resume checkpoint directory does not exist: {path}")
    if not (checkpoint / "trainer_state.json").is_file():
        raise ValueError(
            f"resume checkpoint is missing trainer_state.json: {checkpoint}"
        )
    return checkpoint


def to_prompt_completion(record: Mapping[str, Any]) -> dict[str, Any]:
    joint = "primary_category" in record.get("target", {})
    messages = category_training_messages(record) if joint else training_messages(record)
    prompt = [dict(message) for message in messages[:-1]]
    if not joint:
        prompt[-1]["content"] = f"{prompt[-1]['content']}\n/no_think"
    return {
        "prompt": prompt,
        "completion": [dict(messages[-1])],
    }


def render_prompt_completion(
    record: Mapping[str, Any], tokenizer: Any
) -> dict[str, str]:
    example = to_prompt_completion(record)
    template_options = {"enable_thinking": False} if "primary_category" in record.get("target", {}) else {}
    prompt = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        **template_options,
    )
    completion = example["completion"][0]["content"]
    if tokenizer.eos_token:
        completion = f"{completion}{tokenizer.eos_token}"
    return {"prompt": prompt, "completion": completion}


def build_dataset(records: Sequence[Mapping[str, Any]], tokenizer: Any):
    from datasets import Dataset

    return Dataset.from_list(
        [render_prompt_completion(record, tokenizer) for record in records]
    )


def train(
    *,
    train_path: Path,
    validation_path: Path,
    output_directory: Path,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_length: int,
    resume_from_checkpoint: Path | None = None,
) -> None:
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU in Colab or a configured local environment")

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = load_records(train_path, expected_split="train")
    validation_records = load_records(
        validation_path, expected_split="validation"
    )
    validate_split_separation(train_records, validation_records)
    train_dataset = build_dataset(train_records, tokenizer)
    validation_dataset = build_dataset(validation_records, tokenizer)

    training_config = SFTConfig(
        output_dir=str(output_directory),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=max_length,
        completion_only_loss=True,
        gradient_checkpointing=True,
        fp16=not use_bf16,
        bf16=use_bf16,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        quantization_config=quantization_config,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    from transformers import TrainerCallback

    task = "briefing" if "primary_category" in train_records[0]["target"] else "summary"

    class ContractCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            (checkpoint_dir / "briefing_task.json").write_text(
                json.dumps({"task": task}), encoding="utf-8",
            )

    trainer.add_callback(ContractCallback())
    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint else None
        )
    )
    trainer.save_model(str(output_directory))
    tokenizer.save_pretrained(output_directory)
    (output_directory / "briefing_task.json").write_text(
        json.dumps({"task": task}), encoding="utf-8",
    )
    trainer.save_metrics("train", result.metrics)
    evaluation_metrics = trainer.evaluate()
    trainer.save_metrics("eval", evaluation_metrics)
    print(f"Saved LoRA adapter to {output_directory}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("briefing", "summary"), default="briefing")
    parser.add_argument("--train-path", type=Path)
    parser.add_argument(
        "--validation-path", type=Path
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="resume optimizer, scheduler, and model state from a Trainer checkpoint",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.train_path = args.train_path or (CATEGORY_TRAIN_PATH if args.task == "briefing" else DEFAULT_TRAIN_PATH)
    args.validation_path = args.validation_path or (CATEGORY_VALIDATION_PATH if args.task == "briefing" else DEFAULT_VALIDATION_PATH)

    train_records = load_records(args.train_path, expected_split="train")
    validation_records = load_records(
        args.validation_path, expected_split="validation"
    )
    validate_split_separation(train_records, validation_records)
    expected_joint = args.task == "briefing"
    if any(("primary_category" in record["target"]) != expected_joint for record in [*train_records, *validation_records]):
        raise ValueError("dataset output contract does not match --task")
    print(
        f"Validated {len(train_records)} training and "
        f"{len(validation_records)} validation cases"
    )
    if args.validate_only:
        return
    if expected_joint and args.resume_from_checkpoint:
        state_file = args.resume_from_checkpoint / "briefing_task.json"
        if not state_file.is_file() or json.loads(state_file.read_text(encoding="utf-8")).get("task") != "briefing":
            raise ValueError("cannot resume a v5/v6 summary-only checkpoint for the new briefing task")

    resume_from_checkpoint = validate_resume_checkpoint(
        args.resume_from_checkpoint
    )

    train(
        train_path=args.train_path,
        validation_path=args.validation_path,
        output_directory=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        resume_from_checkpoint=resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
