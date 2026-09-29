# Briefing model experiments

This directory contains offline prompt experiments and evaluation tools for
group-level notification summaries. Application runtime code remains in
`src/briefing`, and runtime code must not import from `briefing_training`.

The first baseline uses `Qwen/Qwen3-0.6B` without fine-tuning. It disables the
thinking mode, requests a one-to-three-line Korean JSON summary, and evaluates
only synthetic notification groups.

## Setup

Install a PyTorch build compatible with the machine first. Then install the
briefing experiment dependencies from the repository root:

```powershell
python -m pip install -r briefing_training/requirements.txt
```

## Run one case

```powershell
python -m briefing_training.smoke_test_model
```

Choose another synthetic case with `--case-index`:

```powershell
python -m briefing_training.smoke_test_model --case-index 1
```

## Run the fixed evaluation set

```powershell
python -m briefing_training.evaluate
```

The evaluator reports structured JSON compliance, expected-fact coverage,
superseded-state violations, per-case pass rate, and average generation latency.
Fact alternatives allow equivalent source expressions such as recovery and
normalization. These metrics are a small baseline check, not a final
model-quality benchmark.

Generated checkpoints, adapters, and experiment outputs must remain under an
ignored `outputs` directory. Never add real notifications or personal data.

## Prepare synthetic fine-tuning data

Generate deterministic training and validation records from synthetic scenario
templates:

```powershell
python -m briefing_training.prepare_dataset
```

The default command writes 200 training cases to `data/train_cases.jsonl` and
40 validation cases to `data/validation_cases.jsonl`. The fixed eight-case
`evaluation_cases.jsonl` file remains separate and must not be used for
fine-tuning. Validation cases use held-out entities and November dates, while
training cases use October dates. All eight official filter categories are
represented, and no real notifications or personal data are included.

Validate the generated fine-tuning records without loading a model:

```powershell
python -m briefing_training.train_lora --validate-only
```

## Train a QLoRA adapter in Colab

Open `colab_train_qwen_lora.ipynb` in Google Colab, select a GPU runtime, and
run the cells in order. The notebook installs the dependencies from
`requirements-lora.txt`, validates the synthetic data, and saves the adapter
under `MyDrive/ALTONG_models/briefing-qwen-lora`.

The notebook removes Colab's preinstalled `torchao` before installing the
training dependencies. This project uses bitsandbytes NF4 rather than torchao,
and an older preinstalled torchao release can prevent current PEFT versions
from loading the saved adapter.

The training command uses 4-bit NF4 quantization and trains only LoRA adapter
parameters. The base model remains unchanged. Training outputs must stay in
Google Drive or the ignored local `outputs` directory and must not be committed
to Git.

## Evaluate a trained adapter

Run the same fixed evaluation set with the saved adapter and optionally write
the JSON report to a file:

```powershell
python -m briefing_training.evaluate `
  --adapter-path /content/drive/MyDrive/ALTONG_models/briefing-qwen-lora `
  --output /content/drive/MyDrive/ALTONG_models/briefing-qwen-lora/evaluation.json
```

The adapter evaluation must use `data/evaluation_cases.jsonl`, which is kept
separate from the training and validation data. Compare its structured output
rate, case pass rate, fact coverage, and latency with the base-model report.
