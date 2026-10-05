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

## Run the human-reviewed evaluation set

```powershell
python -m briefing_training.evaluate
```

The fixed evaluation set contains 30 synthetic cases across all eight official
categories. Eight original regression cases are retained, twelve diverse
cases cover in-progress incidents, review feedback, completed security actions,
rescheduling, changed delivery locations, promotions, and corrected notices,
two cases cover fragmented same-sender chat messages, and eight cases cover
colloquial sentence fragments with omitted subjects, references such as
"that one" or "there," and facts spread across consecutive short messages.

The evaluator reports usable structured-output rate, strict raw JSON-contract
compliance, safe format-repair rate, expected-fact coverage, superseded-state
violations, per-case pass rate, and average generation latency. A bare JSON
string array is safely normalized to `summary_lines` for runtime resilience,
while the raw-contract and repair metrics preserve visibility into the model's
original format violation. Other malformed outputs remain errors.
Fact alternatives allow equivalent source expressions such as recovery and
normalization. Each case also contains a human-written reference summary for
manual review; the reference is never used for fine-tuning or exact-match
scoring. These metrics remain an MVP benchmark rather than a final model-quality
claim.

Write both the machine-readable result and a Markdown report that compares the
source notifications, human reference, and model output:

```powershell
python -m briefing_training.evaluate `
  --adapter-path <local-adapter-directory> `
  --output outputs/evaluation.json `
  --review-output outputs/evaluation-review.md
```

Generated checkpoints, adapters, and experiment outputs must remain under an
ignored `outputs` directory. Never add real notifications or personal data.

## Prepare synthetic fine-tuning data

Generate deterministic training and validation records from synthetic scenario
templates:

```powershell
python -m briefing_training.prepare_dataset
```

The default command writes 1,200 training cases to `data/train_cases.jsonl` and
160 validation cases to `data/validation_cases.jsonl`. This consists of a
balanced 1,000/120 base plus 200/40 failure-targeted examples. The fixed 30-case
`evaluation_cases.jsonl` file remains separate and must not be used for
fine-tuning. The generator contains 32 scenario types, with four scenarios for
each of the eight official filter categories. The 1,000 training cases are
balanced at 125 cases per category. The targeted examples emphasize final-state
completion, named entities, places, deadlines, purposes, and natural summaries
of facts spread across short messages. Validation cases use held-out entities,
different dates, and validation-only surface forms. No real notifications or
personal data are included. The fragmented-chat scenarios use sequential
notification IDs at one-minute intervals so that separate short messages
contribute different facts to one summary. Of the base training cases, 248
specifically use colloquial sentence fragments such as "that one," "you know,"
and omitted predicates; another 248 use more complete but still fragmented
multi-message conversations.
Other scenarios cover recovery, rollback, failover,
completion, cancellation, correction, security response, payment, travel,
delivery, refund, subscription, promotion, maintenance, and lost-property
states. Scenario values are composed independently instead of repeating a
fixed record with a different date, and scenario families are interleaved in
the generated files. Generation fails if any two message-body sequences become
identical after IDs, dates, and times are ignored.

To create a smaller temporary dataset, override the exact record counts:

```powershell
python -m briefing_training.prepare_dataset `
  --train-count 240 `
  --validation-count 48 `
  --targeted-train-count 48 `
  --targeted-validation-count 16
```

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
parameters. The default sequence length is 2,048 tokens so the summary target
is not truncated after the Korean prompt. The base model remains unchanged.
Training outputs must stay in Google Drive or the ignored local `outputs`
directory and must not be committed to Git.

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
