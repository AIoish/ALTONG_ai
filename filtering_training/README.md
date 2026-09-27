# Filtering model training

This directory contains offline dataset preparation, validation, model training,
and evaluation tools for the real-time filtering model.

Application runtime code remains in `src/filtering`. Training code may import
the shared schemas, prompts, and policy from that package, but runtime code must
not import from `filtering_training`.

## Setup

Install the CUDA-compatible PyTorch build as described in the repository root
README, then install the training dependencies:

```powershell
python -m pip install -r filtering_training/requirements.txt
```

## Current checks

Run these commands from the repository root:

```powershell
python -m filtering_training.validate_dataset
python -m filtering_training.demo_policy
python -m filtering_training.smoke_test_model
```

Generated checkpoints, adapters, and experiment outputs belong under
`filtering_training/outputs` and must not be committed.

## One-sample model check

```powershell
python -m filtering_training.infer_sample --sample-index 0
```

This runs `Qwen/Qwen3-0.6B` with thinking disabled, validates the four-field
label JSON, and applies the existing policy. The base model has not been
fine-tuned, so this command checks the input/output path rather than accuracy.

## Prepare SFT data

```powershell
python -m filtering_training.prepare_dataset
```

The command validates the samples and writes train, validation, and test JSONL
files plus a split manifest to `filtering_training/outputs/prepared`. The
current synthetic examples are only for pipeline checks. Notifications with the
same text and different contexts stay in one split. The assistant target contains
only the four model label fields; the prompt includes the normalized current
context. The three splits are not a reliable quality benchmark yet.

## Baseline evaluation and LoRA smoke run

```powershell
python -m filtering_training.evaluate --split test
python -m filtering_training.train --max-steps 2
python -m filtering_training.evaluate --split test --adapter filtering_training/outputs/lora-smoke/adapter
```

Run `prepare_dataset` first. Evaluation writes aggregate metrics and run metadata
to `filtering_training/outputs/evaluation`; it does not save raw notifications.
Model responses are not saved unless `--examples-output` is provided. By default, evaluation prints three model JSON examples; use
`--examples-output PATH` to save those examples as UTF-8 JSON. Record reviewed
experiment results and 3-5 examples in `docs/filtering-experiment-log.md`.
The trainer saves a LoRA adapter and run settings under
`filtering_training/outputs/lora-smoke`. It trains only on the assistant JSON
completion and verifies that prompt tokens are masked from the loss. On GPUs that
support it, the trainer uses BF16; otherwise it uses FP16.

The original 34-sample test split had three synthetic examples, including one urgent
notification. Its saved base/adapter reports belong to that dataset snapshot. Run
the baseline again after changing the dataset or split. These numbers cannot establish
model quality. The next quality step is a larger, independently reviewed dataset.

## Synthetic data audit

```powershell
python -m filtering_training.audit_dataset
```

This writes aggregate label and context coverage to
`filtering_training/outputs/audit/dataset_audit.json`. Review the aggregate findings before expanding the dataset.

## Provisional independent holdout

The 24 hand-written synthetic cases in
`filtering_training/data/evaluation_notifications.jsonl` are separate from the
51 training-pipeline samples. Their labels still need independent review, so this
is a provisional evaluation set rather than a final quality benchmark.
This 24-case file is local and Git-ignored: supply it locally to run holdout
evaluation. Repository tests create temporary synthetic fixtures instead.

The local, Git-ignored `filtering_training/data/real_notifications_sample.json` is
a reference for realistic notification types and wording, not a quota for app names
or topics. The holdout mixes those patterns with other plausible cases. Do not copy
personal content into synthetic data. Raw `sender: null` values normalize to an empty
string in the existing dataset/model-input schema.

```powershell
python -m filtering_training.prepare_holdout
python -m filtering_training.audit_dataset --dataset filtering_training/data/evaluation_notifications.jsonl --minimum-per-category 3
python -m filtering_training.evaluate --dataset filtering_training/data/evaluation_notifications.jsonl --prepared-dir filtering_training/outputs/holdout --split test
```

`prepare_holdout` rejects matching IDs or notification text and creates a
test-only manifest. It does not create a training split. Rebuild the manifest
after editing the holdout dataset; evaluation rejects a stale holdout manifest.
Rerun `prepare_holdout` after changing training data to check for new overlap.
The example outputs and metrics for the first run are recorded in
`docs/filtering-experiment-log.md`.

## Synthetic training candidates (batch 01)

```powershell
python -m filtering_training.generate_synthetic_candidates
```

The generator writes 192 candidate records to the Git-ignored
`filtering_training/outputs/candidates/synthetic_batch_01.jsonl`: 64 manually
written notification texts across all eight categories, each paired with
topic-adjacent, unrelated, and empty contexts. This is 64 distinct notifications, not
192 independent notification types. The private real-notification file is a
reference for plausible app names and short notification wording; the generator
does not read it. The other cases deliberately cover a wider range of work,
calendar, system, personal, and promotional notifications.

These labels and the generated `ai_summary_reason` text are provisional. Review
them before moving any candidates into the versioned training dataset or running
a full training job. Keep the independent 24-case holdout out of training.

## External notification source and processing

Source: [NotifAI synthetic notification dataset](https://huggingface.co/datasets/charlesfeng1/notifai-dataset).
Its dataset card states Apache 2.0 (the dataset API has no structured license
field). The downloaded `training_data.jsonl` contained
10,800 rows (SHA-256 `4e66f9070c5c9bf6bad30fc414feecf07ec6a5c9f7b72980edd00abf632546c9`);
the card's 16,000-row statement does not match that file. The original file is kept
only in the Git-ignored `filtering_training/outputs/external/notifai` directory.

```powershell
New-Item -ItemType Directory -Force filtering_training/outputs/external/notifai | Out-Null
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/charlesfeng1/notifai-dataset/resolve/e641bb311e31c6582143866eed37c88adf1ed8f9/training_data.jsonl' -OutFile filtering_training/outputs/external/notifai/training_data.jsonl
python -m filtering_training.select_external_candidates
python -m filtering_training.adapt_external_pilot
```

Selection removes malformed rows, repeated title/body pairs, overly short or long
texts, non-English alphabetic scripts, and obvious email/URL/long-number patterns.
It then chooses 3,000 local review candidates, 750 from each source folder, with
at most 300 from any one app. Source folders and priorities are selection metadata,
not ALTONG labels. This does not guarantee removal of all personal information;
review is still required. The resulting 3,000 English texts are **not** the 3,000
Korean training samples. They are references for writing new Korean cases,
not text to translate wholesale.

The 58-case pilot uses source notification ideas, rewrites the wording in Korean,
adapts app names to the Windows setting, adds fictional `CurrentContext` values,
and assigns all four ALTONG labels from the project guideline. It does not copy
source priority into `urgency_score`; the source ID mapping is preserved in the
Git-ignored pilot provenance file. The added cases cover all eight categories,
including neutral status notices labeled `기타`, and recent-process lengths 0-3.
A few fictional role labels exercise `sender`; no personal names are used.
These pilot labels remain provisional and the pilot is not used for model
training or the independent holdout. The low count of relevance score 2 and
the concentration of `기타` in repository notices remain coverage gaps.

## Fast external-source translation draft

The selected 3,000 public English notifications can be translated in one GPU batch
using [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) (MIT license).
The source is the Apache-2.0 NotifAI dataset linked above. Run:

```powershell
python -m filtering_training.translate_external_candidates --limit 50 --output filtering_training/outputs/external/notifai/translated_50.jsonl
python -m filtering_training.translate_external_candidates --limit 3000
python -m filtering_training.audit_external_translations
```

The translator keeps the public source ID and English title/body beside its Korean
draft for comparison. It deliberately discards source priority; no ALTONG context
or label is assigned. The output, manifest, and audit report stay Git-ignored.
On the local RTX 3060 Laptop GPU, the 3,000-draft run took about 135 seconds after
model loading. The first audit flagged 98 source IDs: 22 with an empty field, 69
with at least one missing source number, 11 with a duplicate translation, and 19
with very short output; reasons can overlap. Unflagged rows can still contain
mistranslations or omitted clauses. A four-case Qwen3-1.7B rewrite probe also
copied English verbatim in some cases, so it did not replace the review stage.

These 3,000 records are translation drafts, **not** 3,000 training samples.
Before training, rewrite or reject poor translations, add realistic Windows
contexts, assign ALTONG labels independently of the source folder/priority,
and check overlap with training and holdout notification families.
## Rapid Korean candidate set (3,000 rows)

No generation API or local translation model is needed for this path. Codex-authored
Korean notification scenarios, ten topic choices per scenario, and four meaningful
time/impact/action details produce deterministic candidates. The public NotifAI
dataset above and the private real sample informed broad notification types only;
this generator does not read or copy either file. It writes only to Git-ignored
outputs and keeps the original `notification`/`context`/`label` JSON schema.

```powershell
python -m filtering_training.generate_rapid_dataset
python -m filtering_training.audit_dataset --dataset filtering_training/outputs/candidates/rapid_korean_3000.jsonl --minimum-per-category 3
python -m filtering_training.prepare_rapid_dataset
```

The output has 3,000 distinct title/body pairs, 375 per category, but only 80
authored scenario families. Its `.lineage.json` records each family's identity.
`prepare_rapid_dataset` holds out one complete family per category for validation
and one for test; current counts are 2,398 train, 302 validation, 300 test.
The prepared SFT files and manifest are Git-ignored. They are **provisional**:
automatic schema and overlap checks do not establish natural Korean wording,
correct labels, or generalization to real notifications. Review representative
cases and add independently written cases before full training. Keep the separate
24-case human-authored holdout out of this candidate set and its training split.