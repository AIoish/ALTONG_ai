# Filtering model training

This is a historical README snapshot. Some unused tools and old model weights
were removed with user approval on 2026-10-09; recorded commands may reference
deleted artifacts. See `../FILE_GUIDE.md` for cleanup details and
`../../docs/filtering-experiment-log.md` for the chronological experiment log.

This directory contains offline dataset preparation, validation, model training,
and evaluation tools for the real-time filtering model.

Application runtime code remains in `src/filtering`. Training code may import
the shared schemas, prompts, and policy from that package, but runtime code must
not import from `filtering_training`.

The filtering policy is `urgency_score >= 4 or relevance_score >= 4`
(`urgency4_or_relevance4_v2`). Relevance 4 means directly useful to the current
goal; a shared topic alone does not qualify. The labeling guideline documents
the score boundaries. Evaluation reports record the policy version, and older
policy metrics must be recalculated before comparison. Regenerate prepared
SFT inputs before the next training run because the shared prompt has changed.

## Reviewed v3 pilot (2026-10-03)

The current dataset pilot uses realistic notification messages, urgency independent
of context, and relevance 4 for information directly useful to the current goal.
All cases assume focus mode is ON, including cases without an active window.
Older generated datasets are excluded from this pilot.

Artifacts are local under `outputs/v3_reviewed_01/`: 66 notification originals,
198 context variants, split into 150 training, 24 validation, and 24 test cases.
The first 16 originals and 8 boundary originals were individually reviewed;
the remaining 42 are authored synthetic candidates checked structurally. Two
of those candidates were added after review to provide an independent
calendar/advertisement group for evaluation. They are identified in `approval.json`.

`prepare_reviewed_v3.py` preserves the approved 192 cases and keeps all variants
of each notification, shared populated windows, and explicitly listed semantic
families within one split. This guards these known overlap sources; it is not
an exhaustive semantic similarity detector. `split_audit.json` records the
original preservation and SFT consistency checks. Each evaluation split has
only 8 independent notifications, so these are pilot measurements rather than
evidence of production quality. Test cases must remain unused for model selection.

```powershell
python -m filtering_training.preparation.prepare_reviewed_v3 --approval-text "딱 좋다. ㅇㅇ" --approval-date 2026-10-03
python -m filtering_training.modeling.train_linear_baseline --dataset filtering_training/outputs/v3_reviewed_01/dataset.jsonl --prepared-dir filtering_training/outputs/v3_reviewed_01/prepared --run-dir filtering_training/outputs/v3_reviewed_01/linear_baseline
python -m filtering_training.modeling.train --prepared-dir filtering_training/outputs/v3_reviewed_01/prepared --run-dir filtering_training/outputs/v3_reviewed_01/qwen06_pilot --model Qwen/Qwen3-0.6B --max-steps 150 --max-length 768 --qlora
python -m filtering_training.modeling.evaluate --dataset filtering_training/outputs/v3_reviewed_01/dataset.jsonl --prepared-dir filtering_training/outputs/v3_reviewed_01/prepared --split validation --model Qwen/Qwen3-0.6B --load-in-4bit --output filtering_training/outputs/v3_reviewed_01/qwen06_base_validation_report.json --predictions-output filtering_training/outputs/v3_reviewed_01/qwen06_base_validation_predictions.jsonl --show-examples 0
python -m filtering_training.modeling.evaluate --dataset filtering_training/outputs/v3_reviewed_01/dataset.jsonl --prepared-dir filtering_training/outputs/v3_reviewed_01/prepared --split validation --model Qwen/Qwen3-0.6B --adapter filtering_training/outputs/v3_reviewed_01/qwen06_pilot/adapter --load-in-4bit --output filtering_training/outputs/v3_reviewed_01/qwen06_pilot/validation_report.json --predictions-output filtering_training/outputs/v3_reviewed_01/qwen06_pilot/validation_predictions.jsonl --show-examples 0
```

The preparation command rejects nonempty output directories and unresolved review
edits. These commands document the recorded local run; use new output directories
and the actual approval statement for a new review. Model defaults elsewhere
still describe older experiments; this pilot passes the 0.6B model explicitly.

## Setup

Install the CUDA-compatible PyTorch build as described in the repository root
README, then install the training dependencies:

```powershell
python -m pip install -r filtering_training/requirements.txt
```

## Current checks

Run these commands from the repository root:

```powershell
python -m filtering_training.quality.validate_dataset
python -m filtering_training.legacy.diagnostics.demo_policy
python -m filtering_training.legacy.diagnostics.smoke_test_model
```

Generated checkpoints, adapters, and experiment outputs belong under
`filtering_training/outputs` and must not be committed.

## One-sample model check

```powershell
python -m filtering_training.modeling.infer_sample --sample-index 0
```

This runs `Qwen/Qwen3-1.7B` with thinking disabled, validates the four-field
label JSON, and applies the existing policy. The base model has not been
fine-tuned, so this command checks the input/output path rather than accuracy.

## Prepare SFT data

```powershell
python -m filtering_training.preparation.prepare_dataset
```

The command validates the samples and writes train, validation, and test JSONL
files plus a split manifest to `filtering_training/outputs/prepared`. The
current synthetic examples are only for pipeline checks. Notifications with the
same text and different contexts stay in one split. The assistant target contains
only the four model label fields; the prompt includes the normalized current
context. The three splits are not a reliable quality benchmark yet.

## Baseline evaluation and LoRA smoke run

```powershell
python -m filtering_training.modeling.evaluate --split test
python -m filtering_training.modeling.train --max-steps 2
python -m filtering_training.modeling.evaluate --split test --adapter filtering_training/outputs/lora-smoke/adapter
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
python -m filtering_training.quality.audit_dataset
```

This writes aggregate label and context coverage to
`filtering_training/outputs/archive/legacy_20261003/audit/dataset_audit.json`. Review the aggregate findings before expanding the dataset.

## Provisional development evaluation set

The 24 hand-written synthetic cases in
`filtering_training/data/evaluation_notifications.jsonl` are separate from the
51 training-pipeline samples. Their labels still need independent review, so this
is a development evaluation set rather than a final quality benchmark.
These 24 cases were used repeatedly for model and data choices. Use a new untouched,
independently reviewed local set for final reporting.
This 24-case file is local and Git-ignored: supply it locally to run holdout
evaluation. Repository tests create temporary synthetic fixtures instead.

The local, Git-ignored `filtering_training/data/real_notifications_sample.json` is
a reference for realistic notification types and wording, not a quota for app names
or topics. The holdout mixes those patterns with other plausible cases. Do not copy
personal content into synthetic data. Raw `sender: null` values normalize to an empty
string in the existing dataset/model-input schema.

```powershell
python -m filtering_training.preparation.prepare_holdout
python -m filtering_training.quality.audit_dataset --dataset filtering_training/data/evaluation_notifications.jsonl --minimum-per-category 3
python -m filtering_training.modeling.evaluate --dataset filtering_training/data/evaluation_notifications.jsonl --prepared-dir filtering_training/outputs/holdout --split test
```

`prepare_holdout` rejects matching IDs or notification text and creates a
test-only manifest. It does not create a training split. Rebuild the manifest
after editing the holdout dataset; evaluation rejects a stale holdout manifest.
Rerun `prepare_holdout` after changing training data to check for new overlap.
The example outputs and metrics for the first run are recorded in
`docs/filtering-experiment-log.md`.

## Synthetic training candidates (batch 01)

```powershell
python -m filtering_training.legacy.generation.generate_synthetic_candidates
```

The generator writes 192 candidate records to the Git-ignored
`filtering_training/outputs/archive/legacy_20261003/candidates/synthetic_batch_01.jsonl`: 64 manually
written notification texts across all eight categories, each paired with
topic-adjacent, unrelated, and empty contexts. This is 64 distinct notifications, not
192 independent notification types. The private real-notification file is a
reference for plausible app names and short notification wording; the generator
does not read it. The other cases deliberately cover a wider range of work,
calendar, system, personal, and promotional notifications.

These labels and the generated `ai_summary_reason` text are provisional. Review
them before moving any candidates into the versioned training dataset or running
a full training job. Keep the separate 24-case development evaluation set out of training.

## External notification source and processing

Source: [NotifAI synthetic notification dataset](https://huggingface.co/datasets/charlesfeng1/notifai-dataset).
Its dataset card states Apache 2.0 (the dataset API has no structured license
field). The downloaded `training_data.jsonl` contained
10,800 rows (SHA-256 `4e66f9070c5c9bf6bad30fc414feecf07ec6a5c9f7b72980edd00abf632546c9`);
the card's 16,000-row statement does not match that file. The original file is kept
only in the Git-ignored `filtering_training/outputs/archive/legacy_20261003/external/notifai` directory.

```powershell
New-Item -ItemType Directory -Force filtering_training/outputs/archive/legacy_20261003/external/notifai | Out-Null
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/charlesfeng1/notifai-dataset/resolve/e641bb311e31c6582143866eed37c88adf1ed8f9/training_data.jsonl' -OutFile filtering_training/outputs/archive/legacy_20261003/external/notifai/training_data.jsonl
python -m filtering_training.legacy.external.select_external_candidates
python -m filtering_training.legacy.external.adapt_external_pilot
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
training or the separate development evaluation set. The low count of relevance score 2 and
the concentration of `기타` in repository notices remain coverage gaps.

## Fast external-source translation draft

The selected 3,000 public English notifications can be translated in one GPU batch
using [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) (MIT license).
The source is the Apache-2.0 NotifAI dataset linked above. Run:

```powershell
python -m filtering_training.legacy.external.translate_external_candidates --limit 50 --output filtering_training/outputs/archive/legacy_20261003/external/notifai/translated_50.jsonl
python -m filtering_training.legacy.external.translate_external_candidates --limit 3000
python -m filtering_training.legacy.external.audit_external_translations
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
python -m filtering_training.legacy.generation.generate_rapid_dataset
python -m filtering_training.quality.audit_dataset --dataset filtering_training/outputs/archive/legacy_20261003/candidates/rapid_korean_3000.jsonl --minimum-per-category 3
python -m filtering_training.legacy.preparation.prepare_rapid_dataset
```

The output has 3,000 distinct title/body pairs, 375 per category, but only 80
authored scenario families. Its `.lineage.json` records each family's identity.
`prepare_rapid_dataset` holds out one complete family per category for validation
and one for test; current counts are 2,398 train, 302 validation, 300 test.
The prepared SFT files and manifest are Git-ignored. They are **provisional**:
automatic schema and overlap checks do not establish natural Korean wording,
correct labels, or generalization to real notifications. Review representative
cases and add independently written cases before treating results as release evidence. Keep the separate
24-case human-authored holdout out of this candidate set and its training split.
## Current model choice (2026-09-28)

The default training, evaluation, and sample inference model is now
`Qwen/Qwen3-1.7B`. The 500-step local comparison favored it over 0.6B, and BF16
LoRA training succeeded on the RTX 3060 Laptop GPU. The 500-step adapter still
blocked five of nine urgent development cases, so it is not a release model.
The earlier 0.6B experiments remain in the experiment log. On 2026-09-29, a fresh
1.7B LoRA run completed one pass over all 2,398 training rows in about 1,007 seconds.
On the same 24 development cases, policy accuracy fell from 19/24 to 17/24 and
urgent false blocks increased from 5/9 to 6/9. The full-pass adapter is not adopted
as the new baseline; the 500-step adapter remains a development comparison only.
Both adapters and all generated datasets stay local and Git-ignored.

Next: add 2,000 rows, 250 per category, with new scenario families and balanced
IT work, personal, promotional, and neutral cases. Prioritize realistic short
messages, deadlines, impact, and context relevance without copying development
evaluation sentences. Recheck labels and family overlap, then retrain and measure
on an untouched final evaluation set. The repeated 24-case set is for development.

## Targeted extension (2026-09-29): 5,000 local rows

```powershell
python -m filtering_training.legacy.generation.generate_targeted_dataset
python -m filtering_training.legacy.preparation.prepare_rapid_dataset --dataset filtering_training/outputs/archive/legacy_20261003/candidates/combined_korean_5000.jsonl --output-dir filtering_training/outputs/prepared_targeted_5000 --preserve-manifest filtering_training/outputs/archive/legacy_20261003/prepared_rapid/manifest.json
```

The added 2,000 rows contain 1,000 new title/body pairs with two contrasting
contexts each. Forty new scenario families cover operational failures, routine
IT work, imminent schedules, device/security states, personal deadlines, casual
messages, advertising, and neutral notices. All eight categories add 250 rows.
Urgency and category stay constant within context pairs; relevance changes.
Optional marketing deadlines and benign uses of 'now' are included as negatives.
Labels explain impact/deadline evidence and context relevance; they are provisional.
No raw private sample or development-evaluation text is used as source wording.

The combined dataset has 5,000 rows, 4,000 distinct title/body pairs, 120 families,
and 625 rows/category. Original validation/test assignments are preserved. New
context pairs remain in the same family and split. Prepared counts are 3,598
train, 702 validation, and 700 test; not all 5,000 rows are used for training.
Dataset, lineage, audit, and SFT files stay under ignored `outputs/`. Automatic
checks establish schema/overlap consistency, not independent label correctness
or model improvement. Retraining and unused evaluation data are still required.

## 4-bit base-model comparison

The default remains Qwen3-1.7B. A larger candidate can be evaluated without training:

```powershell
python -m filtering_training.modeling.evaluate --dataset filtering_training/data/evaluation_notifications.jsonl --prepared-dir filtering_training/outputs/holdout --split test --model Qwen/Qwen3-4B-Instruct-2507 --load-in-4bit --output filtering_training/outputs/archive/legacy_20261003/evaluation/qwen3_4b_instruct_nf4_dev24.json --examples-output filtering_training/outputs/archive/legacy_20261003/evaluation/qwen3_4b_instruct_nf4_predictions24.json --show-examples 24
```

For an original checkpoint, this requests bitsandbytes NF4 with double quantization and FP16 computation. A prequantized checkpoint uses its stored quantization configuration; the report records the actual settings. It loads
entirely on CUDA device 0; it does not silently offload to CPU. Initial download
uses the original checkpoint size, not the quantized in-memory size. The report
records quantization, raw responses, per-case generation time, model footprint,
and peak CUDA allocated/reserved memory. Timings exclude loading, tokenization,
and parsing and include the first generation; they are not end-to-end latency.
CUDA allocation statistics exclude some driver/library memory, so they are not
identical to `nvidia-smi`. Evaluation results remain local under ignored outputs.
The repeated 24 cases are development data, not a blind final test. Compare base
models separately from trained adapters and record precision differences.

The 2026-09-30 local run used the checksum-verified Unsloth prequantized distribution
of Qwen3-4B-Instruct-2507 (NF4, double quantization, BF16 computation), after the
original download stalled. Local model path:
`filtering_training/outputs/models/qwen3-4b-instruct-2507-bnb-4bit`.
Evaluate that path with the same command's `--model` argument. No Unsloth runtime
or new training was used. On the repeated development 24, policy accuracy was
17/24, urgent false blocks 1/9, and unnecessary passes 6/14. Mean generation time
was 6.53 seconds/case, and observed GPU memory was about 2,923 MiB. These are
local development results and do not establish final quality or QLoRA training fit.
The default remains Qwen3-1.7B pending a trained-candidate comparison.

## QLoRA training on the local 4B checkpoint

```powershell
python -m filtering_training.modeling.train --prepared-dir filtering_training/outputs/prepared_targeted_5000 --run-dir filtering_training/outputs/qlora-qwen3-4b-smoke-20260930 --model filtering_training/outputs/models/qwen3-4b-instruct-2507-bnb-4bit --qlora --max-steps 2 --learning-rate 0.0001 --lora-targets all-linear
```

Use a fresh run directory for each experiment. `--qlora` loads the base entirely
on CUDA 0, prepares k-bit training, and trains adapters only. The 2-step reload
probe succeeded on the local 6GB GPU; this is pipeline evidence, not quality
or worst-case memory evidence. `--save-steps 250 --eval-steps 250` enables
checkpoint saving and teacher-forced validation loss on the prepared validation
split. Validation loss is not policy accuracy. The run configuration records
actual quantization, data/model provenance, trainable parameter count, training
metrics, validation history, and peak CUDA allocation. Default model remains
1.7B, ordinary LoRA remains available, and generated data/adapters stay local.

The 2026-09-30 QLoRA experiment completed 500 steps over the expanded 3,598-row
training split, with all-linear rank-8 adapters and learning rate 0.0001. Full
702-row validation loss improved from 0.2815 at step 250 to 0.2525 at step 500.
Training plus both validations took about 1,270 seconds and peak CUDA reserved
memory was 4.95 GiB. On the repeated development 24, policy decisions were 24/24,
urgent false blocks 0/9, and unnecessary passes 0/14. Category accuracy was 18/24,
so this is not perfect prediction of all output fields. Mean generation time was
7.89 seconds/case. The adapter is a promising development candidate, pending
untouched evaluation, label review, and latency work. No default model switch,
training beyond 500 steps, commit, or push was performed.
