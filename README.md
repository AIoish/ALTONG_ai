# ALTONG_ai

## Environment

- Python 3.14.5

## Shared environment

Install the dependencies shared by the project with:

```powershell
python -m pip install -r requirements.txt
```

### PyTorch (CUDA 12.8)

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

PyTorch is installed separately because the correct wheel depends on the OS,
GPU, CUDA runtime, and Python version.

## Filtering model training

Offline dataset preparation, validation, training, and evaluation code is kept
in `filtering_training`. See `filtering_training/README.md` for setup and usage.

The `src/briefing`, `tests/briefing`, and briefing sample data are maintained
independently from the filtering model workflow.

## Local Qwen briefing provider

The application briefing pipeline uses the deterministic rule-based provider by
default. To use the locally trained Qwen LoRA adapter, construct a
`QwenBriefingProvider` and inject it into `SessionBriefingService`:

```python
from src.briefing import QwenBriefingProvider, SessionBriefingService

provider = QwenBriefingProvider(adapter_path=r"C:\path\to\briefing-qwen-lora")
service = SessionBriefingService(provider=provider)
```

The model and adapter are loaded lazily on the first non-empty summary request.
Model files must remain outside the repository (or under the ignored `outputs`
directory). If dependencies, adapter loading, GPU memory, inference, or JSON
validation fail, the provider logs the error and returns the existing
rule-based summary instead. The runtime never imports from `briefing_training`;
training and evaluation reuse the prompt contract owned by `src/briefing`.

Run the application pipeline against the synthetic briefing fixtures with a
local adapter:

```powershell
python -m briefing_training.smoke_test_runtime_provider `
  --adapter-path "C:\path\to\briefing-qwen-lora"
```

## Conversation grouping and schedule output

The briefing pipeline groups same-app, same-sender notification fragments when
their recognized sequence IDs are adjacent and they arrive within five
minutes. Sequence proximity is only a supporting signal: unrelated senders,
UUID-style IDs, non-adjacent IDs, and missing-sender fallbacks are not merged by
ID alone. Shared-text grouping is limited to a 30-minute group span.

Every schedule-related group, including one without a parseable date, is
returned inside its briefing card at `groups[].schedule_summaries`.

### Dashboard response contract

The response contains exactly `session_id` and `groups`. Each group is one
dashboard card with exactly these seven fields: `session_id`, `group_id`,
`app_name`, `sender`, `primary_category`, `summary_lines`, and
`schedule_summaries`. The category is selected from filtering results; groups
in the same category may still be separate cards.

Schedule reports are flattened: there is no nested `schedule_details`.
Each report contains exactly `summary_id`, `schedule_status`,
`is_all_day`, and the six fields `who`, `when`, `where`, `what`, `why`, `how`.
An empty schedule list is `[]`; source details that cannot be extracted remain
`null`.

- Calendar registration status is not returned by briefing. A downstream
  calendar integration must track its own verified registration results;
  finding a date in text never implies successful registration.
- `schedule_status` describes the source event:
  `scheduled`, `changed`, or `cancelled`. Consumers must not register a cancelled
  event as a new appointment.
- `is_all_day` is `false` for a parsed clock time, `true` for a date-only schedule,
  and `null` if no valid date/time was parsed. Date-only schedules are treated as
  all-day dates; they are not converted into artificial midnight appointments.
- `when` is a UTC ISO 8601 timestamp with `Z` for timed schedules, a
  `YYYY-MM-DD` date for date-only schedules, and `null` when unknown.
  Korean source clock times and relative dates use Asia/Seoul (UTC+09:00).
  Thus October 12 at 7 p.m. in Korea is `2026-10-12T10:00:00Z`.
  Missing or invalid dates must not trigger automatic calendar registration.

`todo_candidates`, `calendar_candidates`, their presence flags, public source
IDs, scores, counters, and `generated_at` are not returned. Task candidate
extraction and the duplicate calendar candidate model were removed. Internal
source references and diagnostic counters remain available for testing and
debugging. Briefing does not require or use urgency/relevance scores.

The following is a complete example, including every dashboard field:

```json
{
  "session_id": "session_001",
  "groups": [
    {
      "session_id": "session_001",
      "group_id": "group_001",
      "app_name": "KakaoTalk",
      "sender": "팀장",
      "primary_category": "일정/회의",
      "summary_lines": [
        "개발팀 중간 점검 회의는 10월 12일 오후 7시 창의관 402호에서 진행됩니다."
      ],
      "schedule_summaries": [
        {
          "summary_id": "schedule_001",
          "schedule_status": "scheduled",
          "is_all_day": false,
          "who": "개발팀",
          "when": "2026-10-12T10:00:00Z",
          "where": "창의관 402호",
          "what": "중간 점검 회의",
          "why": null,
          "how": null
        }
      ]
    }
  ]
}
```

### Role boundaries and integration

- `src/briefing/sqlite_adapter.py:load_session()` reads blocked notifications
  for a focus session. `pipeline.py:build()` validates them, removes duplicates,
  groups them, generates summaries, and extracts schedules.
- `categorization.py:categorize_group()` chooses the group category.
  `qwen_provider.py:QwenBriefingProvider.summarize()` generates summary lines.
  Qwen still outputs only `{"summary_lines":[...]}`, not the dashboard envelope.
- `action_items.py:RuleBasedActionItemProvider.extract()` extracts schedule
  fields from source text without registering them in a calendar.
- The application still needs to invoke this pipeline on session completion,
  deliver the report to the dashboard, and connect calendar registration /
  registration lookup. These actions are not implemented by `to_dict()`.

Changing this serialization or schedule parsing does not require model
retraining. Test the application contract; evaluate or retrain the model only
when its prompt or summary quality requires it. The runtime fixture input is
`data/sample/briefing/raw_notifications.json`.

## Tests

Run the repository test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

## Score-free briefing input

Briefing consumes notification identity, source text, timestamps, filtering
pass/block decisions, and category. Extra filtering score fields are ignored;
Qwen prompts and the rule-based fallback no longer use them. Newly generated
training records omit scores. Existing dataset files may contain historical
score fields; these are ignored by the shared prompt builder.

Changing dashboard serialization alone does not change model inputs. Removing
scores from prompts does change model inputs, so evaluate the existing adapter
with the new prompt before deciding whether new fine-tuning is necessary.
Resume the interrupted v5 run using its original code commit (`a102bb8`),
dataset, and options. Do not resume that checkpoint with the new prompt; use a
new output directory for any training under the score-free contract.
