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
minutes. Kakao additionally requires the same room title; for a known Kakao room
the ID-adjacency requirement is not necessary. Sequence proximity is only a
supporting signal for other apps: unrelated senders,
UUID-style IDs, non-adjacent IDs, and missing-sender fallbacks are not merged by
ID alone. Shared-text grouping is limited to a 30-minute group span.

Every schedule-related group, including one without a parseable date, is
returned inside its briefing card at `groups[].schedule_summaries`.

### Dashboard response contract

The response contains exactly `session_id` and `groups`. Each group is one
dashboard card with exactly these seven fields: `session_id`, `group_id`,
`app_name`, `sender`, `primary_category`, `summary_lines`, and
`schedule_summaries`. The category is predicted by the briefing model; groups
in the same category may still be separate cards.

There is no dedicated conversation ID yet. For KakaoTalk/KakaoTalk.exe only,
`title` is temporarily interpreted as the room name and must match before
messages can merge. `sender` remains the source-provided individual sender.
Grouping uses app, room (Kakao only), sender, body similarity and time proximity.
Known-room Kakao fragments within five minutes can merge even with UUID or
non-sequential IDs; a group spans at most 30 minutes. Other apps do not treat
their notification titles as room identifiers. Same-name rooms, room renaming,
and missing/fallback room titles remain limitations pending client integration.
`group_id` hashes the sorted source notification IDs; re-summarizing the same ID
set preserves it, while adding/removing notifications changes it. `summary_id`
is derived from `group_id`, not from a persistent calendar event identity.
Neither ID includes the session ID, so use `(session_id, group_id)` for a card's
storage key. Registration/deduplication of the same real-world calendar event
across groups requires a separate downstream identity strategy.

Schedule reports are flattened: there is no nested `schedule_details`.
Each report contains exactly `summary_id`, `schedule_status`,
`is_all_day`, `end_at`, and the six fields `who`, `when`, `where`, `what`, `why`, `how`.
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
- `when` is the start: a UTC ISO 8601 timestamp with `Z` for timed schedules, a
  `YYYY-MM-DD` date for date-only schedules, and `null` when unknown.
  Korean source clock times and relative dates use Asia/Seoul (UTC+09:00).
  Thus October 12 at 7 p.m. in Korea is `2026-10-12T10:00:00Z`.
  Missing or invalid dates must not trigger automatic calendar registration.
- `end_at` is a UTC ISO 8601 timestamp with `Z` only when an explicit end time
  can be paired with the start. Otherwise it is `null`, including date-only
  schedules. No default duration is added. Explicit ranges such as
  `오후 2시부터 4시까지` / `14:00~16:00` and separate end-time messages are
  supported. An end at or before the start is rejected unless a later date or
  `다음 날` is stated; ambiguous overnight ranges are not guessed.

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
          "end_at": null,
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
- `qwen_provider.py:QwenBriefingProvider.brief()` predicts the group category and
  summary in one call. Its direct output has exactly `primary_category` and
  `summary_lines`, not the dashboard envelope. Filtering supplies pass/block
  decisions; its category and score values are not model inputs in this mode.
  The offline fallback classifies source text with conservative rules; it is not
  evidence of trained classifier quality. Old v5/v6 summary adapters require
  `contract="summary"` and still do not perform learned classification.
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
pass/block decisions. Filtering category is optional and not used by the new
joint model. Extra filtering score fields are ignored;
Qwen prompts and the rule-based fallback no longer use them. Newly generated
training records omit scores. Existing dataset files may contain historical
score fields; these are ignored by the shared prompt builder.

## Category-organized summary report (v7)

The dashboard JSON stays `session_id + groups`; schedule_summaries is calendar
information inside a card, not the category report itself. `reports.py` collects
cards by the eight official categories in a fixed order, preserving separate
senders/rooms and their schedules. This does not call the LLM again or infer a
room-wide conclusion. `category_report(briefing)` returns a separate optional
`session_id + categories` JSON projection; `render_category_markdown(briefing)`
returns the readable report. The existing dashboard contract is unchanged.

Offline structure demo (no model/GPU, not a quality evaluation):

```powershell
python -m briefing_training.smoke_test_runtime_provider `
  --rule-based `
  --notifications data/sample/briefing/category_notifications.json `
  --filter-results data/sample/briefing/category_filter_results.json `
  --output outputs/category-dashboard-demo.json `
  --report-output outputs/category-briefing-demo.md `
  --category-output outputs/category-report-demo.json
```

For the trained v7 joint classifier, replace `--rule-based` with
`--adapter-path "<v7 adapter directory>"`. The fixture includes same-room
fragments, the same sender in a different room, and a different sender in the
same room. The reports must be requested explicitly; file writing and session
completion triggers are application integration responsibilities.

See `briefing_training/README.md` for the v8 policy-aligned dataset and training steps.
The v7 dataset/adapter remains available as a comparison baseline. v8 adds
source-grounded one-line requests, aligns scenario labels with the category
guide and avoids treating room names as summary facts. Calendar `what` extraction
also uses a concise source-backed event phrase for Kakao messages, rather than
copying a whole attendee/date sentence. These changes require model re-evaluation;
unit tests alone do not establish improved summary quality.
Changing the model's task from summary-only to classification+summary requires
new supervision; v6 checkpoint resumption is not the v7 training workflow.

Changing dashboard serialization alone does not change model inputs. Removing
scores from prompts does change model inputs, so evaluate the existing adapter
with the new prompt before deciding whether new fine-tuning is necessary.
Resume the interrupted v5 run using its original code commit (`a102bb8`),
dataset, and options. Do not resume that checkpoint with the new prompt; use a
new output directory for any training under the score-free contract.
