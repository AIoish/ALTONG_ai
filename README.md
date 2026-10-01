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

Calendar candidates keep the existing non-null `scheduled_at` contract. They
additionally expose `status` (`scheduled`, `changed`, or `cancelled`) and the
same six-question details. Every schedule-related group, including one without
a parseable time, is also returned in `schedule_summaries`:

```json
{
  "schedule_summaries": [
    {
      "status": "scheduled",
      "schedule_details": {
        "who": "캡스톤 참가자들",
        "when": "2026-10-12T19:00:00Z",
        "where": "창의관 402호",
        "what": "졸업 작품 중간 점검",
        "why": "시연 동선 확인",
        "how": "오프라인"
      }
    }
  ]
}
```

Fields that are not stated in the source remain `null`; the pipeline does not
guess missing schedule details. The synthetic raw-notification contract used by
the runtime smoke test is available at
`data/sample/briefing/raw_notifications.json`.

## Tests

Run the repository test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```
