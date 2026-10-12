# v8 → C# 전달용 GGUF 내보내기

실제 생성 파일과 평가 수치: [GGUF_VALIDATION.md](GGUF_VALIDATION.md).

[클라이언트 develop의 AI 연동 안내](https://github.com/AIoish/ALTONG_client/blob/develop/docs/AI_ARCHITECTURE_AND_INTEGRATION_GUIDE.md)는
LLamaSharp에서 `qwen_briefing.gguf`를 사용하는 방향입니다. 구현·속도·품질 검증 완료와는 구분하세요.

## 현재 클라이언트 코드와 먼저 맞출 부분

2026-10-10 확인한 `develop`에는 GGUF 실행 서비스나 LLamaSharp 패키지 참조가 없습니다.
현재 [ReportSummaryProvider.cs](https://github.com/AIoish/ALTONG_client/blob/develop/src/Altong.Client/Services/Calendar/ReportSummaryProvider.cs)는
`<session_id>.json`을 읽고, `schema_version: 1`, `session_id`, `items`를 요구합니다.
우리 최종 출력은 `session_id`, `groups`이므로 그대로 넣으면 읽히지 않습니다.

[NotificationSummaryItem.cs](https://github.com/AIoish/ALTONG_client/blob/develop/src/Altong.Client/Models/NotificationSummaryItem.cs)의
`items`는 `summary_id`, `text`, `is_schedule_related`, `schedule` 형식입니다.
일정은 `event_id`, `title`, `start`, `end`, `time_zone`, `location`, `details`를 사용합니다.
우리 `when/end_at`은 UTC `Z`지만, 현재 클라이언트의 캘린더 등록 검증은
`Asia/Seoul`인 경우 시각의 오프셋도 `+09:00`이어야 합니다.
같은 시점을 시간대에 맞게 변환해야 하며, `Z`를 문자열 치환해서는 안 됩니다.
종료 시각 `null`은 JSON으로 읽을 수 있지만 등록 전에 사용자가 보완해야 합니다.

따라서 팀장에게 다음을 먼저 확인하세요.

- 현재 JSON 파일 공급 방식으로 통합할지, GGUF 실행 기능을 새로 연결할지
- 클라이언트가 `groups`를 읽을지, 별도 연결 어댑터에서 `items`로 변환할지
- 일정의 변경·취소, 여러 일정, 종일 일정도 현재 화면에서 어떻게 표현할지

확인 전에는 기존 브리핑 JSON 계약을 변경하지 않습니다. GGUF 파일 하나만 전달해서 전체 연동이 완료되는 것은 아닙니다.

## 그대로 따르면 안 되는 차이

- 실제 v8 기본 모델은 **Qwen/Qwen3-0.6B**입니다. 문서 예시의 Qwen2.5로 병합하면 안 됩니다.
- 모델 직접 출력은 `primary_category`, `summary_lines` 두 필드입니다.
- 그룹화·일정 추출·UTC 변환·ID 생성·대시보드 JSON은 Python 코드입니다. GGUF에는 들어가지 않습니다.
- C#에서 한 세션 전체를 단순 요약 프롬프트로 넣고 일정까지 생성시키는 방식은 기존 학습 계약과 다릅니다.
- 팀장과 기존 로직의 C# 이식 또는 임시 Python 통합 검증을 먼저 합의해야 합니다.
- 그룹별 동일한 프롬프트와 Qwen3 비사고 모드, 고정 생성 조건을 사용하세요.
- 현재 v8 서비스는 `src/briefing/category_correction.py`로 원문 기반 분류 교정을 적용합니다.
  GGUF만 실행하면 원본 모델 분류입니다. 'v8 + 교정' 결과를 재현하려면 클라이언트에서도 같은 규칙을 적용해야 합니다.

## 1. 기본 모델과 LoRA 병합 (재학습 아님)

CPU·시스템 RAM을 사용하며 원본 v8은 수정하지 않습니다. 출력 폴더가 존재하면 오류로 중단합니다.

```powershell
cd "C:\Users\akio7\OneDrive\바탕 화면\ALTONG_ai"
$briefingPython = "C:\dev\ALTONG_ai_train_v5\.venv\Scripts\python.exe"
& $briefingPython -m briefing_training.export_merged_model `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --output-dir "C:\dev\ALTONG_models\briefing-qwen-v8-merged" --validate-only

& $briefingPython -m briefing_training.export_merged_model `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --output-dir "C:\dev\ALTONG_models\briefing-qwen-v8-merged"
```

병합 결과는 아직 GGUF가 아닌 Hugging Face 폴더입니다.
모델·토크나이저와 `briefing_prompt_example.json`을 저장합니다.
`category_correction_rules.json`도 함께 저장합니다. 이는 교정 규칙 명세이며 자동 실행되는 코드가 아닙니다.
클라이언트 담당자가 이 명세대로 원문을 검사하고 단일 카테고리 후보에만 적용하도록 구현해야 합니다.
예시에는 시스템 규칙, 사용자 JSON, `enable_thinking=False`로 렌더링한 프롬프트가 있습니다.
단순 C# 요약 비서 프롬프트로 교체하지 마세요.

## 2. 공식 변환 도구 준비

[llama.cpp](https://github.com/ggml-org/llama.cpp)의 변환 스크립트와 `llama-quantize.exe`가 필요합니다.
기존 학습 환경을 일괄 업그레이드하지 말고 별도 환경을 사용하세요.
아래 설치 경로가 이미 있다면 덮어쓰지 말고 기존 경로를 확인합니다.

```powershell
git clone --depth 1 --branch b11541 https://github.com/ggml-org/llama.cpp "C:\dev\llama.cpp"
uv venv --python 3.12 "C:\dev\llama.cpp\.venv-convert"
uv pip install --python "C:\dev\llama.cpp\.venv-convert\Scripts\python.exe" `
  -r "C:\dev\llama.cpp\requirements\requirements-convert_hf_to_gguf.txt"
uv pip install --python "C:\dev\llama.cpp\.venv-convert\Scripts\python.exe" "transformers==5.17.0"
```

Windows 실행 파일은 공식 릴리스에서 받거나 공식 빌드 안내에 따라 준비합니다.
팀장의 LLamaSharp 버전이 Qwen3 GGUF를 지원하는지도 확인해야 합니다.
이번 검증은 공식 `b11541` Windows CPU 릴리스와 같은 태그의 변환 스크립트를 사용했습니다.
실행 파일 경로는 `C:\dev\llama-tools-b11541\bin`입니다.
모델 토크나이저를 Transformers 5로 저장했으므로 변환 환경도 5.17.0으로 맞췄습니다.
공식 requirements의 Transformers 4를 그대로 쓰면 `extra_special_tokens` 형식 오류가 발생했습니다.

## 3. F16 변환 후 양자화

공식 변환 스크립트의 `--outtype`에는 `q4_k_m`이 없습니다.
먼저 F16 GGUF를 만들고 `llama-quantize`를 사용합니다. 기존 출력이 있으면 새 이름을 선택하세요.

```powershell
$convertPython = "C:\dev\llama.cpp\.venv-convert\Scripts\python.exe"
& $convertPython "C:\dev\llama.cpp\convert_hf_to_gguf.py" `
  "C:\dev\ALTONG_models\briefing-qwen-v8-merged" `
  --outfile "C:\dev\ALTONG_models\qwen_briefing_v8_f16.gguf" --outtype f16

& "C:\dev\llama-tools-b11541\bin\llama-quantize.exe" `
  "C:\dev\ALTONG_models\qwen_briefing_v8_f16.gguf" `
  "C:\dev\ALTONG_models\qwen_briefing_v8_q4_k_m.gguf" Q4_K_M

& "C:\dev\llama-tools-b11541\bin\llama-quantize.exe" `
  "C:\dev\ALTONG_models\qwen_briefing_v8_f16.gguf" `
  "C:\dev\ALTONG_models\qwen_briefing_v8_q8_0.gguf" Q8_0
```

## 4. 검증과 전달

GGUF 크기·속도·변환 시간은 실제 환경에서 측정합니다. 변환 자체로 품질 향상이 보장되지 않습니다.
병합 모델과 양자화 모델에서 같은 프롬프트로 JSON·한국어·부정 의미·일정 시각·최신 상태를 검증하세요.
클라이언트 예시의 3~5초나 500MB는 우리 모델의 측정 결과가 아닙니다.

Q4_K_M은 파일 생성·실행에는 성공했지만 대화형 8건에서 전체 조건 통과가
기존 HF 5/8에서 2/8로 떨어졌습니다. 생성 성공만으로 배포 파일로 선택하지 않습니다.
모델 병합·실행 엔진·양자화 모두 달라졌으므로 저하 원인을 양자화 하나로 단정하지 않습니다.

검증 서버를 별도 PowerShell에서 실행합니다. 아래 파일명은 비교하려는 후보에 맞춥니다.

```powershell
& "C:\dev\llama-tools-b11541\bin\llama-server.exe" `
  -m "C:\dev\ALTONG_models\qwen_briefing_v8_q8_0.gguf" `
  --host 127.0.0.1 --port 18089 -c 4096 -ngl 0 -np 1 --threads 6
```

다른 창에서 같은 프롬프트와 토큰으로 검증합니다. JSON 강제 문법이나 정답 보정은 사용하지 않습니다.
분류 교정은 기존 Python 평가와 동일하게 별도로 적용합니다.

```powershell
& "C:\dev\ALTONG_ai_train_v5\.venv\Scripts\python.exe" -B -m briefing_training.evaluate_gguf `
  --gguf-path "C:\dev\ALTONG_models\qwen_briefing_v8_q8_0.gguf" `
  --tokenizer-path "C:\dev\ALTONG_models\briefing-qwen-v8-merged" `
  --task briefing --greedy `
  --cases briefing_training/data/category_briefing_v9/conversation_evaluation.jsonl `
  --output outputs/v8-q8-conversation.json --review-output outputs/v8-q8-conversation-review.md
```

검증 후 서버 창에서 Ctrl+C로 종료합니다. LLamaSharp/C# 실제 통합 검증은 별도입니다.

서버가 실행 중일 때 최종 `groups` 계약도 실제 모델로 확인할 수 있습니다.
이것은 Python 연결 검증이며 C# 구현이 아닙니다. 실패를 규칙 기반으로 대체하지 않습니다.

```powershell
& "C:\dev\ALTONG_ai_train_v5\.venv\Scripts\python.exe" -B -m briefing_training.smoke_test_runtime_provider `
  --gguf-path "C:\dev\ALTONG_models\qwen_briefing_v8_q8_0.gguf" `
  --tokenizer-path "C:\dev\ALTONG_models\briefing-qwen-v8-merged" `
  --notifications data/sample/briefing/category_notifications.json `
  --filter-results data/sample/briefing/category_filter_results.json `
  --output outputs/v8-q8-dashboard.json --report-output outputs/v8-q8-category-report.md
```

기본 30건과 짧은 요청 8건은 평가 명령의 `--cases`를 각각
`briefing_training/data/category_briefing_v8/evaluation_cases.jsonl`,
`briefing_training/data/category_briefing_v8/short_request_evaluation.jsonl`로 바꿉니다.
출력 파일명도 변경해 앞선 결과를 보존합니다.

전달할 자료:

- 품질을 확인한 GGUF 후보 (클라이언트 파일명·경로와 맞추기, Q4를 자동 선택하지 않기)
- `briefing_prompt_example.json`, 이 안내와 브리핑 코드 커밋
- `category_correction_rules.json` 및 이를 적용할 클라이언트 코드 (별도 Python 연결이면 해당 교정 모듈)
- 최종 대시보드 JSON 계약(루트 README)
- `outputs/v8-category-evaluation-review.md`, `outputs/v8-short-review.md`
- 변환 후 검증 결과. 아직 검증하지 않았다면 'GGUF 변환 후보'라고 표시

학습 checkpoint의 optimizer/scheduler는 추론 배포에 필요 없습니다.
이 레포에는 Python 평가에서 로컬 llama-server를 검증하는 도구가 있습니다.
C# 이식이나 실제 클라이언트 실행은 아직 검증하지 않았습니다.
