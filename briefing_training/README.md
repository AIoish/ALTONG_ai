# 브리핑 모델 실행·학습·평가 안내

## 기존 어댑터를 이용한 추가 학습 준비

`--initial-adapter`는 기존 LoRA 가중치에서 **새 학습**을 시작합니다.
중단된 동일 학습의 optimizer와 진행 상태를 복구하는 `--resume-from-checkpoint`와는 다르며 두 옵션은 같이 쓸 수 없습니다.
원본 어댑터와 결과 폴더는 별도로 지정해야 합니다. 기존 Q8 GGUF가 아닌 학습된 safetensors 어댑터가 필요합니다.
이 경로로 RTX 3050 4GB에서 카톡형 합성 초안 50건의 1에포크 추가 학습을 완료했습니다.
이는 한 번의 소규모 실험이며 요약 품질의 전반적인 향상이나 다른 환경에서의 동작을 보장하지 않습니다.
`data/chat_continuation_pilot/`은 학습용 6건, 검증용 2건의 수동 합성 예시로, 학습 품질을 판단할 만큼 충분하지 않습니다.
추가 사례를 검토하고 독립적인 평가셋 및 기존 사례와 섞은 데이터를 준비한 뒤 사용하세요.
기존 `long_chat_evaluation*.jsonl` 평가 사례를 학습에 넣으면 비교 평가에 사용할 수 없습니다.

## GGUF JSON 형식 제한 실험

`evaluate_gguf_local`에 `--json-schema-file briefing_training/briefing_response_schema.json`을
추가하면 llama-completion의 JSON 스키마 제한을 실험할 수 있습니다. 옵션을 생략하면 기존 방식입니다.
동일 모델·입력으로 두 번 평가하고 각각 다른 결과 파일을 지정하세요. 모델 생성 내용의 정확성을
보장하지 않으며, 실제 클라이언트 LLamaSharp에는 적용되지 않았습니다. 도구의 문맥 크기와 실행 방식은
클라이언트와 다르므로 앱에서 별도로 검증해야 합니다. 독립 평가 데이터
`data/category_briefing_v9/independent_chat_evaluation.jsonl`은 학습에 사용하지 않습니다.

## Qwen3-1.7B 비교 실험

기존 v8 0.6B와 별도로 실행합니다. 1.7B 기본 모델에는 v8 어댑터나 체크포인트를 사용할 수 없습니다.
첫 실행에는 모델 다운로드가 필요합니다. 4GB VRAM에서 추론·학습 가능 여부는 실제 확인해야 하며,
메모리 부족 시 학습을 중단하고 더 큰 GPU 환경을 사용하세요. 양자화·CPU 오프로딩은 속도와 품질에 영향을 줄 수 있습니다.
기본 모델 평가는 FP16으로 로드하므로 학습의 4비트 로딩과 메모리 요구량이 다릅니다.

```powershell
cd "C:\Users\akio7\OneDrive\바탕 화면\ALTONG_ai"
$briefingPython = "C:\dev\ALTONG_ai_train_v5\.venv\Scripts\python.exe"

# 기존 v8과 같은 평가 사례로 1.7B 기본 모델 확인
& $briefingPython -m briefing_training.evaluate --task briefing --greedy `
  --model-name "Qwen/Qwen3-1.7B" `
  --cases briefing_training/data/category_briefing_v9/conversation_evaluation.jsonl `
  --output outputs/qwen17b-base-conversation.json `
  --review-output outputs/qwen17b-base-conversation-review.md

# 데이터 형식만 확인하며 GPU 학습은 하지 않음
& $briefingPython -m briefing_training.train_lora --task briefing `
  --model-name "Qwen/Qwen3-1.7B" `
  --train-path briefing_training/data/category_briefing_v9/train_cases.jsonl `
  --validation-path briefing_training/data/category_briefing_v9/validation_cases.jsonl --validate-only

# 새 폴더에서 1.7B를 새로 학습. 4GB GPU에서 메모리 부족 가능.
& $briefingPython -m briefing_training.train_lora --task briefing `
  --model-name "Qwen/Qwen3-1.7B" `
  --train-path briefing_training/data/category_briefing_v9/train_cases.jsonl `
  --validation-path briefing_training/data/category_briefing_v9/validation_cases.jsonl `
  --output-dir "C:\dev\ALTONG_models\briefing-qwen3-1.7b-v9-local" `
  --epochs 3 --batch-size 1 --gradient-accumulation-steps 16 `
  --learning-rate 0.0001 --max-length 2048

# 학습 완료 후 같은 평가
& $briefingPython -m briefing_training.evaluate --task briefing --greedy `
  --model-name "Qwen/Qwen3-1.7B" `
  --cases briefing_training/data/category_briefing_v9/conversation_evaluation.jsonl `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen3-1.7b-v9-local" `
  --output outputs/qwen17b-v9-conversation.json `
  --review-output outputs/qwen17b-v9-conversation-review.md
```

기존 비교 결과 `outputs/v8-corrected-conversation-review.md`와 각 입력·요약·누락 사실을 비교합니다.
평가 점수는 원문 기반 카테고리 교정을 기본 적용하며 문장의 자연스러움과 부정 의미는 사람이 확인해야 합니다.
8건은 소규모 합성 사례입니다. 실제 테스트 사례는 개인정보를 제거하고 별도 평가셋으로 분리하세요.
학습 중 끊기면 저장이 완료된 1.7B checkpoint 경로를 `--resume-from-checkpoint`로 지정하고,
위의 모델·데이터·학습 설정과 output-dir을 그대로 유지합니다. 존재하는 비어 있지 않은 폴더에 새 학습은 거부합니다.
체크포인트 저장은 에포크 단위입니다. 완료 전에 중단하면 마지막 저장 이후 진행분은 돌아갑니다.
이 기능은 Qwen3 계열 실험용이며 Gemma 등 다른 계열의 템플릿·LoRA 대상 모듈 호환은 검증하지 않았습니다.

## 현재 모델과 실행 환경

v8은 통합 후보이며 기본 모델은 **Qwen/Qwen3-0.6B**입니다.
v9은 대화형 보완 데이터만 준비한 상태로, 아직 학습하거나 품질 향상을 확인하지 않았습니다.
모델 직접 출력은 `primary_category`, `summary_lines`입니다. 그룹화·일정 추출·대시보드 JSON 조립은 별도 Python 코드입니다.

노트북에서 확인한 환경: Python 3.12, PyTorch 2.7.1+cu118, Transformers 5.17.0,
PEFT 0.21.1, Accelerate 1.15.0. 공통 개발 환경과 학습 환경은 구분합니다.
기존 환경이 정상이라면 통합 직전에 라이브러리를 일괄 업그레이드하지 마세요.

GGUF 내보내기와 로컬 실행 검증은 [GGUF_EXPORT.md](GGUF_EXPORT.md)를 따릅니다.
Q4_K_M은 대화형 품질 저하가 확인되어 자동으로 배포 버전으로 선택하지 않습니다.
GGUF 자체에는 그룹화·일정 추출·분류 교정 Python 코드가 포함되지 않습니다.

```powershell
cd "C:\Users\akio7\OneDrive\바탕 화면\ALTONG_ai"
$briefingPython = "C:\dev\ALTONG_ai_train_v5\.venv\Scripts\python.exe"
```

## 실제 v8 실행

```powershell
& $briefingPython -m briefing_training.smoke_test_runtime_provider `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --notifications data/sample/briefing/category_notifications.json `
  --filter-results data/sample/briefing/category_filter_results.json `
  --output outputs/v8-dashboard.json --report-output outputs/v8-category-report.md
```

이 도구는 모델 실패를 규칙 기반 출력으로 숨기지 않습니다.
직접 호출 시 `QwenBriefingProvider(adapter_path=..., allow_fallback=False)`를
`SessionBriefingService(provider=...)`에 주입하고 `build(...).to_dict()`를 호출합니다.
서비스를 생략하면 학습 모델이 아니라 기본 규칙 기반으로 동작합니다.

## 평가 검토 파일 2개는 무엇인가요?

- `outputs/v8-category-evaluation-review.md`: 기본 30건의 원문, 참고 요약, 모델 출력과 누락 정보.
- `outputs/v8-short-review.md`: 짧은 요청 8건의 원문, 참고 요약, 모델 출력과 불필요한 표현 검사.

두 파일은 사람이 읽는 자료이고 모델 파일이 아닙니다.
대응하는 `v8-category-evaluation.json`, `v8-short-evaluation.json`에는 전체 평가 수치가 있습니다.
출력 폴더는 Git에 포함되지 않으므로 팀원에게 직접 공유하세요.

```powershell
& $briefingPython -m briefing_training.evaluate --task briefing --greedy `
  --cases briefing_training/data/category_briefing_v8/evaluation_cases.jsonl `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --output outputs/v8-category-evaluation.json --review-output outputs/v8-category-evaluation-review.md

& $briefingPython -m briefing_training.evaluate --task briefing --greedy `
  --cases briefing_training/data/category_briefing_v8/short_request_evaluation.jsonl `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --output outputs/v8-short-evaluation.json --review-output outputs/v8-short-review.md
```

카테고리·필수 사실·금지 표현·줄 수·JSON 형식을 함께 검사합니다.
단어 포함률은 의미 정확도를 보장하지 않습니다. 부정 의미·대상·최신 상태를 수동으로 확인하세요.
새 대화 평가의 `require_korean`은 전체 영어 출력을 검사하지만 혼합 언어의 자연스러움까지 보장하지 않습니다.

## v8 유지 + 카테고리 교정 (재학습 없음)

`QwenBriefingProvider`와 `evaluate --task briefing`은 기본으로 같은 원문 기반 교정 함수를 사용합니다.
여러 근거가 함께 일치하고 카테고리 후보가 하나일 때만 적용하며, 근거가 없거나 후보가 충돌하면
모델 판단을 유지합니다. 카카오톡 방 이름, 발신자, 정답 라벨, 필터링 점수, 생성 요약은 교정 근거가 아닙니다.
일정 추출·카테고리 보고서도 교정된 카테고리를 사용합니다. 요약 문장과 최종 대시보드 필드는 바꾸지 않습니다.

평가 JSON의 `raw_category_accuracy`는 모델 원본, `category_accuracy`는 교정 후 서비스 성능입니다.
`model_predicted_category`, `predicted_category`, `category_correction_rules`로 사례별 전후를 볼 수 있습니다.
`raw_case_pass_rate`와 `case_pass_rate`도 분리합니다. 교정 후 수치를 '모델 단독 정확도'라고 공유하지 마세요.
원본 모드만 평가하려면 `--no-category-correction`을 추가합니다. 기존 결과를 덮어쓰지 않도록 새 이름을 사용하세요.

```powershell
& $briefingPython -m briefing_training.evaluate --task briefing --greedy `
  --cases briefing_training/data/category_briefing_v9/conversation_evaluation.jsonl `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --output outputs/v8-corrected-conversation.json --review-output outputs/v8-corrected-conversation-review.md

& $briefingPython -m briefing_training.evaluate --task briefing --greedy `
  --cases briefing_training/data/category_briefing_v8/evaluation_cases.jsonl `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --output outputs/v8-corrected-category.json --review-output outputs/v8-corrected-category-review.md

& $briefingPython -m briefing_training.evaluate --task briefing --greedy `
  --cases briefing_training/data/category_briefing_v8/short_request_evaluation.jsonl `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --output outputs/v8-corrected-short.json --review-output outputs/v8-corrected-short-review.md
```

규칙은 `src/briefing/category_correction.py`입니다. 이미 본 실패 유형을 보완한 검사이므로
동일 사례에서의 점수 개선만으로 실제 카톡 일반화 성능을 입증할 수는 없습니다.
요약 의미 왜곡과 정보 누락은 교정하지 않습니다. 원문 수동 검토와 새로운 사례 평가가 필요합니다.
GGUF에는 Python 코드가 자동 포함되지 않습니다. 모델을 변환해도 이 교정은 클라이언트에서 별도로 적용해야 합니다.

## v9 대화형 보완

본인 활동이 아니라는 부정 의미, 야간 배치 중단, 응답 표 같은 제출물, 일정의 참가자·목적·방식,
한국어 예약 요약, 배송 대상, 완료 상태, 행사 안내문 검토의 일정 오탐을 다룹니다.
학습 96건·검증 24건을 더해 총 **1,376건·200건**, 새 평가 8건입니다.
기존 v8 데이터·모델·평가 결과는 보존합니다. 추가 사례는 합성 템플릿이며 실제 카톡 품질을 보장하지 않습니다.

```powershell
& $briefingPython -m briefing_training.prepare_conversation_repairs
& $briefingPython -m briefing_training.train_lora --task briefing `
  --train-path briefing_training/data/category_briefing_v9/train_cases.jsonl `
  --validation-path briefing_training/data/category_briefing_v9/validation_cases.jsonl --validate-only

& $briefingPython -m briefing_training.evaluate --task briefing --greedy `
  --cases briefing_training/data/category_briefing_v9/conversation_evaluation.jsonl `
  --adapter-path "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v8-local" `
  --output outputs/v8-conversation-evaluation.json --review-output outputs/v8-conversation-review.md
```

v8을 먼저 새 사례로 평가한 뒤, 추가 학습을 결정하면 아래 명령을 사용합니다.
기존 v8 체크포인트를 이어 쓰지 않습니다.

```powershell
& $briefingPython -m briefing_training.train_lora --task briefing `
  --train-path briefing_training/data/category_briefing_v9/train_cases.jsonl `
  --validation-path briefing_training/data/category_briefing_v9/validation_cases.jsonl `
  --output-dir "C:\dev\ALTONG_models\briefing-qwen-lora-conversation-v9-local" `
  --epochs 3 --batch-size 2 --gradient-accumulation-steps 8 --learning-rate 0.0001 --max-length 2048
```

v9도 동일한 기본 30건·짧은 요청 8건·새 대화 8건으로 평가하고 v8과 비교합니다.
학습 중에는 데이터와 프롬프트를 변경하지 않습니다. 중단 복구에만 같은 학습의 체크포인트와 동일 옵션을 사용합니다.

## 모델 전달과 이전 데이터

클라이언트 네이티브 연동 문서는 **GGUF 단일 파일**을 요구합니다.
이는 어댑터 폴더 전달과 다릅니다. [GGUF_EXPORT.md](GGUF_EXPORT.md)의 병합·변환 절차와 C# 계약 차이를 확인하세요.
현재 클라이언트 코드는 JSON 파일의 `items`를 읽지만 우리 출력은 `groups`입니다.
문서의 GGUF 구상과 현재 코드 구현을 구분하고 팀장과 연결 방식을 먼저 맞추세요.
양자화는 재학습이 아니지만 품질이 달라질 수 있어 변환 후 검증이 필요합니다.

- v8: `data/category_briefing_v8/`, 학습 1,280 / 검증 176 / 기본 평가 30 / 짧은 요청 8
- v7: `data/category_briefing/`, 학습 1,200 / 검증 160 / 평가 30
- v5/v6: 기존 `data/train_cases.jsonl`, `validation_cases.jsonl`, `evaluation_cases.jsonl`

v8 재생성 명령은 `python -m briefing_training.prepare_category_dataset`입니다.
분류 정책은 [CATEGORY_POLICY.md](CATEGORY_POLICY.md)에 있습니다. 평가 예측값에 맞춰 정답을 자동 수정하지 않습니다.
v5/v6 요약 전용 학습에는 `--task summary`, 런타임에는 `--contract summary`를 명시합니다.
새 환경 설치에는 학습용 `requirements-lora.txt`를 참고하고 PyTorch는 OS·GPU에 맞게 선택합니다.
모델 가중치·체크포인트·실제 개인 알림은 Git에 커밋하지 않습니다.
