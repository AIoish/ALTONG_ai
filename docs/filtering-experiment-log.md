# 필터링 모델 실험 기록

각 실험은 같은 형식으로 아래에 추가한다. 점수는 해당 실험의 테스트 분할에서만 비교한다. 서로 다른 데이터셋이나 분할의 점수는 직접 비교하지 않는다. 예시는 합성 데이터의 실제 모델 출력이며 정답 라벨과 구분한다.

## 2026-09-26 — 51건 데이터, LoRA 2단계 시험

- 목적: 데이터 준비 → 학습 → 평가 경로와 모델 출력 형식 확인. 성능 개선을 입증하는 실험은 아님.
- 기반 모델: `Qwen/Qwen3-0.6B`
- 데이터: 합성 51건. 학습 분할 41건, 검증 5건, 테스트 5건. 데이터 SHA-256: `06cdef89942060b5979abde56f19ccdead0b278e5b04e380ee97612e0795efa6`.
- 학습: LoRA SFT, rank 8, alpha 16, dropout 0.05, `q_proj`·`v_proj` 대상. BF16, 최대 길이 768, seed 42, 2단계, batch size 1. 학습 분할에 41건이 있지만 이번 실행에서 실제 파라미터 업데이트에 사용된 샘플은 2건이다. 검증 분할은 학습 중 사용하지 않았다.
- 장치: NVIDIA GeForce RTX 3060 Laptop GPU. 최종 학습 loss: 2.041.
- 산출물: `filtering_training/outputs/lora-smoke-51/adapter` (로컬, Git 제외).
- 평가: 테스트 5건에 대해 원본 모델과 학습 어댑터를 같은 조건으로 실행했다.

| 지표 | 원본 모델 | 2단계 LoRA |
| --- | ---: | ---: |
| 유효한 JSON | 5/5 | 5/5 |
| 통과/차단 정답 | 2/5 | 2/5 |
| 긴급도 정확히 일치 | 0/5 | 0/5 |
| 관련도 정확히 일치 | 1/5 | 1/5 |
| 카테고리 macro F1 | 0.0 | 0.0 |
| 차단해야 할 알림을 통과시킨 비율 | 3/3 | 3/3 |

테스트가 5건이라 통계적으로 신뢰할 수 없으며, 2단계 학습 후 개선도 관찰되지 않았다. 아래 JSON은 **학습 어댑터가 실제 생성한 값**이다. 알림 ID와 정답은 비교를 위해 따로 적었다. `ai_summary_reason`을 포함한 JSON 값은 수정하지 않았다.
각 입력 JSON은 공통 system 프롬프트를 제외한 실제 user 메시지다. 원본에서 생략된 `duration_seconds`와 `recent_processes`는 모델 입력에서 각각 0과 빈 목록으로 정규화된다.

### 출력 예시 1 — noti_004

**모델 입력 (user 메시지)**

```json
{
  "notification": {
    "id": "noti_004",
    "app_name": "Slack",
    "sender": "박민준",
    "title": "FilterResult 스키마 변경",
    "body": "relevance_score 필드 타입이 변경됐습니다. 현재 AI 코드도 수정해야 합니다.",
    "timestamp": "2026-09-21T09:20:00Z"
  },
  "context": {
    "active_process": "Code.exe",
    "window_title": "policy.py - ALTONG_ai - Visual Studio Code",
    "last_updated": "2026-09-21T09:19:57Z",
    "duration_seconds": 0,
    "recent_processes": []
  }
}
```

정답: 긴급 업무, 긴급도 4, 관련도 5, 통과. 실제 출력은 통과 결정만 같고 카테고리와 점수가 다르다.

```json
{
  "urgency_score": 3,
  "relevance_score": 4,
  "category": "일정/회의",
  "ai_summary_reason": "현재 AI 코드 수정이 필요하며, 알림 내용과 작업 주제가 일정/회의에 해당합니다."
}
```

### 출력 예시 2 — noti_048

**모델 입력 (user 메시지)**

```json
{
  "notification": {
    "id": "noti_048",
    "app_name": "ShoppingMall",
    "sender": "가상 쇼핑몰",
    "title": "한정 할인 10분 남음",
    "body": "관심 상품의 할인 행사가 10분 뒤 종료됩니다.",
    "timestamp": "2026-09-21T14:55:00Z"
  },
  "context": {
    "active_process": "Code.exe",
    "window_title": "train.py - Visual Studio Code",
    "last_updated": "2026-09-21T14:54:56Z",
    "duration_seconds": 65,
    "recent_processes": [
      "Code.exe",
      "WindowsTerminal.exe"
    ]
  }
}
```

정답: 광고/홍보, 긴급도 1, 관련도 1, 차단. 실제 출력은 할인 광고를 업무와 관련 있는 긴급 알림으로 잘못 판단해 통과시킨다.

```json
{
  "urgency_score": 4,
  "relevance_score": 5,
  "category": "일정/회의",
  "ai_summary_reason": "현재 작업은 'Code.exe'와 'WindowsTerminal.exe'가 활성 창을 사용하고 있으며, 'train.py' 파일이 실행 중입니다. 이 알림은 상품 할인 행사가 10분 남는 상황을 나타내며, 작업과 관련성이 높습니다."
}
```

### 출력 예시 3 — noti_003

**모델 입력 (user 메시지)**

```json
{
  "notification": {
    "id": "noti_003",
    "app_name": "Google",
    "sender": "Google Security",
    "title": "새로운 로그인 감지",
    "body": "새로운 Windows 기기에서 계정 로그인이 감지되었습니다.",
    "timestamp": "2026-09-21T09:15:00Z"
  },
  "context": {
    "active_process": "chrome.exe",
    "window_title": "Operating Systems Lecture - YouTube - Chrome",
    "last_updated": "2026-09-21T09:14:56Z",
    "duration_seconds": 0,
    "recent_processes": []
  }
}
```

정답: 시스템/보안, 긴급도 4, 관련도 1, 통과. 실제 출력은 통과 결정만 같고 카테고리·점수가 다르다.

```json
{
  "urgency_score": 3,
  "relevance_score": 4,
  "category": "일정/회의",
  "ai_summary_reason": "새로운 로그인 감지가 일정/회의 관련 작업에 영향을 미치며, 현재 작업과 일치합니다."
}
```

평가 결과 원본은 `filtering_training/outputs/evaluation/base_test_51.json`, 학습 어댑터는 `filtering_training/outputs/evaluation/adapter_test_51.json`, 예시 JSON은 `filtering_training/outputs/evaluation/adapter_examples_51.json`에 있다. 이 경로들은 Git에서 제외된다. 이 문서에 필요한 숫자와 예시를 보존한다.

다음 실험에서는 정답 라벨 검토 후 데이터 규모를 늘리고, 학습 단계 수와 평가 세트를 별도로 기록한다. 3,000건 데이터 제작 및 본학습은 아직 시작하지 않았다.

## 2026-09-27 — 실제 알림 유형과 다양한 상황을 섞은 평가 초안

- 목적: 실제 수집 알림의 짧은 문구·앱 표기·빈 발신자/본문 패턴을 일부 참고하면서 업무, 일정, 보안, 개인, 홍보 등 다양한 상황을 평가한다.
- 데이터: 합성 24건(현실형 14건, 다른 상황 10건), 카테고리별 3건. 실제 원본 문구와 개인 정보는 복사하지 않았다.
- 평가셋 SHA-256: `214aaa93050e93066986e72a79820a2e77b790ebd75d464619a2d387ba174561`. 기존 51건 학습 데이터와 알림 ID·문구가 겹치지 않는다.
- 모델: `Qwen/Qwen3-0.6B` 원본과 51건 학습 분할에서 2단계만 학습한 LoRA 어댑터. 이번 평가 전에 새 학습은 하지 않았다.
- 평가 조건: 동일한 24건, greedy 생성, thinking 비활성화, 최대 출력 192토큰.
- 상태: 라벨의 독립 검수 전인 **잠정 평가셋**이다. 이전 24건 초안의 입력과 SHA가 달라 점수를 직접 비교하지 않는다.

| 지표 | 원본 모델 | 2단계 LoRA |
| --- | ---: | ---: |
| 유효한 JSON | 24/24 | 24/24 |
| 통과/차단 정답 | 11/24 | 10/24 |
| 긴급도 정확히 일치 | 2/24 | 3/24 |
| 관련도 정확히 일치 | 2/24 | 2/24 |
| 카테고리 macro F1 | 0.028 | 0.028 |
| 긴급 알림 오차단 | 0/9 | 0/9 |
| 차단 대상 오통과 | 13/14 | 14/14 |

아래 세 출력은 **LoRA 어댑터의 실제 생성값**이다. 입력은 공통 system 프롬프트를 제외한 실제 user 메시지다. 세 사례는 현실형 알림, 다른 업무 알림, 홍보 알림을 각각 보여준다.

### 입력·출력 예시 1 — holdout_001

입력:

```json
{
  "notification": {
    "id": "holdout_001",
    "app_name": "Antigravity IDE",
    "sender": "",
    "title": "결제 API 필드 변경",
    "body": "오늘 병합 전 승인 코드 수정 필요",
    "timestamp": "2026-09-23T09:07:00Z"
  },
  "context": {
    "active_process": "Code.exe",
    "window_title": "payment_gateway.py - Visual Studio Code",
    "last_updated": "2026-09-23T09:06:56Z",
    "duration_seconds": 48,
    "recent_processes": [
      "Code.exe",
      "chrome.exe"
    ]
  }
}
```

정답: 긴급 업무, 긴급도 4, 관련도 5, PASS.

실제 출력:

```json
{
  "urgency_score": 4,
  "relevance_score": 5,
  "category": "일정/회의",
  "ai_summary_reason": "결제 API 필드 변경이 일정/회의에 관련되어 있으며, 현재 작업과 일치하는 앱(Visual Studio Code)이 활성화되어 있습니다."
}
```

판단 결과: 모델 PASS.

### 입력·출력 예시 2 — holdout_005

입력:

```json
{
  "notification": {
    "id": "holdout_005",
    "app_name": "GitHub",
    "sender": "가상 리뷰어",
    "title": "현재 PR 리뷰 승인",
    "body": "작성 중인 결제 모듈 PR을 승인했습니다. 추가 수정 요청은 없습니다.",
    "timestamp": "2026-09-23T09:35:00Z"
  },
  "context": {
    "active_process": "Code.exe",
    "window_title": "payment_gateway.py - Visual Studio Code",
    "last_updated": "2026-09-23T09:34:56Z",
    "duration_seconds": 32,
    "recent_processes": [
      "Code.exe",
      "chrome.exe"
    ]
  }
}
```

정답: 일반 업무, 긴급도 2, 관련도 5, BLOCK.

실제 출력:

```json
{
  "urgency_score": 3,
  "relevance_score": 4,
  "category": "일정/회의",
  "ai_summary_reason": "현재 작업과 PR 리뷰 승인 관련이며, 추가 수정 요청 없이 진행 중입니다."
}
```

판단 결과: 모델 PASS.

### 입력·출력 예시 3 — holdout_019

입력:

```json
{
  "notification": {
    "id": "holdout_019",
    "app_name": "Google Chrome",
    "sender": "",
    "title": "쿠폰 5분 후 만료",
    "body": "사용하지 않아도 불이익 없음",
    "timestamp": "2026-09-23T11:13:00Z"
  },
  "context": {
    "active_process": "Code.exe",
    "window_title": "search_index.py - Visual Studio Code",
    "last_updated": "2026-09-23T11:12:56Z",
    "duration_seconds": 45,
    "recent_processes": [
      "Code.exe",
      "chrome.exe"
    ]
  }
}
```

정답: 광고/홍보, 긴급도 2, 관련도 1, BLOCK.

실제 출력:

```json
{
  "urgency_score": 3,
  "relevance_score": 4,
  "category": "일정/회의",
  "ai_summary_reason": "이미 일정/회의 목록에 포함되어 있으며, 쿠폰 만료일이 5분 후 발생하므로 긴급 업무입니다."
}
```

판단 결과: 모델 PASS.

집계 결과는 `filtering_training/outputs/evaluation/base_mixed_holdout_24.json`과 `filtering_training/outputs/evaluation/adapter_mixed_holdout_24.json`에 있다. 모델 예측 전체는 Git에서 제외된 `filtering_training/outputs/evaluation/adapter_mixed_holdout_examples_24.json`에 저장했다.
## 2026-09-27 — 공개 알림 3,000건 번역 초안 시험

원천은 [NotifAI](https://huggingface.co/datasets/charlesfeng1/notifai-dataset)의
선별된 공개 합성 알림 3,000건이다. 번역 모델은
[M2M100 418M](https://huggingface.co/facebook/m2m100_418M)을 사용했다.
방식은 영어 제목과 본문을 따로 한국어로 일괄 번역하는 것이며,
필터링 모델 학습은 진행하지 않았다. 로컬 RTX 3060 Laptop GPU에서
모델 적재 후 3,000건 번역에 135초가 걸렸다.

자동 검사에서 98개 원천 ID에 검토 플래그가 붙었다.
빈 필드 22건, 숫자 누락 69건, 중복 번역 11건, 지나치게 짧은 번역
19건이며 한 건에 여러 사유가 붙을 수 있다. 직접 읽은 결과,
플래그가 없는 출력에도 기술 용어 오역과 문장 누락이 있어
**이 번역본은 학습 데이터로 승인하지 않았다.**
Qwen3-1.7B 재작성 4건도 일부 영어 원문을 그대로 출력하거나
거친 번역의 오역을 따라가 대체 수단으로 채택하지 않았다.

아래는 번역 단계의 실제 입력·출력이다. ALTONG 모델의 네 필드 출력이
아니며, 원문은 공개 합성 자료다.

### 입력·출력 예시 1 — source_id 09625

입력:

```json
{"app":"Linear","title":"TECH-542: Code review feedback","body":"Platform lead requested changes on the authentication refactor. 3 comments awaiting response."}
```

출력:

```json
{"title":"TECH-542 : 코드 검토 반응","body":"플랫폼 리더는 인증 재활용기 변경 요청했습니다. 3 코멘트 응답을 기다리고 있습니다."}
```

검토: `refactor`를 잘못 번역했다.

### 입력·출력 예시 2 — source_id 14034

입력:

```json
{"app":"Linear","title":"INFRA-42 blocked by INFRA-39","body":"Database migration task is waiting on deployment pipeline updates"}
```

출력:

```json
{"title":"INFRA-42는 INFRA-39에 의해 차단되었습니다.","body":"데이터베이스 이주 작업은 배치 파이프 라인 업데이트를 기다리고 있습니다."}
```

검토: `migration`과 `deployment`를 자연스럽게 옮기지 못했다.

### 입력·출력 예시 3 — source_id 12399

입력:

```json
{"app":"GitHub","title":"Sponsorship","body":"GitHub Sponsors: You're eligible! Set up your profile"}
```

출력:

```json
{"title":"스폰서","body":"GitHub 스폰서: 당신은 적합합니다!"}
```

검토: 프로필 설정 요청이 누락됐다.

전체 번역 초안과 오류 검토 목록은 각각 Git에서 제외된
`filtering_training/outputs/external/notifai/translated_candidates.jsonl`과
`filtering_training/outputs/audit/external_translation_audit.json`에 있다.
## 2026-09-27 — 한국어 후보 3,000건 빠른 생성

모델: 없음 (Codex가 작성한 한국어 상황 유형 80개를 결정적으로 조합).
학습 방식/학습 건수: 아직 학습하지 않음/0건. 생성 건수: 3,000건;
8개 범주에 각 375건. 제목·본문 조합은 3,000개이지만 독립적인 상황
유형은 80개다. 공개 [NotifAI](https://huggingface.co/datasets/charlesfeng1/notifai-dataset)의
넓은 알림 종류와 로컬 실제 알림의 형식만 참고했으며, 생성기는 두 원본을
읽거나 복사하지 않는다. 자동 형식 검사에서 오류가 없고 기존 소규모
학습/평가 데이터와 정확히 일치하는 알림은 없다. 문장 자연스러움과
라벨의 타당성은 아직 수동 검토 전이라 **학습용 확정 데이터가 아니다.**

아래 세 건은 생성된 파일에서 뽑은 실제 입력과 목표 출력이다.
모델의 예측값이 아니다. 입력의 `sender` 등 빈 값도 원래 JSON 형식대로 표시했다.

### 예시 1 — 긴급 업무, 현재 작업과 무관

입력:

```json
{"notification":{"id":"rapid_0001","app_name":"Slack","sender":"","title":"예약 서비스 요청 실패","body":"예약 서비스에 들어온 요청이 정상 처리되지 않습니다. 오늘 마감 작업에 영향이 있어 빠른 확인이 필요합니다.","timestamp":"2026-09-27T12:00:23Z"},"context":{"active_process":"EXCEL.EXE","window_title":"개인 가계부 - Excel","last_updated":"2026-09-27T12:00:20Z","duration_seconds":65,"recent_processes":["EXCEL.EXE"]}}
```

목표 출력:

```json
{"urgency_score":4,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"예약 서비스 요청 실패 알림은 현재 작업과 무관하며 현재 작업을 잠시 중단하고 확인할 가치가 있습니다."}
```

### 예시 2 — 개인 중요, 작업 정보 없음

입력:

```json
{"notification":{"id":"rapid_1571","app_name":"은행 앱","sender":"","title":"교통비 결제 처리 보류","body":"교통비 결제 처리가 보류돼 확인이 필요합니다. 오늘 안에 거래 내역을 확인해 주세요.","timestamp":"2026-09-27T22:02:13Z"},"context":{"active_process":"","window_title":"","last_updated":"2026-09-27T22:02:10Z","duration_seconds":0,"recent_processes":[]}}
```

목표 출력:

```json
{"urgency_score":3,"relevance_score":1,"category":"개인 중요","ai_summary_reason":"교통비 결제 처리 보류 알림은 현재 작업 정보가 없어 연관성을 알 수 없으며 비교적 빠른 확인이 필요합니다."}
```

### 예시 3 — 광고/홍보, 현재 작업과 관련

입력:

```json
{"notification":{"id":"rapid_2601","app_name":"쇼핑 앱","sender":"","title":"휴대전화 케이스 쿠폰","body":"휴대전화 케이스 구매에 적용할 쿠폰이 도착했습니다. 이번 주 안에 혜택을 확인할 수 있습니다.","timestamp":"2026-09-28T04:37:03Z"},"context":{"active_process":"chrome.exe","window_title":"휴대전화 케이스 상품 비교 - Chrome","last_updated":"2026-09-28T04:37:00Z","duration_seconds":25,"recent_processes":["chrome.exe","explorer.exe","KakaoTalk.exe"]}}
```

목표 출력:

```json
{"urgency_score":1,"relevance_score":4,"category":"광고/홍보","ai_summary_reason":"휴대전화 케이스 쿠폰 알림은 현재 열어 둔 작업과 직접 관련이 있으며 언제 확인해도 큰 문제가 없습니다."}
```

동일 상황 유형이 학습·검증·평가에 중복되지 않도록 계보 파일의 `scenario`
기준으로 분할했다. 임시 분할은 학습 2,398건, 검증 302건, 평가 300건이다.
별도의 24건 독립 평가 세트는 여기에 넣지 않았다. 두 파일과 분할 결과는
`filtering_training/outputs/` 아래에 있으며 Git에서 제외된다.
## 2026-09-27 — Qwen3-0.6B 로컬 LoRA 100/500단계 파일럿

모델: `Qwen/Qwen3-0.6B`; 방식: LoRA SFT(r=8, alpha=16, q_proj/v_proj), batch 1, 최대 768토큰, seed 42.
데이터: 한국어 생성 후보 3,000건 중 학습 분할 2,398건(80개 상황 유형 중 학습 64개).
100단계는 약 4.2%, 500단계는 약 20.9% epoch이며 전체 학습이 아니다.
검증: 별도 로컬 합성 평가셋 24건. 기준 모델과 두 어댑터 모두 같은 평가셋을 사용했다.

| 지표 | 원본 모델 | 100단계 | 500단계 |
| --- | ---: | ---: | ---: |
| 정책 정확도 | 45.8% | 58.3% | 62.5% |
| 긴급 알림 오차단율 | 0.0% | 100.0% | 88.9% |
| 불필요한 통과율 | 92.9% | 0.0% | 0.0% |
| 범주 macro F1 | 0.028 | 0.202 | 0.495 |
| 유효 JSON | 24/24 | 24/24 | 24/24 |

두 어댑터 모두 긴급 알림 대부분을 차단하므로 실사용 후보로 채택하지 않는다.
학습 손실은 100단계 0.5083, 500단계 0.1432였지만 긴급 알림 미탐이 남았다.
짧은 장애·배포 차단 문장과 명시적 긴급 단어가 없는 사례를 생성 데이터에 보강한 뒤 재평가한다.
아래는 동일한 세 입력에 대한 두 실험의 실제 JSON 출력이다. 공통 system 프롬프트는 생략했다.

### 입력·출력 예시 1 — holdout_001

입력:

```json
{"notification":{"id":"holdout_001","app_name":"Antigravity IDE","sender":"","title":"결제 API 필드 변경","body":"오늘 병합 전 승인 코드 수정 필요","timestamp":"2026-09-23T09:07:00Z"},"context":{"active_process":"Code.exe","window_title":"payment_gateway.py - Visual Studio Code","last_updated":"2026-09-23T09:06:56Z","duration_seconds":48,"recent_processes":["Code.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 구현 중인 결제 API의 필드가 바뀌어 병합 전에 수정해야 함"}
```

100단계 실제 출력:

```json
{"urgency_score":2,"relevance_score":1,"category":"일반 업무","ai_summary_reason":"현재 작업 분야와 간접적으로 관련이 있으며 나중에 확인해도 됩니다."}
```

500단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":3,"category":"개인 중요","ai_summary_reason":"현재 작업 분야와 간접적으로 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```

### 입력·출력 예시 2 — holdout_002

입력:

```json
{"notification":{"id":"holdout_002","app_name":"PagerDuty","sender":"가상 운영팀","title":"운영 작업 대기열 정지","body":"주문 처리 대기열이 15분째 멈춰 신규 주문이 처리되지 않습니다. 담당자 확인이 필요합니다.","timestamp":"2026-09-23T09:14:00Z"},"context":{"active_process":"PowerPoint.exe","window_title":"분기 매출 발표자료 - PowerPoint","last_updated":"2026-09-23T09:13:56Z","duration_seconds":72,"recent_processes":["PowerPoint.exe"]}}
```

정답:

```json
{"urgency_score":5,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"현재 발표 작업과 무관하지만 주문 처리 장애에 즉시 대응해야 함"}
```

100단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"현재 작업 분야와 간접적으로 관련이 있으며 나중에 확인해도 됩니다."}
```

500단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"현재 작업과 무관하며 비교적 빠른 확인이 필요합니다."}
```

### 입력·출력 예시 3 — holdout_003

입력:

```json
{"notification":{"id":"holdout_003","app_name":"Antigravity IDE","sender":"","title":"검색 API 배포 차단","body":"결과 누락 재현. 오늘 배포 전 수정","timestamp":"2026-09-23T09:21:00Z"},"context":{"active_process":"Code.exe","window_title":"search_index.py - Visual Studio Code","last_updated":"2026-09-23T09:20:56Z","duration_seconds":36,"recent_processes":["Code.exe","WindowsTerminal.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 작업 중인 검색 API의 오류로 배포 전에 확인해야 함"}
```

100단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":1,"category":"일반 업무","ai_summary_reason":"현재 작업 분야와 간접적으로 관련이 있으며 나중에 확인해도 됩니다."}
```

500단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":5,"category":"개인 중요","ai_summary_reason":"현재 열어 둔 작업과 직접 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```

집계 결과와 전체 출력은 Git에서 제외된 `filtering_training/outputs/evaluation/rapid_pilot_*`에 보존했다.

## 2026-09-27 — 긴급 알림 미탐 원인 비교: 데이터·프롬프트·모델 크기

모든 평가는 앞 절과 동일한 잠정 합성 평가셋 24건에서 수행했다. 모든 모델 출력 JSON은 유효했다.
긴급 알림은 9건, 정답이 BLOCK인 알림은 14건이다. 보강 데이터의 전체 후보는 3,000건이며 학습 분할은 2,398건이다.

| 실험 | 모델/방식 | 정책 정확도 | 긴급 오차단 | 불필요한 통과 | 범주 macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.6B 보강 500단계 | Qwen/Qwen3-0.6B: LoRA SFT 500단계, 새 한국어 후보 학습 2,398건 중 약 20.9% epoch | 66.7% | 77.8% | 0.0% | 0.588 |
| 0.6B 기준 프롬프트 | Qwen/Qwen3-0.6B: 학습 없음, 점수 정의를 시스템 프롬프트에 추가한 일시적 실험 | 41.7% | 0.0% | 100.0% | 0.028 |
| 1.7B 원본 | Qwen/Qwen3-1.7B: 학습 없음, 기존 시스템 프롬프트 | 70.8% | 0.0% | 50.0% | 0.087 |
| 1.7B 500단계 | Qwen/Qwen3-1.7B: LoRA SFT 500단계, 같은 보강 데이터 학습 2,398건 중 약 20.9% epoch | 79.2% | 55.6% | 0.0% | 0.574 |

보강된 짧은 업무 장애 사례는 0.6B의 긴급 오차단을 8/9에서 7/9로만 줄였다.
프롬프트 기준 추가는 기본 0.6B의 긴급 오차단을 없앴지만 BLOCK 14건 전부를 통과시켜 채택하지 않고 코드를 되돌렸다.
1.7B 원본은 긴급 미탐이 없지만 BLOCK 14건 중 7건을 통과시켰고, 1.7B 500단계는 이 7건을 막았으나 긴급 9건 중 5건을 놓쳤다.
따라서 어느 결과도 실사용 모델로 승인하지 않았다. 24건 평가의 표본 수와 잠정 라벨도 한계다.
아래 세 입력은 모든 실험에 공통이다. system 프롬프트는 생략했고, 각 실험의 실제 JSON 출력 세 건씩을 기록한다.

### 비교 입력·출력 예시 1 — holdout_001

입력:

```json
{"notification":{"id":"holdout_001","app_name":"Antigravity IDE","sender":"","title":"결제 API 필드 변경","body":"오늘 병합 전 승인 코드 수정 필요","timestamp":"2026-09-23T09:07:00Z"},"context":{"active_process":"Code.exe","window_title":"payment_gateway.py - Visual Studio Code","last_updated":"2026-09-23T09:06:56Z","duration_seconds":48,"recent_processes":["Code.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 구현 중인 결제 API의 필드가 바뀌어 병합 전에 수정해야 함"}
```

0.6B 보강 500단계 실제 출력:

```json
{"urgency_score":2,"relevance_score":3,"category":"일반 업무","ai_summary_reason":"결제 API 필드 변경 알림은 현재 작업 분야와 간접적으로 관련이 있으며 나중에 확인해도 됩니다."}
```

0.6B 기준 프롬프트 실제 출력:

```json
{"urgency_score":4,"relevance_score":4,"category":"일정/회의","ai_summary_reason":"결제 API 필드 변경이 병합 전 승인 코드 수정 필요로 인해 현재 작업과 직접 관련되어 있으며, 즉시 대응 필요"}
```

1.7B 원본 실제 출력:

```json
{"urgency_score":3,"relevance_score":4,"category":"긴급 업무","ai_summary_reason":"알림은 결제 API 필드 변경에 대한 정보이며, 현재 활성 창이 \u0027payment_gateway.py - Visual Studio Code\u0027로, 코드 수정이 필요하다고 명시되어 있으므로 긴급한 업무로 판단됨."}
```

1.7B 500단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":5,"category":"일반 업무","ai_summary_reason":"결제 API 필드 변경 알림은 현재 열어 둔 작업과 직접 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```

### 비교 입력·출력 예시 2 — holdout_002

입력:

```json
{"notification":{"id":"holdout_002","app_name":"PagerDuty","sender":"가상 운영팀","title":"운영 작업 대기열 정지","body":"주문 처리 대기열이 15분째 멈춰 신규 주문이 처리되지 않습니다. 담당자 확인이 필요합니다.","timestamp":"2026-09-23T09:14:00Z"},"context":{"active_process":"PowerPoint.exe","window_title":"분기 매출 발표자료 - PowerPoint","last_updated":"2026-09-23T09:13:56Z","duration_seconds":72,"recent_processes":["PowerPoint.exe"]}}
```

정답:

```json
{"urgency_score":5,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"현재 발표 작업과 무관하지만 주문 처리 장애에 즉시 대응해야 함"}
```

0.6B 보강 500단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"운영 작업 대기열 정지 알림은 현재 작업과 무관하며 비교적 빠른 확인이 필요합니다."}
```

0.6B 기준 프롬프트 실제 출력:

```json
{"urgency_score":4,"relevance_score":4,"category":"일정/회의","ai_summary_reason":"주문 처리 대기열이 15분째 멈춰 신규 주문이 처리되지 않습니다. 담당자 확인이 필요합니다."}
```

1.7B 원본 실제 출력:

```json
{"urgency_score":3,"relevance_score":4,"category":"긴급 업무","ai_summary_reason":"알림은 주문 처리 대기열이 15분째 멈춰 신규 주문이 처리되지 않는 문제를 나타내며, 현재 활성 창이 PowerPoint.exe로 열려 있으며, 이는 분기 매출 발표에 관련된 작업이 진행 중이므로 긴급한 대응이 필요합니다."}
```

1.7B 500단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"운영 작업 대기열 정지 알림은 현재 작업과 무관하며 비교적 빠른 확인이 필요합니다."}
```

### 비교 입력·출력 예시 3 — holdout_003

입력:

```json
{"notification":{"id":"holdout_003","app_name":"Antigravity IDE","sender":"","title":"검색 API 배포 차단","body":"결과 누락 재현. 오늘 배포 전 수정","timestamp":"2026-09-23T09:21:00Z"},"context":{"active_process":"Code.exe","window_title":"search_index.py - Visual Studio Code","last_updated":"2026-09-23T09:20:56Z","duration_seconds":36,"recent_processes":["Code.exe","WindowsTerminal.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 작업 중인 검색 API의 오류로 배포 전에 확인해야 함"}
```

0.6B 보강 500단계 실제 출력:

```json
{"urgency_score":3,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"검색 API 배포 차단 알림은 현재 열어 둔 작업과 직접 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```

0.6B 기준 프롬프트 실제 출력:

```json
{"urgency_score":4,"relevance_score":4,"category":"일정/회의","ai_summary_reason":"결과 누락 재현. 오늘 배포 전 수정"}
```

1.7B 원본 실제 출력:

```json
{"urgency_score":4,"relevance_score":3,"category":"긴급 업무","ai_summary_reason":"알림은 배포 전 수정이 필요한 결과 누락으로, 현재 활성 창인 \u0027search_index.py - Visual Studio Code\u0027에서 사용 중인 앱으로, 긴급한 수정이 필요합니다."}
```

1.7B 500단계 실제 출력:

```json
{"urgency_score":4,"relevance_score":5,"category":"시스템/보안","ai_summary_reason":"검색 API 배포 차단 알림은 현재 열어 둔 작업과 직접 관련이 있으며 현재 작업을 잠시 중단하고 확인할 가치가 있습니다."}
```

전체 모델 어댑터, 평가 보고서, 출력 예시는 `filtering_training/outputs/`에 로컬 보관하며 Git에 올리지 않는다.

평가셋 사용 주의: 같은 24건을 보고 데이터와 모델을 반복 조정했으므로 이후에는 개발 비교용으로 사용한다. 최종 성능 보고에는 새로 작성하고 검수한 미사용 평가셋을 별도로 준비한다.

## 2026-09-29 — Qwen3-1.7B, 학습 분할 전체 1회 학습

3,000건 후보 중 학습 2,398건을 2,398스텝으로 한 번 학습했다. 검증 302건과 테스트 300건은 이번 학습과 평가에 사용하지 않았다. 이번 비교 평가는 기존 개발용 합성 24건으로 실행했다.

- 모델: `Qwen/Qwen3-1.7B`, 기존 500스텝 어댑터를 이어 학습하지 않고 기반 모델에서 새 LoRA 학습.
- 데이터 SHA-256: `7aae83b590d4a5aeb1fc91d0aec40fae5fefb4f92be4e093beb938da22cbc378` (1.7B 500스텝 실험과 동일).
- 방식: LoRA SFT, BF16, rank 8, alpha 16, dropout 0.05, `q_proj`/`v_proj`, batch 1, accumulation 1, learning rate 0.0002, 최대 길이 768, seed 42. 응답 부분만 손실 계산.
- 장치: 로컬 NVIDIA GeForce RTX 3060 Laptop GPU (6GB).
- 학습 시간: 1,007초 (약 16분 47초). 평균 학습 loss: 0.0281197641.
- 평가: 기존 24건, 같은 프롬프트와 결정적 생성 설정 (`enable_thinking=false`, `do_sample=false`, 최대 출력 192토큰).
- 로컬 어댑터: `filtering_training/outputs/lora-qwen3-1_7b-epoch1-20260928/adapter`.
- 로컬 결과: `filtering_training/outputs/evaluation/qwen3_1_7b_epoch1_holdout_24.json`, 전체 24개 출력은 `qwen3_1_7b_epoch1_predictions_24.json`.

| 지표 | 기존 500스텝 | 전체 1회 (2,398스텝) |
| --- | ---: | ---: |
| 유효 JSON | 24/24 | 24/24 |
| 통과·차단 정답 | 19/24 (79.2%) | 17/24 (70.8%) |
| 긴급 알림 오차단 | 5/9 | 6/9 |
| 불필요한 알림 오통과 | 0/14 | 0/14 |
| 긴급도 정확히 일치 | 12/24 | 11/24 |
| 관련도 정확히 일치 | 16/24 | 12/24 |
| 카테고리 macro F1 | 0.5735 | 0.6256 |
| 긴급도 평균 절대 오차 | 0.5833 | 0.6667 |
| 관련도 평균 절대 오차 | 0.5000 | 0.9167 |

학습 loss가 낮아졌지만 정책 정확도와 긴급 알림 통과가 악화됐다. 이 개발 비교 기준으로는 전체 1회 어댑터를 기존 500스텝 어댑터 대신 채택하지 않는다. 두 어댑터 모두 실사용 품질을 확보하지 못했다. 스텝 수 변경으로 학습률 감소 일정도 바뀌므로 학습량만의 효과를 분리한 실험은 아니다.

정책 오답 7건은 모두 오차단이다. 현재 코드와 알림의 직접 관련성, 현재 작업과 무관한 운영 장애, 곧 시작할 회의, 관련 발표 순서 변경, 보안 보호 해제, 납부 실패, 임박한 가족 일정에서 긴급도 또는 관련도를 낮게 매겼다. 24건은 이미 개발 판단에 반복 사용했으므로 최종 일반화 성능으로 해석하지 않는다.

다음 추가 2,000건은 8개 카테고리별 250건으로 구성한다. 업무 사례는 IT 중심으로 유지하고, 새 상황·짧은 메시지·마감과 영향의 차이·직접/간접/무관한 맥락을 함께 보강한다. 평가 문장을 그대로 복제하지 않으며, 늘린 데이터에는 별도 상황 계보를 기록하고 분할 간 상황 중복을 검사한다. 추가 데이터에서 3,000건에 없던 상황 다양성을 늘리는 것이 목적이다. 최종 평가는 별도의 미사용 평가셋으로 진행한다.

아래는 이번 모델의 실제 입력과 출력이다. 정답과 모델 출력을 구분하며 출력 문구를 수정하지 않았다.

### 입력·출력 예시 1 — holdout_001

입력 (공통 system 프롬프트를 제외한 user 데이터):

```json
{
    "notification":  {
                         "id":  "holdout_001",
                         "app_name":  "Antigravity IDE",
                         "sender":  "",
                         "title":  "결제 API 필드 변경",
                         "body":  "오늘 병합 전 승인 코드 수정 필요",
                         "timestamp":  "2026-09-23T09:07:00Z"
                     },
    "context":  {
                    "active_process":  "Code.exe",
                    "window_title":  "payment_gateway.py - Visual Studio Code",
                    "last_updated":  "2026-09-23T09:06:56Z",
                    "duration_seconds":  48,
                    "recent_processes":  [
                                             "Code.exe",
                                             "chrome.exe"
                                         ]
                }
}
```

정답:

```json
{
    "urgency_score":  4,
    "relevance_score":  5,
    "category":  "긴급 업무",
    "ai_summary_reason":  "현재 구현 중인 결제 API의 필드가 바뀌어 병합 전에 수정해야 함"
}
```

실제 모델 출력:

```json
{
    "urgency_score":  3,
    "relevance_score":  2,
    "category":  "일반 업무",
    "ai_summary_reason":  "결제 API 필드 변경 알림은 현재 작업 분야와 간접적으로 관련이 있으며 비교적 빠른 확인이 필요합니다."
}
```

### 입력·출력 예시 2 — holdout_002

입력 (공통 system 프롬프트를 제외한 user 데이터):

```json
{
    "notification":  {
                         "id":  "holdout_002",
                         "app_name":  "PagerDuty",
                         "sender":  "가상 운영팀",
                         "title":  "운영 작업 대기열 정지",
                         "body":  "주문 처리 대기열이 15분째 멈춰 신규 주문이 처리되지 않습니다. 담당자 확인이 필요합니다.",
                         "timestamp":  "2026-09-23T09:14:00Z"
                     },
    "context":  {
                    "active_process":  "PowerPoint.exe",
                    "window_title":  "분기 매출 발표자료 - PowerPoint",
                    "last_updated":  "2026-09-23T09:13:56Z",
                    "duration_seconds":  72,
                    "recent_processes":  [
                                             "PowerPoint.exe"
                                         ]
                }
}
```

정답:

```json
{
    "urgency_score":  5,
    "relevance_score":  1,
    "category":  "긴급 업무",
    "ai_summary_reason":  "현재 발표 작업과 무관하지만 주문 처리 장애에 즉시 대응해야 함"
}
```

실제 모델 출력:

```json
{
    "urgency_score":  3,
    "relevance_score":  1,
    "category":  "긴급 업무",
    "ai_summary_reason":  "운영 작업 대기열 정지 알림은 현재 작업과 무관하며 비교적 빠른 확인이 필요합니다."
}
```

### 입력·출력 예시 3 — holdout_003

입력 (공통 system 프롬프트를 제외한 user 데이터):

```json
{
    "notification":  {
                         "id":  "holdout_003",
                         "app_name":  "Antigravity IDE",
                         "sender":  "",
                         "title":  "검색 API 배포 차단",
                         "body":  "결과 누락 재현. 오늘 배포 전 수정",
                         "timestamp":  "2026-09-23T09:21:00Z"
                     },
    "context":  {
                    "active_process":  "Code.exe",
                    "window_title":  "search_index.py - Visual Studio Code",
                    "last_updated":  "2026-09-23T09:20:56Z",
                    "duration_seconds":  36,
                    "recent_processes":  [
                                             "Code.exe",
                                             "WindowsTerminal.exe",
                                             "chrome.exe"
                                         ]
                }
}
```

정답:

```json
{
    "urgency_score":  4,
    "relevance_score":  5,
    "category":  "긴급 업무",
    "ai_summary_reason":  "현재 작업 중인 검색 API의 오류로 배포 전에 확인해야 함"
}
```

실제 모델 출력:

```json
{
    "urgency_score":  4,
    "relevance_score":  5,
    "category":  "긴급 업무",
    "ai_summary_reason":  "검색 API 배포 차단 알림은 현재 열어 둔 작업과 직접 관련이 있으며 현재 작업을 잠시 중단하고 확인할 가치가 있습니다."
}
```

## 2026-09-29 — 오류 유형 보강, 로컬 5,000행 준비 (모델 학습 전)

기존 3,000행에 새 2,000행을 추가했다. 새 상황 40종 × 대상 5개 × 의미가 다른 상태 5개 × 작업 맥락 2개로 생성했다. 추가분은 알림 내용 1,000개와 맥락 대조쌍 2,000행이며 전체는 고유 알림 내용 4,000개, 상황 유형 120종, 입력 5,000행이다. 5,000개의 독립적인 상황을 만든 것은 아니다.

- 기반/예정 모델: `Qwen/Qwen3-1.7B`, LoRA SFT. 이번 단계에서는 모델을 학습하거나 추론하지 않았다.
- 각 카테고리: 추가 250행, 전체 625행. 업무 사례는 IT 개발·운영 중심이며 개인·광고·기타도 함께 보강했다.
- 맥락 대조쌍: 같은 알림의 긴급도와 카테고리는 유지하고 관련도만 바꾼다. 직접 관련, 같은 분야, 무관한 작업, 맥락 없음, 최근 앱 정보만 있는 경우를 포함한다. 동일 짝은 같은 데이터 분할에만 속한다.
- 긴급 근거: 실제 운영 요청 실패·데이터 손상·보호 실패·금전 피해·임박한 참석/승인 마감을 구분한다. 현재 작업과 무관해도 긴급도를 낮추지 않는다.
- 경계 사례: 정상 빌드 성공은 일반 업무, 보안/장치 오류는 시스템/보안으로 구분한다. 선택형 광고나 취미 이벤트의 마감, 정상 완료 알림의 '지금'은 실제 피해/필수 대응과 구분한다.
- 라벨 이유는 알림의 마감·영향 근거와 맥락의 관련성을 함께 적었다. 체류 시간·최근 앱 개수로 점수를 추측하지 않도록 그 값도 다양화했다.
- 실제 개인 알림을 읽거나 복사하지 않았다. 평가 원문을 생성 템플릿으로 쓰지 않았다. 공개 데이터는 이전 문서에 기록한 유형 참고만 유지하며 이번 추가분에는 외부 원문 인용이 없다.
- 자동 검사: ID 중복 없음, 동일 입력의 상충 라벨 없음, 같은 알림의 긴급도/카테고리 상충 없음, 최근 프로세스 최대 3개. 기존 분할의 모든 ID와 상황 배정을 보존했고 분할 간 상황 중복도 없음. 자동 검사 60개 통과.
- 토큰 길이: Qwen3-1.7B 토크나이저로 추가 2,000행의 입력+정답을 확인했으며 최대 515토큰이다. 학습 3,598행도 최대 길이 768에서 잘림 없이 준비됨을 확인했다.
- 분할: 학습 3,598행, 검증 702행, 테스트 700행. 기존 3,000행의 검증/테스트 상황은 학습에 넣지 않았다. 새 상황 중 카테고리별 한 상황씩 검증/테스트에 배정했다.
- 문장/라벨 검토: 대표 사례를 읽고 개인 납부·방문 대상과 근거 문장을 수정했다. 전체 라벨에 대한 독립 검수는 미완료이며 자동 검사로 라벨의 의미 정확성을 보장하지 않는다.
- 기존 개발 24건은 별도 유지한다. 성능 향상은 아직 측정하지 않았으며 재학습과 미사용 평가로 확인해야 한다. 전체 1회 학습에서 악화됐으므로 다음 실험은 학습 분량/체크포인트도 함께 비교한다.

로컬 파일:

- 추가분: `filtering_training/outputs/candidates/targeted_korean_2000.jsonl`
- 통합본: `filtering_training/outputs/candidates/combined_korean_5000.jsonl`
- 각 `.lineage.json`: 상황 유형, 맥락 짝, 생성 근거 기록
- 학습 준비: `filtering_training/outputs/prepared_targeted_5000/`
- 검사 보고서: `filtering_training/outputs/audit/targeted_5000_audit.json`

아래 예시는 **새 데이터의 입력과 작성한 정답 라벨**이다. 학습된 모델의 출력이 아니다.

통합 데이터 SHA-256: `489f749adefaed0a1e62b767daafe47c23ce7503e73afd067f819d430a61ce33`.

전체 정책 분포: PASS 1543, BLOCK 3457.

### 데이터 예시 — targeted_0001

입력:

```json
{"notification":{"id":"targeted_0001","app_name":"Sentry","sender":"","title":"상품 API 운영 할당량 초과","body":"상품 API 호출 한도 소진으로 실제 사용자 요청이 거절됩니다.","timestamp":"2026-09-29T00:00:00Z"},"context":{"active_process":"Code.exe","window_title":"상품 API 핸들러 수정 - Visual Studio Code","last_updated":"2026-09-28T23:59:58Z","duration_seconds":43,"recent_processes":[]}}
```

작성한 정답 라벨:

```json
{"urgency_score":5,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"사용자 요청이 거절되는 운영 장애. 현재 창 제목에 같은 대상이 있어 직접 관련됨."}
```

### 데이터 예시 — targeted_0002

입력:

```json
{"notification":{"id":"targeted_0002","app_name":"Sentry","sender":"","title":"상품 API 운영 할당량 초과","body":"상품 API 호출 한도 소진으로 실제 사용자 요청이 거절됩니다.","timestamp":"2026-09-29T00:00:17Z"},"context":{"active_process":"EXCEL.EXE","window_title":"가계부 정리 - Excel","last_updated":"2026-09-29T00:00:15Z","duration_seconds":180,"recent_processes":["EXCEL.EXE"]}}
```

작성한 정답 라벨:

```json
{"urgency_score":5,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"사용자 요청이 거절되는 운영 장애. 현재 다른 분야의 작업을 하고 있어 관련성이 낮음."}
```

### 데이터 예시 — targeted_0751

입력:

```json
{"notification":{"id":"targeted_0751","app_name":"Windows 보안","sender":"","title":"개발 노트북 의심 파일 격리","body":"개발 노트북에서 악성 파일 실행 시도가 계속됩니다. 격리 실패로 보호 조치가 필요합니다.","timestamp":"2026-09-29T03:32:30Z"},"context":{"active_process":"SystemSettings.exe","window_title":"개발 노트북 장치 점검 - 설정","last_updated":"2026-09-29T03:32:28Z","duration_seconds":600,"recent_processes":[]}}
```

작성한 정답 라벨:

```json
{"urgency_score":5,"relevance_score":5,"category":"시스템/보안","ai_summary_reason":"악성 파일 실행이 이어지고 격리도 실패. 현재 창 제목에 같은 대상이 있어 직접 관련됨."}
```

### 데이터 예시 — targeted_1501

입력:

```json
{"notification":{"id":"targeted_1501","app_name":"쇼핑 앱","sender":"","title":"SSD 타임 세일","body":"SSD 할인 3분 남음! 선택 구매 이벤트이며 기존 주문에는 영향 없습니다.","timestamp":"2026-09-29T07:05:00Z"},"context":{"active_process":"chrome.exe","window_title":"SSD 구매 비교 - Chrome","last_updated":"2026-09-29T07:04:58Z","duration_seconds":600,"recent_processes":["chrome.exe","explorer.exe"]}}
```

작성한 정답 라벨:

```json
{"urgency_score":1,"relevance_score":4,"category":"광고/홍보","ai_summary_reason":"할인 마감이며 실제 의무나 기존 주문 영향 없음. 현재 창 제목에 같은 대상이 있어 직접 관련됨."}
```

## 2026-09-30 — Qwen3-4B-Instruct-2507, 학습 전 4비트 비교

추가 학습 없이 Qwen3-4B-Instruct-2507을 기존 개발용 합성 24건에 평가했다. 같은 데이터 SHA-256, system 프롬프트 SHA-256, 최대 출력 192토큰, 생각 모드 비활성화, 결정적 생성 조건을 사용했다. 1.7B 기존 결과는 BF16이며 4B는 NF4이므로 크기만의 효과를 분리한 비교는 아니다. 모델의 지시 튜닝 버전도 다르다.

- 기반 모델: `Qwen/Qwen3-4B-Instruct-2507`.
- 실제 배포본: [Unsloth의 Qwen3-4B-Instruct-2507-bnb-4bit](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-bnb-4bit). 학습 어댑터 없음, 이번 프로젝트 학습 데이터 사용 0건.
- 배포본 revision: `f12db89cd5156e090618dded9b4367f23f8f3b33`.
- 가중치 SHA-256: `f632ab95da1d8f307237094aca7c7dfa9e416b6d2f87bdd8ecda6dadfedb204d`. 공개 LFS 체크섬과 일치함을 확인했다. 가중치 파일 크기 2,653,133,894바이트.
- 양자화: bitsandbytes NF4, 이중 양자화, 배포본의 실제 연산 dtype BF16. 로더에서 나머지 dtype은 FP16으로 지정했다. 임베딩·출력층 등 배포본의 제외 모듈은 4비트로 바꾸지 않았다.
- 장치: 로컬 NVIDIA GeForce RTX 3060 Laptop GPU 6GB. 전체 CUDA device 0에 로드했으며 CPU offload 없음.
- 전송: 원본 Xet/일반 다운로드가 정체돼 사전 양자화 배포본의 파일을 작은 HTTP 구간으로 내려받고 체크섬을 검증했다. 다운로드 실패를 추론 실패로 집계하지 않았다.
- 데이터 SHA-256: `214aaa93050e93066986e72a79820a2e77b790ebd75d464619a2d387ba174561`.
- 프롬프트 SHA-256: `0820a8b9405212299385b5b9838cfb0c55650b0f6b4b3b5b6e73b5d0a8f82a63`.

| 지표 | 1.7B 기본 BF16 | 1.7B 500스텝 LoRA BF16 | 4B Instruct 기본 NF4 |
| --- | ---: | ---: | ---: |
| 유효 JSON | 24/24 | 24/24 | 24/24 |
| 통과·차단 정답 | 17/24 (70.8%) | 19/24 (79.2%) | 17/24 (70.8%) |
| 긴급 알림 오차단 | 0/9 | 5/9 | 1/9 |
| 불필요한 알림 오통과 | 7/14 | 0/14 | 6/14 |
| 긴급도 정확히 일치 | 8/24 | 12/24 | 10/24 |
| 관련도 정확히 일치 | 3/24 | 16/24 | 5/24 |
| 카테고리 macro F1 | 0.0871 | 0.5735 | 0.3149 |
| 긴급도 평균 절대 오차 | 1.0000 | 0.5833 | 0.7917 |
| 관련도 평균 절대 오차 | 1.9167 | 0.5000 | 1.2917 |

4B는 1.7B 기본 모델보다 카테고리와 점수 오차가 개선됐지만 정책 정답 수는 같았다. 현재 프로젝트 기준에 맞춘 학습 전 모델이므로 이 결과만으로 4B 채택을 결정하지 않는다. 기존 1.7B 학습 모델보다 긴급 오차단은 적지만 불필요한 통과는 많다. 현재 기본 모델 설정은 1.7B로 유지했다.

4B 정책 오답은 긴급 오차단 1건과 불필요한 통과 6건이다. 시작이 임박한 회의 변경을 낮은 긴급도로 평가했고, 가벼운 문서 수정·리뷰 승인·점심 투표·과거 회의 기록·선택형 기술 홍보·집중 시간 종료의 긴급도 또는 관련도를 높게 평가했다. 점수 기준에 대한 보정과 업무 관련성/긴급성 분리가 여전히 필요하다.

다음에는 4B의 소규모 QLoRA 학습이 6GB에 들어가는지 먼저 확인한 뒤, 확장 데이터로 학습해 같은 개발 기준에서 비교한다. 학습량은 검증 데이터로 선택하고 최종 성능은 미사용 평가 데이터로 확인한다. 현재 24건은 반복 사용한 개발 비교이며 독립적인 최종 성능 증거가 아니다.

산출물은 모두 Git 제외된 로컬 경로에 보관한다:

- 체크포인트: `filtering_training/outputs/models/qwen3-4b-instruct-2507-bnb-4bit/`
- 평가 보고서: `filtering_training/outputs/evaluation/qwen3_4b_instruct_nf4_dev24.json`
- 전체 실제 응답 24개: `filtering_training/outputs/evaluation/qwen3_4b_instruct_nf4_predictions24.json`

아래는 이번 모델이 실제 생성한 출력이다. 정답과 구분하고 문구를 수정하지 않았다.

생성 시간: 알림당 평균 6.53초, 전체 156.66초. 모델 로딩·입력 토큰화·응답 파싱은 제외하며 첫 생성도 포함한다.

메모리: 모델 footprint 2.42GiB, 생성 중 CUDA peak allocated 2.63GiB, peak reserved 2.74GiB. nvidia-smi 관측은 약 2,923MiB였다. CUDA 통계는 일부 드라이버 메모리와 로딩 순간의 최대치를 포함하지 않는다.

### 입력·출력 예시 — holdout_001

입력 (공통 system 프롬프트를 제외한 user 데이터):

```json
{"notification":{"id":"holdout_001","app_name":"Antigravity IDE","sender":"","title":"결제 API 필드 변경","body":"오늘 병합 전 승인 코드 수정 필요","timestamp":"2026-09-23T09:07:00Z"},"context":{"active_process":"Code.exe","window_title":"payment_gateway.py - Visual Studio Code","last_updated":"2026-09-23T09:06:56Z","duration_seconds":48,"recent_processes":["Code.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 구현 중인 결제 API의 필드가 바뀌어 병합 전에 수정해야 함"}
```

실제 모델 출력:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"결제 API 필드 변경과 관련된 승인 코드 수정이 필요하다는 내용이 현재 개발 작업 중인 파일과 직접 연결되어 있어 긴급 업무로 분류됨"}
```

### 입력·출력 예시 — holdout_007

입력 (공통 system 프롬프트를 제외한 user 데이터):

```json
{"notification":{"id":"holdout_007","app_name":"Calendar","sender":"가상 일정 앱","title":"회의 접속 링크 변경","body":"2분 뒤 시작할 정기 회의의 접속 링크가 바뀌었습니다. 새 링크를 사용해 주세요.","timestamp":"2026-09-23T09:49:00Z"},"context":{"active_process":"Excel.exe","window_title":"개인 지출 내역.xlsx - Excel","last_updated":"2026-09-23T09:48:56Z","duration_seconds":40,"recent_processes":[]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":1,"category":"일정/회의","ai_summary_reason":"현재 작업과 무관하지만 회의 시작이 임박해 새 링크를 확인해야 함"}
```

실제 모델 출력:

```json
{"urgency_score":3,"relevance_score":2,"category":"일정/회의","ai_summary_reason":"회의 접속 링크가 변경되었음을 알리는 알림이지만, 현재 활성 창이 Excel이므로 회의 관련 업무와 직접적인 연결은 없음"}
```

### 입력·출력 예시 — holdout_020

입력 (공통 system 프롬프트를 제외한 user 데이터):

```json
{"notification":{"id":"holdout_020","app_name":"Webinar","sender":"가상 교육 업체","title":"검색 기술 강연 신청","body":"검색 서비스 설계 온라인 강연 신청을 받고 있습니다. 다음 달 참석 여부를 결정해 주세요.","timestamp":"2026-09-23T11:20:00Z"},"context":{"active_process":"Code.exe","window_title":"search_index.py - Visual Studio Code","last_updated":"2026-09-23T11:19:56Z","duration_seconds":67,"recent_processes":["Code.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":1,"relevance_score":3,"category":"광고/홍보","ai_summary_reason":"현재 검색 개발 주제와 간접적으로 관련된 홍보지만 즉시 확인할 필요는 없음"}
```

실제 모델 출력:

```json
{"urgency_score":3,"relevance_score":4,"category":"일반 업무","ai_summary_reason":"현재 활성 창이 개발 환경인 Visual Studio Code로, 검색 기술 관련 강연 신청 알림은 개발 업무와 관련이 있어 일반 업무 범주에 해당합니다."}
```

## 2026-09-30 — Qwen3-4B QLoRA 2스텝, 저장·재로딩 시험

체크섬을 검증한 로컬 Unsloth NF4 배포본을 고정하고 어댑터만 학습했다.
전체 확장 데이터 5,000행 중 학습 분할 3,598행을 준비했으며 이번 업데이트에는
2스텝만 사용했다. 검증·테스트 분할은 학습에 넣지 않았다.

- 모델: Qwen3-4B-Instruct-2507, NF4 이중 양자화, BF16 계산.
- 방식: PEFT k-bit 학습 준비 + QLoRA SFT. rank 8, alpha 16, dropout 0.05,
  `all-linear` 대상, 학습률 0.0001, batch 1, accumulation 1, 최대 길이 768,
  seed 42, gradient checkpointing, 응답 부분만 손실 계산.
- 기반 모델은 고정하고 이름에 `lora_`가 들어간 파라미터만 학습 가능한지 검사했다.
  학습 파라미터 16,515,072개. 일반 LoRA와 구별해 실제 양자화 설정을 기록했다.
- 장치: 로컬 RTX 3060 Laptop GPU 6GB. 학습 시간 3.4817초, 평균 loss 1.8635051.
  peak CUDA allocated 4.79GiB, reserved 4.95GiB. 로딩 순간의 최대치는 포함하지 않는다.
- 어댑터: `filtering_training/outputs/qlora-qwen3-4b-smoke-20260930/adapter`.
- 별도 프로세스에서 같은 4비트 기반 모델과 어댑터를 재로딩했다.
  개발 24건 중 첫 3건에서 유효 JSON 3/3, 정책 정답 3/3을 확인했다.
  세 사례가 모두 긴급 업무이며 학습도 2스텝뿐이므로 성능 개선 증거로 해석하지 않는다.
- 집계: `filtering_training/outputs/evaluation/qlora_qwen3_4b_smoke3.json`.
  실제 응답: 같은 폴더의 `qlora_qwen3_4b_smoke3_examples.json`.
  모델과 결과는 Git 제외된 로컬 경로에 보관한다.

데이터 SHA-256: `489f749adefaed0a1e62b767daafe47c23ce7503e73afd067f819d430a61ce33`.

### 재로딩 입력·출력 — holdout_001

입력:

```json
{"notification":{"id":"holdout_001","app_name":"Antigravity IDE","sender":"","title":"결제 API 필드 변경","body":"오늘 병합 전 승인 코드 수정 필요","timestamp":"2026-09-23T09:07:00Z"},"context":{"active_process":"Code.exe","window_title":"payment_gateway.py - Visual Studio Code","last_updated":"2026-09-23T09:06:56Z","duration_seconds":48,"recent_processes":["Code.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 구현 중인 결제 API의 필드가 바뀌어 병합 전에 수정해야 함"}
```

실제 모델 출력:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"결제 API 필드 변경과 관련된 승인 코드 수정이 필요하며 현재 개발 환경에서 진행 중인 작업과 직접적으로 연결되어 있습니다."}
```

### 재로딩 입력·출력 — holdout_002

입력:

```json
{"notification":{"id":"holdout_002","app_name":"PagerDuty","sender":"가상 운영팀","title":"운영 작업 대기열 정지","body":"주문 처리 대기열이 15분째 멈춰 신규 주문이 처리되지 않습니다. 담당자 확인이 필요합니다.","timestamp":"2026-09-23T09:14:00Z"},"context":{"active_process":"PowerPoint.exe","window_title":"분기 매출 발표자료 - PowerPoint","last_updated":"2026-09-23T09:13:56Z","duration_seconds":72,"recent_processes":["PowerPoint.exe"]}}
```

정답:

```json
{"urgency_score":5,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"현재 발표 작업과 무관하지만 주문 처리 장애에 즉시 대응해야 함"}
```

실제 모델 출력:

```json
{"urgency_score":4,"relevance_score":3,"category":"긴급 업무","ai_summary_reason":"주문 처리 대기열이 15분간 멈춰 신규 주문이 처리되지 않아 운영 팀의 즉각적인 개입이 필요합니다."}
```

### 재로딩 입력·출력 — holdout_003

입력:

```json
{"notification":{"id":"holdout_003","app_name":"Antigravity IDE","sender":"","title":"검색 API 배포 차단","body":"결과 누락 재현. 오늘 배포 전 수정","timestamp":"2026-09-23T09:21:00Z"},"context":{"active_process":"Code.exe","window_title":"search_index.py - Visual Studio Code","last_updated":"2026-09-23T09:20:56Z","duration_seconds":36,"recent_processes":["Code.exe","WindowsTerminal.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 작업 중인 검색 API의 오류로 배포 전에 확인해야 함"}
```

실제 모델 출력:

```json
{"urgency_score":4,"relevance_score":4,"category":"긴급 업무","ai_summary_reason":"검색 API 배포 차단 문제는 현재 개발 작업 중인 코드 실행과 직접적으로 관련되어 있으며, 배포 전 수정이 필요하기 때문에 긴급도와 관련성이 높습니다."}
```

## 2026-09-30 — Qwen3-4B QLoRA 500스텝, 확장 데이터 실험

4B 기반 모델에서 새 QLoRA 어댑터를 학습했다. 2스텝 스모크 어댑터를 이어 학습한 것이 아니다. 로컬 5,000행 중 학습 분할 3,598행을 준비했고, batch 1로 500스텝(약 0.139 epoch)을 실행했다. 검증 702행은 250·500스텝에서 손실 평가에만 사용했다. 테스트 700행과 별도 개발 24건은 학습에 넣지 않았다.

- 모델/배포본: Qwen3-4B-Instruct-2507 / Unsloth NF4 배포본. 체크섬과 revision은 앞선 4B 비교와 동일하다.
- 방식: QLoRA SFT, NF4 이중 양자화, BF16 연산, PEFT k-bit 학습 준비.
  기반 모델은 고정, `all-linear` LoRA만 업데이트. rank 8, alpha 16, dropout 0.05,
  학습률 0.0001, batch 1, accumulation 1, 최대 길이 768, seed 42,
  gradient checkpointing, 응답 부분만 손실 계산, 학습 파라미터 16,515,072개.
- 학습·중간검증·최종검증 시간: 1,269.9931초 (약 21분 10초). 평균 학습 loss 0.2454303.
- 검증 loss: 250스텝 0.2815225 → 500스텝 0.2524993. 검증 시간은 각각 289.2687초,
  289.9404초였다. 검증 loss가 낮은 500스텝을 먼저 개발 평가했다.
- 장치: 로컬 RTX 3060 Laptop GPU 6GB. peak CUDA allocated 4.80GiB,
  reserved 4.95GiB. batch 1의 이번 데이터에서 OOM 없이 완료했으며 더 긴 입력이나
  다른 배치 크기의 학습 가능성을 보장하지 않는다. 로딩 순간의 최대치는 제외한다.
- 어댑터: `filtering_training/outputs/qlora-qwen3-4b-500-20260930/adapter`.
- 중간/최종 체크포인트: 같은 run의 `trainer/checkpoint-250`, `trainer/checkpoint-500`.
- 평가 보고서: `filtering_training/outputs/evaluation/qlora_qwen3_4b_500_dev24.json`.
  전체 실제 응답은 `qlora_qwen3_4b_500_predictions24.json`에 저장했다.

| 지표 | 1.7B 기존 500스텝 LoRA | 4B 기본 NF4 | 4B QLoRA 500스텝 |
| --- | ---: | ---: | ---: |
| 유효 JSON | 24/24 | 24/24 | 24/24 |
| 통과·차단 정답 | 19/24 (79.2%) | 17/24 (70.8%) | 24/24 (100%) |
| 긴급 알림 오차단 | 5/9 | 1/9 | 0/9 |
| 불필요한 알림 오통과 | 0/14 | 6/14 | 0/14 |
| 긴급도 정확히 일치 | 12/24 | 10/24 | 16/24 |
| 관련도 정확히 일치 | 16/24 | 5/24 | 17/24 |
| 카테고리 macro F1 | 0.5735 | 0.3149 | 0.7440 |
| 긴급도 평균 절대 오차 | 0.5833 | 0.7917 | 0.3750 |
| 관련도 평균 절대 오차 | 0.5000 | 1.2917 | 0.3750 |

카테고리 자체의 정답은 18/24다. 정책 정답 24/24는 모든 점수·카테고리·설명까지 정확하다는 뜻이 아니다. 예를 들어 아래 기술 강연 홍보는 올바르게 차단했지만 카테고리를 `개인 중요`로 잘못 예측했다. 설명의 근거와 카테고리 오류도 계속 검토해야 한다.

4B 학습 전과는 같은 기반/양자화/평가 데이터/프롬프트/결정적 생성 조건으로 비교했다. 기존 1.7B 학습 모델과는 데이터 규모, LoRA 대상 모듈, 학습률, 양자화 방식도 다르므로 개선 전체를 모델 크기 효과로 분리할 수 없다. 평가 문장을 학습에 넣지는 않았으나 개발 24건의 오류 유형을 데이터 확장 방향에 반영했으므로 이 점수는 반복 사용한 개발 평가 결과다. 독립적 최종 성능으로 주장하지 않는다.

현재 어댑터를 개발 비교에서 가장 나은 후보로 보존한다. 더 큰 스텝 수보다 미사용 알림의 자연스러운 문장과 라벨에서 먼저 확인한다. 500스텝을 넘는 학습이나 전체 1회 학습은 아직 실행하지 않았다. 최종 모델 채택과 기본 모델 변경은 하지 않았다. 데이터셋·기반 모델·어댑터·집계 결과는 로컬에서 Git 제외 상태다.

학습 데이터 SHA-256: `489f749adefaed0a1e62b767daafe47c23ce7503e73afd067f819d430a61ce33`.

추론 생성 시간 평균 7.89초/알림. 모델 로딩·입력 토큰화·응답 파싱은 제외하며 첫 생성도 포함한다. 긴급 알림 전달 지연 측면에서 추론 최적화가 남아 있다.

추론 시 peak CUDA allocated 2.7GiB, reserved 2.8GiB.

### 학습 후 입력·출력 — holdout_001

입력:

```json
{"notification":{"id":"holdout_001","app_name":"Antigravity IDE","sender":"","title":"결제 API 필드 변경","body":"오늘 병합 전 승인 코드 수정 필요","timestamp":"2026-09-23T09:07:00Z"},"context":{"active_process":"Code.exe","window_title":"payment_gateway.py - Visual Studio Code","last_updated":"2026-09-23T09:06:56Z","duration_seconds":48,"recent_processes":["Code.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 구현 중인 결제 API의 필드가 바뀌어 병합 전에 수정해야 함"}
```

실제 모델 출력:

```json
{"urgency_score":4,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"오늘 병합 전 승인 코드 수정 필요 알림은 현재 열어 둔 작업과 직접 관련이 있으며 현재 작업을 잠시 중단하고 확인할 가치가 있습니다."}
```

### 학습 후 입력·출력 — holdout_007

입력:

```json
{"notification":{"id":"holdout_007","app_name":"Calendar","sender":"가상 일정 앱","title":"회의 접속 링크 변경","body":"2분 뒤 시작할 정기 회의의 접속 링크가 바뀌었습니다. 새 링크를 사용해 주세요.","timestamp":"2026-09-23T09:49:00Z"},"context":{"active_process":"Excel.exe","window_title":"개인 지출 내역.xlsx - Excel","last_updated":"2026-09-23T09:48:56Z","duration_seconds":40,"recent_processes":[]}}
```

정답:

```json
{"urgency_score":4,"relevance_score":1,"category":"일정/회의","ai_summary_reason":"현재 작업과 무관하지만 회의 시작이 임박해 새 링크를 확인해야 함"}
```

실제 모델 출력:

```json
{"urgency_score":4,"relevance_score":1,"category":"일정/회의","ai_summary_reason":"2분 뒤 시작할 정기 회의 접속 링크 변경 알림은 현재 작업과 무관하며 현재 작업을 잠시 중단하고 확인할 가치가 있습니다."}
```

### 학습 후 입력·출력 — holdout_020

입력:

```json
{"notification":{"id":"holdout_020","app_name":"Webinar","sender":"가상 교육 업체","title":"검색 기술 강연 신청","body":"검색 서비스 설계 온라인 강연 신청을 받고 있습니다. 다음 달 참석 여부를 결정해 주세요.","timestamp":"2026-09-23T11:20:00Z"},"context":{"active_process":"Code.exe","window_title":"search_index.py - Visual Studio Code","last_updated":"2026-09-23T11:19:56Z","duration_seconds":67,"recent_processes":["Code.exe","chrome.exe"]}}
```

정답:

```json
{"urgency_score":1,"relevance_score":3,"category":"광고/홍보","ai_summary_reason":"현재 검색 개발 주제와 간접적으로 관련된 홍보지만 즉시 확인할 필요는 없음"}
```

실제 모델 출력:

```json
{"urgency_score":2,"relevance_score":1,"category":"개인 중요","ai_summary_reason":"다음 달 참석 여부 결정 필요. 현재 다른 분야의 작업을 하고 있어 관련성이 낮음."}
```

## 2026-09-30 - Frozen 100-case synthetic candidate comparison

- Data: 100 new author-written Korean synthetic notifications, 12-13 per category; 27 urgent, 37 policy PASS, 63 BLOCK. Labels fixed before reading candidate outputs.
- Data SHA-256: `f8d3e5e9ea27389a9043c26f2ce437797cd8e3b861bf0441955392dcd1a1f984`. Prompt SHA-256: `0820a8b9405212299385b5b9838cfb0c55650b0f6b4b3b5b6e73b5d0a8f82a63`.
- Checked distinct IDs and normalized alert title/body against the local 5,000-row set, repeated development 24, and tracked sample data. No exact alert text overlap; stylistic independence is not established.
- Gold labels are provisional and lack separate human review. No real private notification is included. Full dataset, predictions, audit, and adapters stay under Git-ignored local paths.
- Same 100 inputs and prompt. Candidate pipelines differ in base model, training dataset (1.7B on prior 3,000 rows; 4B on expanded 5,000 rows), LoRA target modules, learning rate, and precision. Results cannot isolate model size.

| Metric | Qwen3-1.7B LoRA 500 | Qwen3-4B Instruct NF4 QLoRA 500 |
| --- | ---: | ---: |
| Policy accuracy | 86/100 | 94/100 |
| Valid JSON | 98/100 | 100/100 |
| Explicit urgent false blocks | 4/27 | 2/27 |
| Unnecessary passes | 3/63 | 2/63 |
| Exact category | 73/100 | 85/100 |
| Category macro F1, valid JSON | 0.7422 | 0.8477 |
| Exact urgency | 68/100 | 79/100 |
| Exact relevance | 43/100 | 67/100 |
| Mean generation time/alert | 3.21 s | 7.70 s |

- 4B policy errors: IDs 016 (today accessibility review, relevance underestimated), 022 (today customer reply, urgency overestimated), 031 and 034 (imminent meeting/appointment urgency underestimated), 033 (workshop change relevance underestimated), and 046 (already blocked login attempt urgency overestimated).
- 4B has 100% policy decisions in urgent work, personal important, personal general, promotion, and other categories; meeting/schedule is 10/13. It still misses 2 of 27 urgent cases, so do not claim final safety.
- Timing measures only model.generate, includes first generation, and excludes model loading, tokenization and parsing. 4B is materially slower despite lower quantized CUDA allocation.
- This set is now development evidence because its results were inspected. Do not train on its 100 examples. Review labels and author a fresh unseen set before claiming generalization after tuning.

### Actual unlabeled JSON inputs and adapter outputs

The three examples below were rerun through `infer_sample --input-json` with the saved 4B adapter in new processes. Input JSON has no label; gold labels are shown separately for comparison.

**Example 001**

Input:
```json
{"notification":{"id":"independent_20260930_001","app_name":"Slack","sender":"Slack","title":"정산 금액 이상","body":"방금 배포한 반올림 코드 때문에 고객 정산 금액이 두 배로 기록되고 있습니다. 배치부터 멈춰 주세요.","timestamp":"2026-09-30T05:00:00Z"},"context":{"active_process":"Code.exe","window_title":"settlement_rounding.cs - Visual Studio Code","last_updated":"2026-09-30T05:00:00Z","duration_seconds":43,"recent_processes":["Code.exe","chrome.exe"]}}
```
Gold:
```json
{"urgency_score":5,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"현재 수정 중인 정산 코드의 금전 오류가 진행 중이다."}
```
Model output:
```json
{"notification_id":"independent_20260930_001","is_passed":true,"urgency_score":5,"relevance_score":5,"category":"긴급 업무","ai_summary_reason":"정산 금액 이상 알림은 현재 열어 둔 작업과 직접 관련이 있으며 즉시 확인하거나 대응해야 합니다."}
```

**Example 068**

Input:
```json
{"notification":{"id":"independent_20260930_068","app_name":"카카오톡","sender":"카카오톡","title":"오늘 저녁 약속","body":"오늘 저녁 식당 예약 인원을 정해야 해. 오후 중에 올 수 있는지 답해 줘.","timestamp":"2026-09-30T05:00:00Z"},"context":{"active_process":"KakaoTalk.exe","window_title":"오늘 저녁 약속 대화","last_updated":"2026-09-30T05:00:00Z","duration_seconds":12,"recent_processes":["KakaoTalk.exe","chrome.exe","Code.exe"]}}
```
Gold:
```json
{"urgency_score":3,"relevance_score":5,"category":"개인 일반","ai_summary_reason":"오늘 필요한 개인 약속 응답이며 현재 같은 대화를 보고 있다."}
```
Model output:
```json
{"notification_id":"independent_20260930_068","is_passed":true,"urgency_score":3,"relevance_score":5,"category":"개인 중요","ai_summary_reason":"오늘 저녁 약속 알림은 현재 열어 둔 작업과 직접 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```

**Example 080**

Input:
```json
{"notification":{"id":"independent_20260930_080","app_name":"쇼핑 앱","sender":"쇼핑 앱","title":"쿠폰 만료 10분 전","body":"받아 둔 쿠폰이 10분 뒤 사라집니다. 사용하지 않아도 결제되는 금액은 없습니다.","timestamp":"2026-09-30T05:00:00Z"},"context":{"active_process":"Excel.exe","window_title":"업무 정산.xlsx","last_updated":"2026-09-30T05:00:00Z","duration_seconds":12,"recent_processes":["Excel.exe","chrome.exe","KakaoTalk.exe"]}}
```
Gold:
```json
{"urgency_score":2,"relevance_score":1,"category":"광고/홍보","ai_summary_reason":"쿠폰 만료는 금전 피해 없는 광고이며 현재 업무와 무관하다."}
```
Model output:
```json
{"notification_id":"independent_20260930_080","is_passed":false,"urgency_score":2,"relevance_score":1,"category":"개인 일반","ai_summary_reason":"사용하지 않아도 결제되는 금액이 없는 쿠폰. 현재 다른 분야의 작업을 하고 있어 관련성이 낮음."}
```

Example 001 matches the gold category and policy. Examples 068 and 080 have correct policy decisions but wrong categories, so policy accuracy alone does not describe full-label quality.

### Persistent inference probe

The same three unlabeled JSON inputs were sent as newline-delimited JSON to one 4B process. Outputs matched the three single-call results exactly. One model load plus three predictions took 37.06 seconds end-to-end (12.35 seconds/request averaged across only three requests). This includes process startup and model loading. The 100-case per-alert generation mean remains 7.70 seconds with p95 9.76 seconds, so persistent loading does not eliminate generation latency. No Windows client integration has been tested.

## 2026-09-30 - Continued 4B QLoRA, 250 additional steps

- New run loaded the saved 500-step 4B NF4 adapter as trainable, kept the same 5,000-row dataset and 3,598/702 training/validation split, and used all-linear rank-8 LoRA with a fresh optimizer, seed 43 and learning rate 0.00002. The original adapter was preserved.
- This is 250 further updates from the 500-step adapter, not a Trainer checkpoint/optimizer resume. It does not isolate the effect of training duration because the optimizer schedule and seed changed.
- Full validation loss after the extra 250 updates was 0.2465 versus 0.2525 at the original 500 steps. Training plus validation took about 654.5 seconds. Lower loss did not improve policy behavior.
- On the inspected synthetic 100-case development set, original 500-step adapter: 94/100 policy, 85/100 exact category, 2/27 urgent false blocks, 2/63 unnecessary passes, 7.70 s mean generation. Continued adapter: 92/100 policy, 85/100 exact category, 2/27 urgent false blocks, 2/63 unnecessary passes, 7.18 s mean generation.
- New-only policy errors: independent_20260930_058, independent_20260930_092. Fixed prior policy errors: none. The source evaluation set was already inspected; this is model development, not a new blind result.
- Decision: retain the original 500-step 4B adapter. Do not switch defaults or use the continued adapter for the demo. Do not move these 100 evaluation cases into training. Review borderline labels and add genuinely new evaluation cases before another training decision.

### Actual inputs and before/after outputs

**independent_20260930_058**

Input:
```json
{"notification":{"id":"independent_20260930_058","app_name":"택배","sender":"택배","title":"본인 확인 안내","body":"해외 주문 통관에 본인 확인이 필요합니다. 오늘 중 앱에서 확인하면 됩니다.","timestamp":"2026-09-30T05:00:00Z"},"context":{"active_process":"chrome.exe","window_title":"해외 주문 배송 조회","last_updated":"2026-09-30T05:00:00Z","duration_seconds":110,"recent_processes":["chrome.exe","KakaoTalk.exe"]}}
```
Gold:
```json
{"urgency_score":3,"relevance_score":4,"category":"개인 중요","ai_summary_reason":"오늘 필요한 배송 확인이 현재 배송 조회와 연관된다."}
```
Original 500-step output:
```json
{"urgency_score":3,"relevance_score":4,"category":"개인 중요","ai_summary_reason":"해외 주문 통관에 본인 확인이 필요합니다. 오늘 중 앱에서 확인하면 됩니다. 현재 창 제목에 같은 대상이 있어 직접 관련됨."}
```
Continued output:
```json
{"urgency_score":2,"relevance_score":4,"category":"개인 중요","ai_summary_reason":"해외 주문 통관에 본인 확인 안내 알림은 현재 열어 둔 작업과 직접 관련이 있으며 나중에 확인해도 됩니다."}
```

**independent_20260930_092**

Input:
```json
{"notification":{"id":"independent_20260930_092","app_name":"기기 안내","sender":"기기 안내","title":"연결 완료","body":"책상 조명이 블루투스로 연결됐습니다. 밝기는 이전 설정으로 유지됩니다.","timestamp":"2026-09-30T05:00:00Z"},"context":{"active_process":"Code.exe","window_title":"http_retry.py - Visual Studio Code","last_updated":"2026-09-30T05:00:00Z","duration_seconds":12,"recent_processes":["Code.exe","chrome.exe","KakaoTalk.exe"]}}
```
Gold:
```json
{"urgency_score":1,"relevance_score":1,"category":"기타","ai_summary_reason":"일반 주변 기기 상태 안내이며 현재 코드와 무관하다."}
```
Original 500-step output:
```json
{"urgency_score":1,"relevance_score":1,"category":"기타","ai_summary_reason":"책상 조명이 블루투스로 연결됐습니다. 밝기는 이전 설정으로 유지됩니다."}
```
Continued output:
```json
null
```

**independent_20260930_016**

Input:
```json
{"notification":{"id":"independent_20260930_016","app_name":"Jira","sender":"Jira","title":"오늘 중 확인 부탁","body":"접근성 개선안에 담당자 의견이 필요합니다. 다음 논의를 위해 오늘 오후까지 코멘트 부탁드립니다.","timestamp":"2026-09-30T05:00:00Z"},"context":{"active_process":"Code.exe","window_title":"keyboard_navigation.ts - Visual Studio Code","last_updated":"2026-09-30T05:00:00Z","duration_seconds":12,"recent_processes":["Code.exe","chrome.exe"]}}
```
Gold:
```json
{"urgency_score":3,"relevance_score":5,"category":"일반 업무","ai_summary_reason":"오늘 필요한 의견 요청이 현재 접근성 구현과 직접 관련된다."}
```
Original 500-step output:
```json
{"urgency_score":3,"relevance_score":3,"category":"일반 업무","ai_summary_reason":"접근성 개선안에 담당자 의견이 필요합니다. 다음 논의를 위해 오늘 오후까지 코멘트 부탁드립니다."}
```
Continued output:
```json
{"urgency_score":3,"relevance_score":3,"category":"일반 업무","ai_summary_reason":"접근성 개선안에 담당자 의견이 필요합니다. 다음 논의를 위해 오늘 오후까지 코멘트 부탁드립니다."}
```

## 2026-09-30 - New 80-case synthetic stress evaluation

- 80 new authored Korean notifications (10 per category), gold labels frozen before either model output. No exact normalized title/body overlap with the 5,000-row training candidates, first 100-case evaluation, previous development 24, or tracked sample.
- Data SHA-256: `cc05b7b1470f5b71b5b6d3e36acf0fe15c8d45fc3d1a4b1206a2e66c442c9e69`. Prompt SHA-256: `0820a8b9405212299385b5b9838cfb0c55650b0f6b4b3b5b6e73b5d0a8f82a63`. 24 urgent cases, 30 gold policy PASS cases.
- Created after inspecting the first 100-case errors, so this is a targeted stress test, not an unbiased final blind evaluation. Labels were written by the same author and have not received independent human review. No private notification data was used.

| Metric | Qwen3-1.7B LoRA 500 | Qwen3-4B NF4 QLoRA 500 |
| --- | ---: | ---: |
| Policy accuracy | 65/80 | 76/80 |
| Valid JSON | 79/80 | 80/80 |
| Urgent false blocks | 6/24 | 0/24 |
| Unnecessary passes | 2/50 | 3/50 |
| Exact category | 56/80 | 65/80 |
| Category macro F1 on valid JSON | 0.6943 | 0.8161 |
| Mean generation time | 3.10 s | 7.11 s |

- 4B policy-error IDs: stress_20260930_019, stress_20260930_025, stress_20260930_043, stress_20260930_054. Urgent misses were zero on this stress set, but false passes remain. The 4B adapter remains the preferred development candidate; no default switch or model publication was performed.
- Local blind review sheets and prediction-aware triage remain Git ignored under `filtering_training/outputs/independent_stress_80/review/`. Review the blind sheet first to avoid anchoring on model outputs.

### Actual input, gold, and candidate outputs

**stress_20260930_019**

Input:
```json
{"notification":{"id":"stress_20260930_019","app_name":"Jira","sender":"Jira","title":"오후 문의 회신 요청","body":"재현한 UI 오탈자의 수정 일정을 오늘 오후까지 고객에게 알려 주세요.","timestamp":"2026-09-30T06:00:00Z"},"context":{"active_process":"KakaoTalk.exe","window_title":"친구와 산책 약속","last_updated":"2026-09-30T06:00:00Z","duration_seconds":145,"recent_processes":["KakaoTalk.exe"]}}
```
Gold:
```json
{"urgency_score":3,"relevance_score":1,"category":"일반 업무","ai_summary_reason":"오늘 회신할 업무 문의지만 현재 개인 대화와 무관하다."}
```
1.7B output:
```json
{"urgency_score":3,"relevance_score":1,"category":"일반 업무","ai_summary_reason":"재현한 UI 오탈자의 수정 일정을 오늘 오후까지 고객에게 알려 주세요."}
```
4B output:
```json
{"urgency_score":4,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"오늘 오후 고객에게 수정 일정을 알려야 하는 오탈자 수정 요청. 현재 다른 분야의 개인 약속을 하고 있어 관련성이 낮음."}
```

**stress_20260930_025**

Input:
```json
{"notification":{"id":"stress_20260930_025","app_name":"Google Calendar","sender":"Google Calendar","title":"내일 면접 장소 확정","body":"내일 면접은 별관 2층으로 확정됐습니다. 이동 준비에는 아직 하루가 남았습니다.","timestamp":"2026-09-30T06:00:00Z"},"context":{"active_process":"chrome.exe","window_title":"내일 면접 일정표","last_updated":"2026-09-30T06:00:00Z","duration_seconds":30,"recent_processes":["chrome.exe"]}}
```
Gold:
```json
{"urgency_score":2,"relevance_score":5,"category":"일정/회의","ai_summary_reason":"현재 일정표와 연결되지만 급하게 대응할 상황은 아니다."}
```
1.7B output:
```json
{"urgency_score":3,"relevance_score":5,"category":"일정/회의","ai_summary_reason":"내일 면접 장소 확정 알림은 현재 열어 둔 작업과 직접 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```
4B output:
```json
{"urgency_score":3,"relevance_score":4,"category":"개인 중요","ai_summary_reason":"내일 면접 장소 확정 알림은 현재 열어 둔 작업과 직접 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```

**stress_20260930_043**

Input:
```json
{"notification":{"id":"stress_20260930_043","app_name":"병원","sender":"병원","title":"오늘 예약 취소 기한","body":"내일 검사 예약을 취소해야 한다면 오늘 저녁까지 연락해 주세요. 그대로 방문하면 조치할 필요가 없습니다.","timestamp":"2026-09-30T06:00:00Z"},"context":{"active_process":"chrome.exe","window_title":"내일 검사 예약 관리","last_updated":"2026-09-30T06:00:00Z","duration_seconds":145,"recent_processes":["chrome.exe"]}}
```
Gold:
```json
{"urgency_score":2,"relevance_score":4,"category":"개인 중요","ai_summary_reason":"예약 관리 화면과 연결되지만 취소 여부는 선택이다."}
```
1.7B output:
```json
{"urgency_score":3,"relevance_score":5,"category":"개인 중요","ai_summary_reason":"내일 검사 예약 취소 기한 알림은 현재 열어 둔 작업과 직접 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```
4B output:
```json
{"urgency_score":3,"relevance_score":5,"category":"개인 중요","ai_summary_reason":"오늘 예약 취소 기한 알림은 현재 열어 둔 작업과 직접 관련이 있으며 비교적 빠른 확인이 필요합니다."}
```

## 2026-10-03 — 검수 반영 v3 데이터와 Qwen3-0.6B 첫 파일럿

### 데이터와 검수

- 정책: `urgency4_or_relevance4_v2` — 긴급도 4 이상 또는 관련도 4 이상이면 통과.
- 집중 모드 ON을 전제로 하며, 빈 창에서도 쉬는 상태로 추정하지 않는다.
- 원문 66개, 문맥 변형 198건. 사람이 개별 검수한 원문은 24개이며 나머지 42개는 합성 후보다.
- 두 번째 8개 검수 승인: 사용자 발언 `딱 좋다. ㅇㅇ`.
- 기존 후보 192건은 보존했다. 독립적인 일정·광고 평가 그룹을 확보하기 위해 원문 2개를 추가했으며, 이 2개는 개별 검수 완료로 표시하지 않았다.
- 학습 150건(원문 50개), 검증 24건(원문 8개), 테스트 24건(원문 8개).
- 같은 알림의 변형, 동일한 정보가 있는 창, 지정한 유사 시나리오 그룹은 같은 분할에 둔다. 지정하지 않은 모든 의미 유사성까지 자동 검증한 것은 아니다.
- 데이터 SHA-256: `6341f93f8ff0844144f9972723845cdb29e59b93a113aeb276c4fc1d695e424f`.

### 학습 설정과 모델 저장

- 모델: `Qwen/Qwen3-0.6B`, NF4 이중 양자화 QLoRA, BF16 연산.
- LoRA: r=8, alpha=16, dropout=0.05, 대상 `q_proj`, `v_proj`.
- 150스텝 / 1 epoch, batch 1, 최대 768토큰, 학습률 0.0002, seed 42.
- 장비: NVIDIA GeForce RTX 3060 Laptop GPU.
- 학습 시간 약 61.7초, 평균 학습 loss 약 0.9740, 최대 CUDA 예약 메모리 2,032 MiB.
- 학습 중 검증 loss는 측정하지 않았다. 저장 후 별도 추론으로 검증 24건을 평가했다.
- 어댑터: `filtering_training/outputs/v3_reviewed_01/qwen06_pilot/adapter/adapter_model.safetensors` (2,308,544 bytes).
- 어댑터 설정과 토크나이저는 같은 `adapter/` 폴더에 저장했다. 전체 기반 모델을 병합한 체크포인트는 아니다.
- 실행 설정: `filtering_training/outputs/v3_reviewed_01/qwen06_pilot/run_config.json`.

### 동일 검증 세트 비교

| 모델 | 통과·차단 정답 | 유효 JSON |
| --- | --- | --- |
| TF-IDF + LinearSVC 기준선 | 18/24 (75.0%) | 해당 없음: 프로그램이 구성 |
| Qwen3-0.6B 학습 전 | 9/24 (37.5%) | 20/24 |
| Qwen3-0.6B 150스텝 학습 후 | 17/24 (70.8%) | 24/24 |

학습 후 긴급 알림 6건 중 4건을 차단했다. 이는 서로 다른 원문 2개의 무관·빈 창 변형이다.
계약 금액 오류와 누수 위험 알림에서 관련도가 낮다는 이유로 긴급도도 1로 낮췄다.
검증 원문 8개 중 3개에서 같은 알림의 긴급도가 문맥에 따라 변했다.
판단 이유에도 부정확한 표현이 있어 이번 어댑터는 앱 적용 후보로 채택하지 않는다.

학습 전 모델의 잘못된 JSON 4건은 정확도에서 오답으로 계산하지만 통과·차단으로 분류하지 않는다.
따라서 긴급 오차단률만으로 학습 전 모델의 안정성을 판단하면 안 된다.
검증 원문은 8개뿐이므로 일반 성능이나 모델 간 우열을 확정할 수 없다.
테스트 세트는 사용하지 않았으며 일반 PC의 CPU 지연·앱 메모리는 측정하지 않았다.
이전 실험과 데이터 및 정책이 달라 과거 정확도와 직접 비교하지 않는다.

### 결과 파일과 다음 보완

- 전체 정리: `filtering_training/outputs/v3_reviewed_01/pilot_summary.md`.
- 비교 지표: `filtering_training/outputs/v3_reviewed_01/pilot_comparison.json`.
- 학습 후 평가: `filtering_training/outputs/v3_reviewed_01/qwen06_pilot/validation_report.json`.
- 학습 후 원문·출력: `filtering_training/outputs/v3_reviewed_01/qwen06_pilot/validation_predictions.jsonl`.
- 통과·차단 오류 7건: `filtering_training/outputs/v3_reviewed_01/qwen06_pilot/policy_errors.csv`.
- 다음 보완: 무관·빈 창에서도 긴급한 알림과, 긴급하지 않아도 현재 목적에 직접 도움이 되는 알림을 보강한다. 긴급도 입력에서 작업 문맥을 제외하는 분리 실험과 점수 중심 학습을 비교한다.
- 추가 실험용 데이터·어댑터는 로컬 `outputs/`에 저장돼 있으며, Git에 커밋하거나 원격에 업로드하지 않았다.

각 실험은 같은 형식으로 아래에 추가한다. 점수는 해당 실험의 테스트 분할에서만 비교한다. 서로 다른 데이터셋이나 분할의 점수는 직접 비교하지 않는다. 예시는 합성 데이터의 실제 모델 출력이며 정답 라벨과 구분한다.
