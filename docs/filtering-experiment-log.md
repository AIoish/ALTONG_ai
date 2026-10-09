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

## 2026-10-03 — 0.6B 고정 데이터의 출력 축소·긴급도/관련도 분리 비교

기존 v3 학습 150건·검증 24건으로 전체 네 필드(A), 두 점수(B), 작업별
분리 어댑터(C)를 비교했다. Qwen/Qwen3-0.6B, NF4 이중 양자화·BF16,
LoRA r=8/alpha=16/dropout=0.05, q_proj/v_proj, 학습률 0.0002,
batch 1, 최대 768토큰, seed 42를 고정하고 각 작업을 새 기반 모델에서 시작했다.
A/B는 각각 150스텝, C는 중복 알림을 제거한 긴급도 50스텝·관련도 150스텝이다.
총 학습량이 같은 비교는 아니다. 긴급도에는 문맥과 변형 접미사가 있는 ID를 넣지 않았다.

| 방식 | 정책 정답 | 긴급 오차단 | 유효 출력 | 오통과 | 평균 / p95 전체 처리 |
| --- | --- | --- | --- | --- | --- |
| A 네 필드 | 17/24 | 4/6 | 24/24 | 0 | 3.358 / 4.036초 |
| B 두 점수 | 20/24 | 4/6 | 24/24 | 0 | 0.852 / 0.889초 |
| C 분리 | 12/24 | 6/6 | 24/24 | 0 | 0.977 / 1.022초 |

처리 시간은 워밍업 후 토큰화·생성·파싱·어댑터 전환을 포함하며 로딩은 별도다.
B는 관련 문맥 3건을 추가로 맞췄지만 계약 금액 오류와 누수 알림의 무관·빈 창
오차단 4건이 남았다. C는 24건 모두 두 점수를 1로 출력했다. 문맥별 긴급도는
같아졌지만 정답으로 개선되지 않았으며, 원인 확인 없이 추가 학습하지 않는다.
형식만 맞는 출력과 낮은 loss를 올바른 판단의 증거로 사용하지 않는다.

사용자 발언 `ㅇㅇ 이대로 진행해보자.`로 보강 원문 8개·문맥 24건을 승인했다.
기존 보류 유사 그룹과 공유 창 연결을 보수적으로 적용해 이번 학습에서는 전체
보강 묶음을 제외했다. 기존 원문 66개·198건의 해시와 검수 범위를 유지했다.
새 후보 승인 기록은 `filtering_training/outputs/v3_score_separation_01/approval.json`이다.
원본 데이터 SHA-256은 `6341f93f8ff0844144f9972723845cdb29e59b93a113aeb276c4fc1d695e424f`이다.

### 실제 출력 예시

1. `v3_20_unrelated`: “오늘 발송할 계약서의 금액이 견적과 다릅니다. 발송 전 수정해 주세요”
   — 정답 긴급도 4·관련도 1, A/B/C 모두 1·1로 차단.
2. `v3_44_empty`: “천장에서 물 계속 떨어져. 전등 쪽까지 번지는데 관리실에 지금 연락해야 할 듯”
   — 정답 긴급도 5·관련도 1, A/B/C 모두 1·1로 차단.
3. `v3_26_related`: “3페이지 표의 합계가 다른 것 같아요. 오늘 중으로 숫자 확인 부탁드립니다”
   — 정답 긴급도 3·관련도 5. B는 정책을 맞췄고 A/C는 1·1로 차단.

전체 예측·설정·가중치·로그·검증된 비교와 요약은 로컬
`filtering_training/outputs/runs/2026-10-03-qwen06-score-pilot-01/`에 보관했다.
입력·정답·분할·예측 해시와 원시 출력 기반 지표 재계산을 확인했다.
필터링 관련 테스트 43개 통과. 최종 테스트 추론·앱 적용·커밋·푸시는 하지 않았다.
독립 검증 원문 8개, 단일 seed의 합성 파일럿이며 일반 PC CPU 성능도 측정하지 않았다.
세 방식 모두 앱 적용 후보로 채택하지 않는다. 다음은 보류 그룹과 겹치지 않는
작은 보강 검수 및 단일 점수 학습의 분포·숫자 토큰 기여 확인이다.

## 2026-10-04 — 0.6B 분리 모델의 숫자 loss와 점수별 샘플링 대조

기존 v3 학습 분할에서 숫자 비중과 정답 분포의 영향을 따로 비교했다.
Qwen/Qwen3-0.6B, NF4 이중 양자화·BF16, LoRA r=8/alpha=16/dropout=0.05,
q_proj/v_proj, 학습률 0.0002, seed 42, batch 1, 최대 768토큰을 유지하고
각 대조를 기반 모델에서 새로 시작했다. 긴급도 50스텝·관련도 150스텝으로 고정했다.
검증은 같은 원문 8개·문맥 24건이며 최종 테스트는 추론하지 않았다.

| 대조 | 정책 정답 | 긴급 오차단 | 불필요 오통과 | 유효 출력 | 알림 평균 처리 |
| --- | --- | --- | --- | --- | --- |
| 기존 분리 | 12/24 | 6/6 | 0/12 | 24/24 | 0.977초 |
| 일반 nll 기준선 | 12/24 | 6/6 | 0/12 | 24/24 | 0.974초 |
| 숫자 가중치 7 | 12/24 | 6/6 | 0/12 | 24/24 | 0.977초 |
| 점수별 균등 샘플링 | 12/24 | 0/6 | 12/12 | 24/24 | 0.969초 |

기존 분리·일반 nll·숫자 가중치 모델은 검증 전체에서 두 점수를 1로 생성했다.
숫자 토큰 비중은 completion 8토큰 중 1/8에서 1/2로 높였지만 해결되지 않았다.
기본 chunked_nll과 새 callback의 로깅 충돌은 nll 경로로 명시해 수정했고,
동일 경로의 가중치 1 기준선으로 기존 결과가 재현되는지 확인했다.

균등 샘플링은 train 행에서만 replacement로 추출했고 긴급도 점수당 10건,
관련도 점수당 30건이었다. 고유 행은 각각 28건·64건이며 반복 횟수를 기록했다.
균등 모델은 긴급도를 모두 4 또는 5로 출력해 모든 알림을 통과시켰다.
긴급 오차단 0건은 필터링 품질 개선이 아니며 세 대조 모두 앱 적용 후보로 채택하지 않는다.

각 점수의 첫 두 원본 학습 사례씩 10건을 진단했을 때 일반 nll·숫자 가중치는
긴급도/관련도 각각 2/10, 균등 샘플링은 3/10·6/10이었다.
이는 전체 학습 정확도가 아니다. 단독 어댑터 로딩과 두 어댑터 추론 결과는 일치했다.

### 실제 출력 예시

- `v3_44_empty`: 전등 쪽으로 번지는 누수. 정답 긴급도 5·관련도 1,
  균등 모델은 4·1로 PASS.
- `v3_50_unrelated`: 피아노 재생목록 홍보. 정답 1·1,
  균등 모델은 5·1로 잘못 PASS.
- `v3_56_empty`: 충전기 할인 광고. 정답 1·1,
  균등 모델은 4·1로 잘못 PASS.

보강 원문 8개·문맥 24건은 사용자 수정 `6번 관련도 2. 7번 관련도 4로 변환.`과
뒤이은 `ㄱㄱ` 진행 승인을 기록해 확정했다. 7번은 일반 농담 기준의 사용자
지정 예외로 식별하며 공통 프롬프트와 기존 가이드는 바꾸지 않았다.
이번 고정 데이터 대조에는 새 보강 자료를 넣지 않았다. 승인 자료는
`filtering_training/outputs/v3_score_review_02_approved_01/`에 보존했다.

설정·예측 원문·가중치·학습 로그·진단·검증된 비교는 로컬
`filtering_training/outputs/runs/2026-10-04-qwen06-score-controls-01/`에 있다.
원본 v3 SHA-256은 `6341f93f8ff0844144f9972723845cdb29e59b93a113aeb276c4fc1d695e424f`로 유지됐다.
입력·정답·분할·원시 응답 기반 지표와 실제 샘플링 인덱스를 확인했다.
다음은 형식 생성을 제외한 숫자 1~5 직접 학습·평가를 먼저 비교하고,
그 뒤 승인 보강 자료의 추가 효과를 따로 비교한다.
단일 seed·소규모 합성 검증이며 일반 PC CPU 지연은 측정하지 않았다.
최종 테스트 추론·앱 적용·커밋·푸시는 하지 않았다.

## 2026-10-04 — 0.6B 다섯 숫자 후보의 직접 분류 비교

JSON·EOS completion 대신 프롬프트 마지막 위치의 숫자 1~5 토큰 logit만
선택해 다섯 클래스 조건부 cross entropy를 학습했다. 추론은 같은 다섯 후보의
argmax를 점수로 변환하고 두 점수에 기존 PASS 정책을 적용한다.
별도 분류 head는 추가하지 않았다. 출력 형식은 코드가 보장하므로 생성 모델의
JSON 준수율 개선으로 해석하지 않는다. 지시 프롬프트와 loss·추론이 함께 바뀌어
결과를 형식 제거 하나의 효과로 단정할 수 없다.

Qwen/Qwen3-0.6B, NF4 이중 양자화·BF16, LoRA r=8/alpha=16/dropout=0.05,
q_proj/v_proj, 학습률 0.0002, seed 42, batch 1, 최대 768토큰을 유지했다.
긴급도에는 ID·문맥을 제외한 알림만, 관련도에는 알림과 문맥을 입력했다.
기반 모델에서 각 어댑터를 새로 시작해 긴급도 50스텝·관련도 150스텝씩 학습했다.
기존 원문 50개·문맥 150건의 학습 분할과 검증 원문 8개·문맥 24건을 고정했다.
승인 보강 데이터와 최종 테스트는 사용하지 않았다.

| 방식 | 정책 정답 | 긴급 오차단 | 불필요 오통과 | 관련도 정확 일치 | 알림 평균 처리 |
| --- | --- | --- | --- | --- | --- |
| 기존 JSON 분리, 자연 분포 | 12/24 | 6/6 | 0/12 | 이전 실험 참조 | 0.977초 |
| 숫자 직접 분류, 자연 분포 | 20/24 | 4/6 | 0/12 | 21/24 | 0.150초 |
| 숫자 직접 분류, 균등 샘플링 | 10/24 | 2/6 | 12/12 | 19/24 | 0.158초 |

자연 분포 모델의 긴급도는 검증 24건 모두 1로 쏠렸다. 관련도는 8건에 5,
16건에 1을 출력해 정책 성능을 회복했지만 긴급도 문제는 해결하지 못했다.
균등 모델의 긴급도는 3이 3건, 4가 6건, 5가 15건이었다.
불필요한 알림 12건 전부를 통과시켜 정책 성능이 오히려 떨어졌다.
두 조건 모두 같은 알림의 문맥 3종에서 긴급도가 같았으며 8개 그룹 모두 확인했다.
문맥 독립성과 올바른 긴급도 판단은 별개이다.

균등 샘플링은 기존 대조와 같은 seed·인덱스 규칙으로 train에서만 replacement
추출했다. 긴급도 점수당 10건·관련도 점수당 30건, 고유 행은 각각 28건·64건이었다.
자연 분포 학습 loss는 긴급도 2.0218·관련도 1.0311, 시간은 19.20초·58.10초였다.
균등 조건은 loss 2.3485·1.7902, 시간 18.95초·57.21초였다.
이 조건부 분류 loss를 이전 전체 어휘 JSON loss 수치와 직접 비교하지 않는다.
실행 전 별도 긴급도 2스텝 smoke 학습도 완료했고 결과를 보존했다.

### 실제 출력 예시

- `v3_20_empty`: 발송 전 계약서 금액 수정. 정답 4·1,
  균등 모델은 3·1로 여전히 차단했다.
- `v3_44_empty`: 전등 쪽으로 번지는 누수. 정답 5·1,
  자연 분포는 1·1로 차단했고 균등 모델은 5·1로 통과시켰다.
- `v3_56_empty`: 충전기 할인 광고. 정답 1·1,
  균등 모델은 5·1로 잘못 통과시켰다.

설정·가중치·UTF-8 학습 로그·예측·다섯 후보의 조건부 확률·검증된 비교는
`filtering_training/outputs/runs/2026-10-04-qwen06-digit-01/`에 보존했다.
원본 v3 SHA-256은 `6341f93f8ff0844144f9972723845cdb29e59b93a113aeb276c4fc1d695e424f`로 유지됐다.
원본 입력·정답·분할 해시·샘플링 인덱스·확률 argmax·정책 지표를 재검증했다.
필터링 테스트 54개가 통과했다. 두 모델 모두 앱 적용 후보로 채택하지 않는다.
다음은 확정된 보강 원문 8개·문맥 24건의 추가 효과를 자연 분포 직접 분류와
비교한다. 검증 오류 원문을 학습에 복사하지 않고 검증 분할·최종 테스트를 유지한다.
단일 seed의 소규모 합성 검증이며 CPU 성능은 측정하지 않았다.
최종 테스트 추론·앱 적용·커밋·푸시는 하지 않았다.

## 2026-10-04 — 승인 보강 8개 추가와 동일 스텝 대조

확정 자료 `filtering_training/outputs/v3_score_review_02_approved_01/`의 원문 8개·
문맥 24건을 기존 학습 분할에만 추가했다. 사용자 수정 6번 관련도 2·7번 관련도 4를
유지했고 7번의 일반 농담 기준 예외를 학습 설정에 식별했다. 공통 프롬프트는 바꾸지 않았다.
승인 자료 SHA-256은 `920904275c18382616d9318369d320a84f444070203ea1b8e3222284d90789fd`이다.
확정 당시의 `training_used: false`는 당시 상태 기록으로 보존했고 이번 사용 이력은
새 준비 manifest와 run_config의 supplement 항목으로 기록했다.

긴급도는 중복 문맥을 제거해 50개에서 58개, 관련도는 150개에서 174개로 늘렸다.
기존 학습 행을 순서대로 보존하고 새 자료를 뒤에 추가했다. 검증 JSONL은 바이트까지
기존과 동일하게 유지했다. 알림·빈 창 외 작업 창의 정확한 중복이 없고 지정 의미 계열의
연결 대상이 모두 기존 train인지 확인했다. 이는 모든 의미 중복이 없다는 증명은 아니다.
기존 최종 테스트의 ID 목록은 누출 검사에만 사용했으며 테스트 추론은 하지 않았다.

자연 분포 직접 숫자 분류, Qwen/Qwen3-0.6B, NF4 이중 양자화·BF16,
LoRA r=8/alpha=16/dropout=0.05, q_proj/v_proj, 학습률 0.0002, seed 42,
batch 1, 최대 768토큰을 유지하고 기반 모델에서 새 어댑터를 학습했다.
보강 모델은 긴급도 58스텝·관련도 174스텝으로 한 번씩 학습했다.
학습 횟수 증가 효과를 확인하기 위해 기존 자료만 같은 58·174스텝으로 학습한
대조군도 실행했다. 대조군은 약 1.16 epoch, 보강 모델은 1 epoch이다.

| 방식 | 긴급도/관련도 스텝 | 정책 정답 | 긴급 오차단 | 불필요 오통과 | 관련도 정확 일치 | 알림 평균 처리 |
| --- | --- | --- | --- | --- | --- | --- |
| 이전 자연 분포 | 50/150 | 20/24 | 4/6 | 0/12 | 21/24 | 0.150초 |
| 기존 자료 동일 스텝 대조 | 58/174 | 20/24 | 4/6 | 0/12 | 21/24 | 0.154초 |
| 승인 보강 추가 | 58/174 | 20/24 | 4/6 | 0/12 | 21/24 | 0.153초 |

검증 24건의 점수 예측은 세 조건에서 모두 같았다. 긴급도는 전부 1,
관련도는 8건이 5·16건이 1이었다. 같은 알림의 문맥 3종 간 긴급도는
8개 그룹 모두 동일했지만 계약서 금액 오류와 누수 위험을 무관·빈 창에서 여전히 놓쳤다.
이번 단일 seed·소규모 검증에서 보강 효과를 확인하지 못했으며 일반적인 무효로 단정하지 않는다.

동일 스텝 대조군의 긴급도/관련도 학습 loss는 1.9066/0.8826,
학습 시간은 22.31초/66.81초였다. 보강 모델은 loss 1.9873/1.1533,
학습 시간 23.30초/66.56초였다. 학습 설정·보강 출처·승인 해시·예외 ID·
검증 입력·정답·점수 확률 argmax와 재계산한 정책 지표를 확인했다.

### 전체 학습 자료의 적합도 진단

검증 성능과 별도로 각 어댑터를 단독 로딩해 전체 학습 자료를 진단했다.
아래 수치는 학습 정확도이며 일반화 성능이 아니다.

| 점수 | 전체 학습 정확 일치 | 보강 학습 정확 일치 | 예측 분포 |
| --- | --- | --- | --- |
| 긴급도 | 23/58 | 3/8 | 58건 모두 1 |
| 관련도 | 149/174 | 20/24 | 116건이 1, 58건이 5 |

긴급도는 학습의 긴급 원문 20개 모두 4 미만으로 예측했고, 새 긴급 원문 3개도
모두 놓쳤다. 보강 자료를 넣었지만 현재 짧은 학습 설정에서 긴급도 구분을 배우지 못했다.
관련도는 학습 점수 2·3·4를 하나도 출력하지 않았다. 특히 사용자 검수에서 관련도 2로
지정한 `sep02_06_related` 광고도 5로 예측했다. 7번 관련도 4 예외 역시 5로 예측했다.
이 진단은 원인 확정이 아니며 검증의 정책 20/24만으로 학습이 충분하다고 볼 수 없다.

새 준비 자료는 `filtering_training/outputs/v3_digit_supplement_01/prepared/`,
설정·가중치·예측·확률·학습 적합도 진단·검증된 비교는
`filtering_training/outputs/runs/2026-10-04-qwen06-digit-supplement-01/`에 보존했다.
원본 v3 SHA-256과 기존 승인·검수·가중치는 유지했다. 필터링 테스트 56개가 통과했다.
두 모델 모두 앱 적용 후보로 채택하지 않는다. 다음은 다섯 점수 학습을 유지하면서
긴급 여부(4 이상)를 구분하는 보조 목표를 긴급도 학습에 추가하는 짧은 대조 실험이다.
관련도 어댑터와 검증 분할을 고정하고 긴급 오차단과 불필요 오통과를 함께 비교한다.
최종 테스트 추론·앱 적용·커밋·푸시는 하지 않았다.

## 2026-10-04 — 긴급도 4 이상 보조 목표 추가 대조

이전 승인 보강 자료와 자연 분포, 58스텝을 유지하고 긴급도 loss만 바꿨다.
기존 다섯 숫자 점수의 cross entropy에 점수 1~3과 4~5의 합산 확률을
구분하는 이진 cross entropy를 가중치 1로 더했다. 그룹 logit은 각 후보의
logsumexp이며 0부터 시작하는 정답 인덱스 3 이상을 긴급으로 처리한다.
보조 가중치는 실행 전 1로 고정했다. 점수 1~5 출력과 argmax 추론,
긴급도 ≥4 또는 관련도 ≥4인 기존 PASS 기준은 유지했다.

Qwen/Qwen3-0.6B, NF4 이중 양자화·BF16, LoRA r=8/alpha=16/dropout=0.05,
q_proj/v_proj, 학습률 0.0002, seed 42, batch 1, 최대 768토큰을 유지했다.
긴급도 어댑터는 기반 모델에서 새로 학습했고 관련도는 이전 보강 실험의
174스텝 어댑터를 직접 참조했다. 관련도 가중치를 복사하거나 재학습하지 않았다.
검증 원문 8개·문맥 24건과 승인 보강 자료를 포함한 학습 긴급도 58개를 고정했다.

| 방식 | 정책 정답 | 긴급 오차단 | 불필요 오통과 | 긴급도 학습 정확 일치 | 알림 평균 처리 |
| --- | --- | --- | --- | --- | --- |
| 기존 다섯 점수 loss | 20/24 | 4/6 | 0/12 | 23/58 | 0.153초 |
| 긴급 여부 보조 loss 추가 | 20/24 | 4/6 | 0/12 | 23/58 | 0.148초 |

검증 점수 예측은 기준선과 같았고 긴급도는 24건 모두 1이었다.
학습 자료에서도 단독 어댑터로 58건 모두 1을 예측했고 긴급 원문 20개를
모두 놓쳤다. 승인 보강 원문 8개의 긴급도 정확 일치는 3/8이며 새 긴급
원문 3개 역시 모두 놓쳤다. 학습 적합도 수치는 검증 성능과 구분한다.
보조 목표 추가만으로 쏠림이 해결되지 않았으며 원인을 확정하지 않는다.

보조 모델 학습 loss는 2.8774, 시간은 22.38초였다. 기준선 loss 1.9873과는
목표 항목이 달라 수치를 직접 성능 비교에 쓰지 않는다. 그룹의 경계가 4인지,
보조 항이 긴급 후보의 확률을 올리는 gradient를 주는지, 가중치 0이 기존
loss와 같은지 테스트했다. 필터링 테스트 58개가 통과했다.
데이터·프롬프트·학습 행·스텝·관련도 어댑터 해시·예측 확률 argmax와
원본 검증 입력·정답·정책 지표·학습 적합도를 재확인했다.

사용자의 파일 수 지적에 따라 새 코드·문서 파일은 만들지 않고 기존
loss·학습/평가·실행·비교·진단·테스트 파일을 확장했다. 기존 코드를 삭제하지 않았다.
설정·가중치·예측·학습 로그·적합도·검증된 비교만 새 실행 폴더
`filtering_training/outputs/runs/2026-10-04-qwen06-digit-threshold-01/`에 보존했다.
다음은 원래 train의 소수 긴급 사례를 반복 학습해 실제 정답 점수를 배울 수 있는지
점검하는 짧은 실험이다. 이를 통해 학습 경로와 현재 학습량의 문제를 먼저 확인한다.
이번 모델은 앱 적용 후보로 채택하지 않는다.
최종 테스트 추론·앱 적용·커밋·푸시는 하지 않았다.

## 2026-10-04 — 소수 사례 반복으로 긴급도 학습 경로 진단

기존 train에서 정답 긴급도 1·4·5의 첫 사례를 각각 선택하고,
선택한 3건만 60스텝(20회 순회) 반복 학습했다. 검증 오류를 복사하지 않았고
새 데이터나 코드 파일도 만들지 않았다. 기존 실행기·학습기·진단기를 확장했다.
선택 ID는 `v3_08_related`, `v3_05_related`, `v3_01_related`이며
승인 보강 준비 train의 인덱스는 7·4·0이다. 각 점수 1건씩, 중복 행 추가 없이
Trainer의 반복 epoch로 학습했다. 다른 55건은 이번 학습에 넣지 않았다.

Qwen/Qwen3-0.6B, NF4 이중 양자화·BF16, LoRA r=8/alpha=16/dropout=0.05,
q_proj/v_proj, 학습률 0.0002, seed 42, batch 1, 최대 768토큰과
다섯 숫자 cross entropy를 유지했다. 보조 목표 가중치는 0이며 기반 모델에서
새 어댑터를 시작했다. 관련도 어댑터는 재학습하지 않았다.

| 선택 사례 | 정답 긴급도 | 이전 보강 모델 | 반복 학습 후 | 정답 후보 조건부 확률 |
| --- | --- | --- | --- | --- |
| 검사 완료·위협 없음 (`v3_08_related`) | 1 | 1 | 1 | 0.99865 |
| 진행 중 발표 회의 링크 변경 (`v3_05_related`) | 4 | 1 | 4 | 0.99623 |
| 결제 API 오류·주문 실패 (`v3_01_related`) | 5 | 1 | 5 | 0.99784 |

이 세 사례의 정확 일치는 이전 보강 모델 1/3에서 3/3으로 바뀌었고,
선택 긴급 원문 2개 모두 4 이상으로 예측했다. 이 수치는 학습 사례를
외울 수 있는지 확인한 적합도이며 검증 정확도나 일반화 성능이 아니다.
평균 학습 loss는 0.6900, 마지막 스텝 loss는 약 0.00214,
학습 시간은 22.89초였다. LoRA 텐서 112개 중 112개가 실제 변경됐고
전체 변경량 L2는 1.46503이었다. 현재 학습 경로가 정답 점수 변화를 만들 수
있다는 근거이며, 이전 쏠림의 원인이 학습 횟수 부족 하나라고 확정할 수는 없다.

진단을 위해 기존 train 58건을 단독 어댑터로 다시 추론했을 때 정확 일치는
13/58, 긴급 원문 오차단은 11/20이었다. 이번에 학습하지 않은 55건도 기존
train 자료이므로 새로운 검증 분할로 취급하지 않는다. 3건만 외운 모델의
성능을 이전 전체 학습 모델과 공정한 성능 비교로 해석하지 않는다.

설정·가중치·로그·전체 train 진단 예측·선택 3건 적합도는
`filtering_training/outputs/runs/2026-10-04-qwen06-digit-fit-01/`에 보존했다.
선택한 ID·train 인덱스·정답·20회 반복·실제 가중치 갱신·예측 확률 argmax와
이전 모델의 동일 사례 예측을 재확인했다. 필터링 테스트 60개가 통과했다.
다음은 전체 긴급도 train 58건에서 보조 목표 없이 3회 순회(174스텝)하는
짧은 대조이다. 기존 1회 학습과 비교하며 관련도 어댑터와 검증 분할을 유지한다.
이번 진단 모델은 앱 적용 후보로 채택하지 않는다.
검증·최종 테스트 추론·앱 적용·커밋·푸시는 하지 않았다.

## 2026-10-04 — 전체 긴급도 3회 학습 비교와 데이터 확장 전 검수 준비

승인 보강 자료를 포함한 긴급도 train 58건을 자연 분포로 3회(174스텝)
학습해 기존 1회(58스텝)와 비교했다. 기반 모델에서 새 어댑터로 시작했으며
Qwen/Qwen3-0.6B, NF4 이중 양자화·BF16, LoRA r=8/alpha=16/dropout=0.05,
q_proj/v_proj, 학습률 0.0002, seed 42, batch 1, 최대 768토큰을 유지했다.
다섯 숫자 cross entropy만 사용했고 긴급 여부 보조 목표 가중치는 0이다.
관련도는 기존 보강 자료의 174스텝 어댑터를 직접 재사용했다.
스텝 증가에 따라 선형 learning-rate decay의 길이와 각 사례 노출 횟수도 달라진다.

| 방식 | 정책 정답 | 긴급 오차단 | 불필요 오통과 | 긴급도 학습 정확 일치 | 긴급도 학습 시간 |
| --- | --- | --- | --- | --- | --- |
| 전체 1회 학습 | 20/24 | 4/6 | 0/12 | 23/58 | 23.30초 |
| 전체 3회 학습 | 20/24 | 4/6 | 0/12 | 23/58 | 67.88초 |

검증 24건의 점수 예측은 두 조건에서 모두 같았다. 긴급도는 전부 1이었다.
전체 train 58건에서도 전부 1로 예측했고 긴급 원문 20개를 모두 놓쳤다.
평균 학습 loss는 1.9873에서 1.7436으로 낮아졌지만 점수 구분은 개선되지 않았다.
단일 seed의 3회 학습 결과만으로 더 많은 데이터·학습량의 효과를 부정할 수는 없으나,
현재 문제를 반복 횟수 부족 하나로 확정할 수도 없다. 이 방식은 적용 후보로 채택하지 않는다.

실행기·비교기에 옵션을 추가해 새 코드 파일 없이 진행했다. 학습 행·프롬프트·
모델 설정·174스텝·3 epoch·고정 관련도 어댑터 해시·원본 검증 입력·정답·
예측 확률 argmax·정책 지표·단독 어댑터 train 적합도를 재검증했다.
설정·가중치·로그·예측·진단·검증된 비교는
`filtering_training/outputs/runs/2026-10-04-qwen06-digit-epochs3-01/`에 보존했다.
필터링 테스트 61개가 통과했다. 최종 테스트 추론·앱 적용·커밋·푸시는 하지 않았다.

### 다음 단계: 원문 다양성을 우선한 확장

사용자는 5천 건을 최대한 다양하게 만들고 이후 2만·3만 개로 누적 학습할 계획이라고 밝혔다.
확장 목표를 서로 다른 알림 원문 5천 개로 잡고 문맥 변형 행 수는 별도로 관리한다.
같은 문장의 상품명·인명 치환만으로 원문 다양성을 채우지 않으며 분야·상황·
행동·발신자·문체·기한·피해 정도와 현재 목적의 조합을 다양하게 구성한다.
생성 자료는 합성 후보이며 인간 검수 여부·의미 계열·출처·수정 이력을 보존한다.
추가 버전에서도 원문 계열의 분할을 유지하고 검증·최종 테스트를 학습에 섞지 않는다.

기존 검수 생성기를 확장해 원문 8개·문맥 10행의 소규모 후보를
`filtering_training/outputs/v3_expansion_review_01/`에 준비했다. 긴급도 1~5는
각 2행, 관련도 1~5를 모두 포함하며 같은 주제의 농담·인접 분야 광고·직접 도움·
정확한 작업 대상의 경계를 담았다. 문맥이 없어도 긴급한 사례와 무관한 작업 중
긴급한 사례도 넣었다. 기존 데이터와 본문·작업 창의 정확한 중복 및 지정 계열의
train 연결을 확인했다. 모든 의미 중복이 없다는 증명은 아니며 사용자 검수가 필요하다.
이 10행은 미승인·학습 미사용 상태이며 검수 후 대량 생성을 진행한다.
관련도 1 행을 기계적으로 두 배 붙이는 구성은 확장 기본값으로 사용하지 않는다.
기존 7번 농담 관련도 4 지정은 해당 승인 사례의 예외로 유지한다.
5천 후보로 본 학습할 때는 현재 숫자 logit 방식의 쏠림을 해결할 학습 방식도 함께 검토하며,
현 검증 원문 8개만으로 대량 모델의 품질을 단정하지 않도록 독립 검증 확대를 설계한다.

## 2026-10-04 — 생성 기준 승인과 5천 원문 합성 후보 준비

사용자의 `지금 좋다. 이대로 진행하자`를 직전 표시한 원문 8개·문맥 10행의
점수 승인 및 그 기준에 따른 생성 진행 승인으로 기록했다. 원본 검수표와 후보는
보존하고 `filtering_training/outputs/v3_expansion_review_01_approved/`에 확정했다.
8번의 표시 카테고리 `보안/인증`은 코드의 허용값 `시스템/보안`으로 이름만
정규화했다. 본문·긴급도·관련도는 바꾸지 않았으며 2개 문맥 행의 이름 변환을
approval.json에 남겼다. 이 승인은 대량 생성 자료의 개별 인간 검수로 확장하지 않는다.

기존 검수 생성기에 bulk 경로를 추가해 20개 분야의 서로 다른 작업 100개와
사건 유형 50개를 조합한 원문 5천 개를 생성했다. 단순 인명·상품명·숫자 치환만으로
수량을 채우지 않고 작업 대상·확인 항목·발생 사건·필요 행동·기한과 손실 상황을
바꿨다. 다만 조합형 합성이며 5천 개가 모두 독립적으로 작성된 사건은 아니다.
일반 작업·학습·취미·구매·여행·집안 관리·개인 재무·행사·설계·시스템 관리 등을 담았다.

| 항목 | 결과 |
| --- | --- |
| 원문 | 5,000개 |
| 기본 문맥 행 | 5,000행 |
| 별도 문맥 대비 행 | 500행 |
| 합계 | 5,500행 |
| 분야 / 구체적 작업 / 사건 유형 | 20 / 100 / 50 |
| 기본 행의 긴급도 1~5 | 각 1,000행 |
| 기본 행의 관련도 1~5 | 각 1,000행 |
| 대량 자료 중 개별 인간 검수 원문 | 0개 |

모든 행을 합치면 긴급도 점수별 1,100행, 관련도 1은 1,500행,
관련도 2~5는 각 1,000행이다. 원문마다 무관·빈 창 2행을 기계적으로
붙이지 않고 전체 중 500개에만 문맥 대비 행을 추가했다.
이 500쌍은 실제로 작업 창이 다르며 긴급도와 카테고리는 모두 유지됐다.
같은 주제의 농담·지난 모임 소식은 관련도 3 이하로 제한했다. 광고 마감만으로
긴급도를 높이지 않으며 광고의 관련도 4·5에는 실제 구매 비용·정확한 구매 조건
비교 목적을 지정했다. 빈 창은 휴식으로 추정하지 않았다.

첫 후보 `v4_diverse_5000_candidates_01/`을 만든 뒤 창 정보만으로 점수가
추측되는 구성을 검사했다. 빈 창을 제외한 행 중 같은 창 제목에 점수가 하나만
나오는 행은 3,000/4,724(63.51%)였다. 같은 창에서 다른 알림·작업 대상에 따라
점수가 달라지도록 구성한 개선본 `v4_diverse_5000_candidates_02/`에서는
2,710/4,727(57.33%)로 낮아졌고, 여러 점수를 갖는 창은 100개에서 252개로 늘었다.
이는 전체 후보에서 측정한 창-점수 결합의 진단이며 검증 정확도·편향 해결의 증명이 아니다.
두 버전은 같은 원문 풀의 문맥 개선 이력이며 1만 원문으로 중복 집계하지 않는다.

### 품질 확인과 보존 범위

개선본의 스키마·카테고리 허용값·ID·원문 5천 개·행 수·점수 분포·출처 해시·
500쌍의 긴급도 불변을 독립 확인했다. 정규화 본문 완전 중복, 원본 v3와
기존 승인 묶음의 본문·작업 창 완전 중복, 알림 긴급도 모순, 동일 입력의
정답 충돌은 발견되지 않았다. 모든 의미 중복·점수 정확성이 검증됐다는 뜻은 아니다.
후보의 provenance는 규칙 제안 라벨·개별 미검수·train 전용으로 명시했다.
각 원문에 작업·사건 계열·안정적 recipe hash를 기록해 이후 2만·3만 개 확장 시
계열과 이력을 유지하도록 했다. 추가 수량에는 새 작업과 사건 의미를 늘려야 하며
같은 원문을 단어만 바꾸거나 기존 버전과 중복 합산해서 채우지 않는다.

개선본은 `filtering_training/outputs/v4_diverse_5000_candidates_02/`의
`candidates.jsonl`, `lineage.jsonl`, `manifest.json`, `audit.json`에 있다.
후보 SHA-256은 `c4a4d41e11a2fbb3b865592fdb292ccb5bdd881e2320bbd4670396bae9932e27`이며 실제 파일과 대조했다.
기존 v3와 이전 승인 자료 해시는 유지했다. 새 코드 파일 없이 생성기·확정 도구·
기존 테스트를 확장했고 필터링 테스트 62개가 통과했다.

### 남은 제한과 다음 단계

사건 표현의 반복과 규칙 기반 라벨, 창 문구 편향이 남아 있다. 긴급도 5는
디지털 원본 손실·개인정보 공개 상황에 편중됐고 `기타` 카테고리는 포함되지 않았다.
균등 점수 분포도 실제 사용 환경의 알림 비율이 아니라 학습 범위 확보를 위한 구성이다.
개선본은 합성 후보 단계이며 자동 검사 통과를 학습 자료 확정으로 간주하지 않는다.
다음에는 이 후보의 라벨·분야 편중을 표본 확인하고 필요한 사례를 보완한 뒤,
현재 숫자 logit의 쏠림을 고려한 학습 방식과 함께 학습 자료를 확정한다.
이 작업에서 대량 학습·검증/최종 테스트 추론·앱 적용·커밋·푸시는 하지 않았다.

## 2026-10-04 — 5천 후보 점수 경계 검수

- 대상: `v4_diverse_5000_candidates_02`에서 경계 사례 10건 선정. 사용자 검수 대기.
- 발견: 참고 자료·주석 제안의 관련도 4가 모호함. 조사 오류와 긴급도 5의 디지털 사고 편중도 보완 필요.
- 제안: 검수표 3·6번 관련도 4→3. 아직 원본 라벨에는 반영하지 않음.
- 자료: `filtering_training/outputs/v4_diverse_5000_candidates_02/review_batch_01.md`.
- 다음: 점수 검수 반영과 생성 규칙 보완 후 학습 자료 확정. 이번에는 학습·평가 없음.
- 이후 로그는 설정·핵심 결과·판단·다음 작업만 간단히 기록.

## 2026-10-04 — 표본 점수 수정

- 확정: 2번 긴급도 2, 3번 관련도 4, 8·9번 긴급도 5.
- 반영: `v4_diverse_5000_candidates_02/review_feedback_01.json`에 동일 알림 문맥 변형까지 7개 행 기록. 원본 후보 보존.
- 확인: 수정 값·문맥 간 긴급도 일치·UTF-8 재읽기 통과. 학습·평가 없음.
- 다음: 학습 자료 준비 시 수정 내역 적용. 나머지 표본과 전체 후보는 미확정.

## 2026-10-04 — 5천 후보 수정본

- 자료: `v4_diverse_5000_candidates_03`, 원문 5천·문맥 5,500행. 이전 버전과 합쳐 세지 않음.
- 수정: 검수 점수 반영, 같은 사건 유형 334행 긴급도 규칙 보완, 642행 조사 교정. 유형 전파는 합성 라벨로 유지.
- 확인: 테스트 62개 통과. 원본 보존·ID/문맥/관련도 유지·문맥 간 긴급도 일치 확인.
- 판단: 긴급도 5의 1,200원문 중 디지털 사고 1,000개로 편중이 남음. 학습·평가 없음.
- 다음: 건강·안전·금전·필수 일정의 다양한 긴급 사례 보강 및 검수 후 학습 자료 확정.

## 2026-10-04 — 비디지털 긴급 사례 검수

- 자료: `v4_urgent_review_01`, 독립 작성 원문 10개·문맥 13행. 건강·화재·금전·접수/시험과 비긴급 대비 사례.
- 확인: 테스트 63개 통과. 3개 원문의 빈 창 변형에서 긴급도 유지, 기존 자료와 본문/창 완전 일치 없음, 지정 유사 계열은 학습 분할에만 연결.
- 상태: 사용자 검수 대기. 5천 후보에 미합산, 학습·평가 없음.
- 다음: 점수 검수 후 신규 사건 유형을 확장해 긴급도 5의 디지털 사고 편중 보완.

## 2026-10-04 — 비디지털 긴급 사례 확정

- 승인: 사용자 “이대로 확정”. 원문 10개·문맥 13행, 제안 점수 그대로 확정.
- 자료: `v4_urgent_review_01_approved`. 기존 후보·검수표·원본 분할 보존.
- 확인: 승인 문구·점수·출처 해시·문맥 간 긴급도 일치 확인. 학습·평가 없음.
- 다음: 승인한 사건 유형을 다양하게 확장해 5천 후보의 긴급 사례 편중 보완.

## 2026-10-04 — 긴급 상태 대비 보완

- 자료: `v4_training_pool_01`, 통합 원문 5,090개·문맥 5,753행. 20개 상황×4개 사건 상태로 합성 원문 80개 추가, 승인 자료 10개 포함.
- 확인: 테스트 63개 통과. ID·본문 수·승인 범위·문맥 간 긴급도 일치 확인, 자동 감사에서 충돌 없음. 기존 자료 보존.
- 판단: 긴급도 5 원문 1,225개 중 디지털 사고 1,000개로 원자료 편중은 남음. 학습 추출 시 25% 상한 적용 계획이며 아직 미구현.
- 다음: 분할·샘플링·입력 길이 확인 후 0.6B 학습. 추가 검수는 실제 문제 발견 시 진행. 이번 학습·평가 없음.

## 2026-10-04 — 대량 학습 입력 준비 완료

- 자료: `v4_training_pool_01/prepared`, 긴급도 4,230건·관련도 5,937건. 기존 검증 24행과 최종 테스트 분할 보존.
- 샘플링: 긴급도 5 중 디지털 사고 77/308=25%. 926개는 긴급도 학습 입력에서만 제외, 원자료와 기존 학습 사례 보존.
- 확인: 테스트 65개 통과, 검수 수정 유지, 긴급도 문맥/ID 제외. 실제 토크나이저 최대 길이 283/422토큰(제한 768).
- 다음: 기존 실행기의 `pool` 모드로 0.6B 두 점수를 각각 1회 학습 후 고정 검증 평가. 이번 학습·모델 평가 없음.

## 2026-10-04 — 대량 분리 학습·검증 완료

Qwen3-0.6B에서 두 점수를 각각 1회 학습했다. 긴급도는 알림만, 관련도는 알림과 작업 문맥을 입력한다. 두 입력 모두 알림 ID를 제외한다.

| 학습 설정 | 긴급도 | 관련도 |
| --- | ---: | ---: |
| 학습 건수 / 스텝 | 4,230 / 4,230 | 5,937 / 5,937 |
| 학습 시간 | 31.3분 | 40.5분 |
| 평균 학습 손실 | 0.160 | 0.619 |

같은 검증 24건으로 비교했다. 긴급도 ≥4 또는 관련도 ≥4이면 PASS이며, 최종 테스트는 사용하지 않았다.

| 검증 지표 | 이전 소량 분리 학습 | 이번 대량 분리 학습 |
| --- | ---: | ---: |
| 통과·차단 정답 | 20/24 (83.3%) | 19/24 (79.2%) |
| 긴급 알림을 잘못 차단 | 4/6 | 2/6 |
| 차단 대상의 불필요한 통과 | 0/12 | 2/12 |

이번 점수별 정답은 긴급도 15/24·관련도 20/24이며, 동일 알림의 긴급도는 8개 원문 모두 문맥 간에 유지됐다. 긴급 누락은 줄었지만 전체 정확도는 개선되지 않아 앱 적용은 보류한다. 데이터·샘플링·입력 정책이 함께 바뀌어 학습량만의 효과로 단정할 수 없다.

### 대표 입출력 3개

서로 다른 오답 유형을 실제 검증 예측에서 골랐다. 아래 입력은 관련도 모델의 실제 사용자 메시지이며, 긴급도 모델에는 같은 입력의 `notification`만 전달했다. 출력은 두 모델이 선택한 숫자를 평가 코드가 합친 값이다. 모델이 사유나 JSON 문장을 생성한 결과는 아니다. 사례 ID는 기록용이며 모델에 전달하지 않았다.

**1. 계약 금액 오류 — 무관한 작업 중에도 긴급한 알림 (`v3_20_unrelated`)**

입력:

```json
{
  "notification": {
    "app_name": "Jira",
    "sender": "검토 담당",
    "title": "납품 계약 검토",
    "body": "오늘 발송할 계약서의 금액이 견적과 다릅니다. 발송 전 수정해 주세요",
    "timestamp": "2026-10-03T04:09:00Z"
  },
  "context": {
    "active_process": "chrome.exe",
    "window_title": "집 누수 수리 접수 - Chrome",
    "last_updated": "2026-10-03T04:08:57Z",
    "duration_seconds": 180,
    "recent_processes": ["chrome.exe"]
  }
}
```

실제 출력:

```json
{"urgency_score": 3, "relevance_score": 1}
```

정답은 긴급도 4·관련도 1 → PASS. 예측은 BLOCK으로, 발송 전 금액 수정이 필요한 알림을 놓쳤다.

**2. 일반 로그인 승인 — 긴급도 경계를 과대평가 (`v3_36_unrelated`)**

입력:

```json
{
  "notification": {
    "app_name": "Microsoft Authenticator",
    "sender": "",
    "title": "로그인 승인 요청",
    "body": "회사 계정 로그인을 승인하시겠습니까? 본인이 요청하지 않았다면 거부하세요.",
    "timestamp": "2026-10-03T04:57:00Z"
  },
  "context": {
    "active_process": "explorer.exe",
    "window_title": "발표 파일 압축하기 - 파일 탐색기",
    "last_updated": "2026-10-03T04:56:57Z",
    "duration_seconds": 180,
    "recent_processes": ["explorer.exe"]
  }
}
```

실제 출력:

```json
{"urgency_score": 4, "relevance_score": 1}
```

정답은 긴급도 3·관련도 1 → BLOCK. 예측은 PASS로, 침해가 확인되지 않은 일반 승인 요청을 긴급 알림으로 통과시켰다.

**3. 집중 음악 추천 — 현재 목적에 직접 도움이 되는 알림 (`v3_50_related`)**

입력:

```json
{
  "notification": {
    "app_name": "Spotify",
    "sender": "",
    "title": "새 재생목록 추천",
    "body": "조용한 피아노와 함께하는 집중 시간 — 추천 재생목록을 들어보세요",
    "timestamp": "2026-10-03T05:39:00Z"
  },
  "context": {
    "active_process": "Spotify.exe",
    "window_title": "집중 음악 재생목록 비교 - Spotify",
    "last_updated": "2026-10-03T05:38:57Z",
    "duration_seconds": 65,
    "recent_processes": ["Spotify.exe"]
  }
}
```

실제 출력:

```json
{"urgency_score": 1, "relevance_score": 2}
```

정답은 긴급도 1·관련도 4 → PASS. 예측은 BLOCK으로, 현재 재생목록 비교에 직접 유용한 추천을 차단했다.

- 확인: 관련도 문맥 변형 ID 누출 제거 후 테스트 66개 통과. 중단·재시도 기록 보존.
- 자료: `outputs/runs/2026-10-04-qwen06-pool-02/comparison.json`, `validation.predictions.jsonl`(filtering_training 기준). 완료 어댑터는 `urgency`와 `relevance_id_clean`.
- 다음: 검증 원문을 학습에 복사하지 않고 긴급도 3↔4 경계와 현재 목적에 직접 도움이 되는 관련도 기준을 점검한다.

## 2026-10-04 — 1만 원문 확장·새 검증 준비

| 항목 | 준비 결과 / 목표 |
| --- | --- |
| 새 검증 | 첫 후보 원문 10개·문맥 30행, 사용자 검수 대기 / 원문 100개 목표 |
| 학습 후보 | 기존 5,090개 보존 / 새 원문 4,910개 추가해 10,000개 목표, 신규 생성은 아직 0개 |
| 최근 작업 문맥 | 사용자 확인: 최근 앱 3개의 이름·창 제목 수집 가능. 기존 대량 후보 5,753행은 최근 앱 목록이 모두 비어 있음 |
| 비교 계획 | 같은 데이터의 누적 학습·새 어댑터 학습 비교. 관련도는 기록 없음·앱 이름만·앱과 창 제목 비교 |

- 자료: `filtering_training/outputs/v5_validation_review_01/review.md`. 첫 후보는 최근 창 확장 전 자료이며 학습·모델 평가 미사용, 최종 테스트 보존.
- 확인: 기존 원문·정보 창과 완전 일치 없음, 문맥 간 긴급도 유지 및 기존 자료 해시 보존 확인. 의미상 완전 독립을 보장하지 않으며 새 사건 계열은 이후 학습 후보에서도 제외해야 함.
- 다음: 첫 묶음 점수 검수 후 최근 창 입력·작업 연속/전환 사례를 보강. 학습과 평가 결과는 완료 시 이 로그에 추가.

## 2026-10-04 — 첫 검증 확정·최근 창 입력 준비

| 항목 | 결과 |
| --- | --- |
| 사용자 승인 | “검수해보니 괜찮은듯” — 첫 검증 원문 10개·문맥 30행, 점수 변경 없이 확정 |
| 최근 창 입력 | 오프라인 입력에 앱 이름·창 제목 최대 3개 보존. 기록 없음/앱 이름만/이름과 제목 비교 지원, 긴급도에는 전달하지 않음 |
| 다음 검수 | 새 원문 5개·최근 기록 변형 10건. 현재 작업을 뒷받침하는 기록·무관한 기록·빈 창 포함, 사용자 미승인 |
| 확인 | 관련 테스트 69개 통과. 승인 원본·후보·분할 보존, 학습·모델 평가 없음 |

- 자료: `filtering_training/outputs/v5_validation_review_01/approval.json`, `filtering_training/outputs/v5_history_review_01/review.md`. 최근 창 입력은 오프라인 실험용이며 앱 실행 코드에는 아직 연결하지 않음.
- 다음: 최근 창 사례 점수 검수 후 검증 원문 100개와 학습 후보 1만 원문 구성을 확장. 이번 10건은 기록에 끌리는 오판 점검용이며 성능 개선을 입증한 결과가 아님.

## 2026-10-04 — 최근 창 검수 수정

| 검수 번호 | 관련도 수정 | 수정 후 판단 |
| --- | --- | --- |
| 4 | 4 → 1 | BLOCK |
| 5 | 3 → 2 | BLOCK |
| 6 | 3 → 1 | BLOCK |
| 8 | 5 → 1 | BLOCK |
| 9 | 1 → 5 | PASS |

- 사용자 지정대로 반영, 긴급도·알림·창 정보는 유지. 기존 후보와 원래 검수표는 보존하고 같은 폴더의 `feedback.json`·`revised.jsonl`에 변경 이력·수정 점수를 저장.
- 수정 기준: 현재 창 제목만으로 관련도를 고정하지 않고 최근 창의 작업 흐름을 함께 판단. 빈 창에서도 최근 기록이 정확한 대상·행동에 연결되면 높은 관련도가 가능함. 장식용 광고는 분야가 같아도 직접 유용한 자료로 취급하지 않음.
- 범위: 지정한 5건의 사용자 점수 수정이며 나머지 점수의 최종 승인으로 확대 해석하지 않음. 모든 최근 기록이 현재 창보다 우선한다는 규칙도 아님. 학습·모델 평가 없음.

## 2026-10-04 — 최근 창 검수 2번 추가 수정

| 검수 번호 | 관련도 수정 | 판단 |
| --- | --- | --- |
| 2 | 4 → 1 | BLOCK |

- 사용자 “0”은 완전 무관이라는 뜻이며 기존 최저점 1 사용으로 확인. 최근 빵 만들기 작업 흐름과 견학 영문 안내는 무관하다고 지정. 긴급도 2 유지.
- 같은 묶음의 수정은 총 6건. 첫 제안·이전 수정본·사용자 발언을 기존 피드백 파일에 보존. 학습·모델 평가 없음.
- 판단: 현재 창 제목만으로 점수를 고정한 생성 기준이 잘못됐음. 이 묶음을 그대로 대량 확대하지 않고 현재 창과 최근 작업 흐름을 함께 반영하는 기준부터 재정리.

## 2026-10-04 — 최근 작업 흐름 관련도 규칙 반영

| 항목 | 변경 |
| --- | --- |
| 판단 순서 | 작업 대상·행동 확인 → 알림의 도움 확인 → 관련도 점수 |
| 충돌·빈 창 | 현재 창 단어 일치만으로 점수 고정 금지. 빈 창도 최근 제목이 정확한 대상·행동에 연결되면 관련도 4~5 가능 |
| 적용 | 최근 창 오프라인 입력·숫자 선택 프롬프트·기존 생성기에 수정 6건 기준 반영 |
| 확인 | 관련 테스트 35개 통과. 기존 학습 프롬프트·긴급도 입력 보존, 추가 파일 없이 기존 계획 문서 갱신 |

- 현재 창과 최근 앱 개수의 다수결로 목적을 정하지 않으며, 앱 이름만으로 작업을 추측하지 않음. 정보 제거 비교에서는 판단 근거 부족을 별도 구분.
- 학습·모델 평가 없음. 기존 후보·수정 이력·검증 분할 유지. 다음은 이 기준을 적용한 새로운 사건·작업 목적의 후보 확장.

## 2026-10-04 — 새 검증 후보 100원문 구성

| 항목 | 결과 |
| --- | --- |
| 최근 창 검수 확정 | 수정된 원문 5개·10행, 사용자 다음 작업 지시 기록 |
| 검증 후보 풀 | 기존 승인 15원문·40행 + 신규 85원문·340행 = 100원문·380행 |
| 신규 구성 | 25개 작업 영역, 작업 일치·최근 작업 전환·빈 창과 최근 기록·모든 기록 없음 |
| 검수 범위 | 신규 첫 10행 제시. 신규 85원문은 아직 개별 검수 전이며 전체 승인으로 취급하지 않음 |

- 자료: `filtering_training/outputs/v5_validation_pool_01/review.md`. 학습 전용 1만 원문 확장과 별도인 검증 후보 풀임.
- 확인: 기존 승인 자료·보호 원본 해시 유지, 정규화 본문·현재 정보 창 중복 없음, 긴급도 문맥 불변 및 ID 제거 확인. 현재·최근 창과 사건 계열을 향후 학습에서도 제외할 대상으로 기록. 검수 전 제목의 내부 사건 구분값 제거 이력 보존. 무관한 최근 작업을 단일 식빵 창으로 반복하지 않고 25개 작업을 교차 사용해 창만 보고 차단 점수를 외우는 편향을 줄임.
- 학습·모델 평가·최종 테스트 사용 없음. 다음은 제시한 경계 사례 점수 검수 후 나머지 검증 라벨 감사와 학습 후보 확장.

## 2026-10-04 — 신규 검수 10건 승인·학습 격리 확인

| 항목 | 결과 |
| --- | --- |
| 승인 | 사용자 “ㅇㅇ 이번엔 괜찮네”, 표시한 10행·8원문 점수 그대로 승인 |
| 검수 범위 | 기존 승인 15원문 + 신규 표시 원문 8개. 같은 원문의 미표시 문맥과 나머지 후보는 개별 승인하지 않음 |
| 격리 감사 | 기존 학습 후보 5,753행과 검증 후보 380행 사이 ID·정규화 원문·현재/최근 창·선언된 사건 계열 완전 일치 0건 |
| 확인 | 승인 범위·최근 창 혼입 탐지 등 해당 테스트 11개 통과 |

- 승인 기록은 검증 풀의 기존 폴더 `approval.json`에 저장. 기존 감사 도구에 전체 검증 후보를 학습에서 제외하는 검사를 추가하고 결과를 같은 풀 manifest에 기록. 의미상 모든 유사성이 배제됐다는 보장은 아님.
- 신규 학습 원문 4,910개 생성과 모델 학습·평가는 아직 진행하지 않음. 다음은 검증 원문·98개 현재/최근 창·지정 계열을 제외해 학습 후보를 확장.

## 2026-10-04 — 학습 후보 1만 원문 확장

| 항목 | 결과 |
| --- | --- |
| 자료 | 기존 5,090원문 보존 + 신규 합성 4,910원문 = 10,000원문·17,477행 |
| 신규 구성 | 30영역·150작업 × 일반 문장 틀 30종 = 4,500원문. 긴급 사건 41종 × 말투 10종 = 410원문 |
| 신규 긴급도 분포 | 1: 1,050 / 2: 750 / 3: 1,500 / 4: 1,200 / 5: 410원문 |
| 입력 길이 | 실제 신규 입력 최대 긴급도 283·관련도 689토큰, 768 초과 0건 |
| 확인 | 기존 후보 보존, 검증 원문·현재/최근 창·명시 계열 혼입 0건. 대표 10행 검수 준비 |

- 자료: `filtering_training/outputs/v5_training_pool_01/`. 기존 생성기를 확장하고 후보·계열·설정·검수표만 저장. 신규 개별 인간 검수 0개, 학습·모델 평가·최종 테스트 사용 없음.
- 한계: 일반 문장 틀은 각각 150작업에 반복됨. 긴급 410원문도 독립 사건 410종이 아닌 41종의 말투 변형. 원문 개수가 독립 사건 다양성을 뜻하지 않음. 초안·주석의 정확한 작업 연결, 최근 앱 중복, 부자연스러운 긴급 창 제목을 검수 전 자체 점검하고 수정 이력 보존.
- 다음: 대표 점수 검수와 반복 문장 보완 후 학습 입력 준비. 문장 반복 목표를 충족했다고 주장하지 않으며 현재 후보를 그대로 학습하지 않음.

## 2026-10-04 — 학습 후보 대표 10건 검수 확정

| 항목 | 결과 |
| --- | --- |
| 4번 관련도 | 3 → 4, BLOCK → PASS |
| 5번 관련도 | 2 → 3, BLOCK 유지 |
| 승인 범위 | 표시한 10행·9원문. 나머지 8행 점수 승인, 긴급도 모두 유지 |
| 확인 | 관련도 2건만 변경, 원본 후보·기존 자료·미표시 문맥 보존 |

- 기존 학습 후보 폴더에 수정 자료와 승인 이력 저장. 4번 점수는 해당 사례의 사용자 지정으로 기록하며 다른 농담 사례에 일괄 적용하지 않음.
- 학습·모델 평가 없음. 다음은 반복 문장 보완과 학습 입력 준비.

## 2026-10-04 — 문장 반복 보완·최근 창 학습 입력 준비

| 항목 | 결과 |
| --- | --- |
| 표현 보완 | 신규 일반 원문 4,494개 수정. 30사건 × 표현 8종, 표현당 최대 19원문 |
| 보존 | 승인 원문 9개와 모든 문맥·점수 유지. 기존 후보·수정본·학습 자료 보존 |
| 학습 입력 초안 | 긴급도 9,140원문 / 관련도 17,661문맥 행. 최대 283 / 694토큰 |
| 검증·확인 | 기존 검증 유지, 별도 검증 후보 380행 학습 제외. 중복 감사 통과, 해당 테스트 39개 통과 |

- 기존 생성·준비·감사 도구를 확장하고 같은 자료 폴더에 보완본·학습 초안 저장. 최근 창은 관련도 입력에 보존하고 긴급도에는 문맥·ID를 넣지 않음. 150작업·30사건의 독립 다양성은 늘지 않았으며 긴급 41사건의 말투 변형은 그대로임.
- 새 표현 8건 검수 준비. 단순 보관 안내의 기존 합성 관련도 4는 직접 도움 근거가 부족해 3을 제안하며, 승인 전 데이터 점수는 유지. 검수 중인 초안의 학습 실행을 차단함.
- 학습·모델 평가·최종 테스트 사용 없음. 검수 후 보관 안내 라벨을 점검하고 누적 학습 설정으로 이어감.

## 2026-10-04 — 보완 문장 검수 점수 수정

| 항목 | 결과 |
| --- | --- |
| 관련도 | 2·3·4번 → 4. 3번은 기존 데이터 점수 4 유지, 검수 제안 3 철회 |
| 긴급도 | 7·8번 → 5. 동일 알림의 모든 문맥에 적용 |
| 학습 입력 | 긴급도 원문 2건·관련도 문맥 2건 수정, 나머지 점수 유지 |

- 원래 제안·수정 이력을 같은 폴더의 승인 기록에 보존. 누적 검수 범위는 표시한 18행·17원문이며 미표시 합성 후보까지 개별 승인으로 확대하지 않음.
- 학습·모델 평가 없음. 다음은 기존 어댑터를 이어 학습할 설정 준비.

## 2026-10-04 — 검증 24행에서 74행으로 확대

| 항목 | 결과 |
| --- | --- |
| 기본 검증 | 기존 24행 + 추가 검수 50행 = 74문맥 행·31원문 |
| 구성 | 긴급 알림 문맥 17행, PASS 34행 / BLOCK 40행 |
| 승인 범위 | 새 검증 후보 중 승인된 50행만 사용. 미검수 330행은 정식 지표에서 제외 |
| 확인 | 현재 학습 후보와 검증 후보 380행 격리 감사 통과, 관련 테스트 27개 통과 |

- 사용자 지시에 따라 기존·새 검증을 합친 74행을 기본으로 평가하도록 기존 평가 도구 수정. 기존 24행 결과도 보조 기록하며 최종 테스트는 유지. 같은 원문의 문맥 변형이 있어 독립 알림 74개로 해석하지 않음.
- 기존 검증 폴더에 승인 자료 50행을 확정하고 원래 후보·승인 이력 보존. 학습·모델 추론은 실행하지 않았음. 다음 누적 학습 전후를 동일한 74행으로 비교.

## 2026-10-04 — 누적 학습 직전 준비 완료·실행 보류

| 항목 | 준비 결과 |
| --- | --- |
| 학습 구성 | 긴급도 신규 4,910 + 이전 검수 26 = 4,936건. 관련도 신규 11,724 + 이전 검수 47 = 11,771행 |
| 시작 가중치 | 기존 pool-02 긴급도·ID 제거 관련도 어댑터 유지, 옵티마이저 새로 초기화 |
| 설정 | 학습률 5e-5, 각 자료 1회, 기존 QLoRA 설정·숫자 CE 유지 |
| 검증 | 학습 전후 동일한 74행, 이후 비교 표·실제 입출력 3건 기록하도록 연결 |
| 확인 | 두 작업 dry-run 통과, 관련 테스트 45개 통과 |

- 기존 도구에 어댑터 이어 학습·해시 확인·읽기 전용 점검·전후 비교를 추가. 같은 데이터 폴더에 선택한 학습 자료를 저장하고 기존 원본·전체 학습 초안·가중치를 보존. 이전 긴급도 제한에서 빠졌던 검수 원문 1개는 승인 자료에서 복원.
- 사용자 지시에 따라 실제 학습·모델 추론·예약 실행 없음. 실행 명령과 세팅은 기존 `filtering_training/docs/SCORE_SEPARATION_PLAN.md` 맨 아래에 기록.

## 2026-10-05 — 누적 학습 시작·사전 검증

| 항목 | 학습 전 결과 |
| --- | --- |
| 전체 검증 | PASS/BLOCK 55/74, 긴급 오차단 2/17, 불필요한 통과 7건 |
| 기존 / 추가 검증 | 19/24 / 36/50 |
| 실제 학습 설정 | 긴급도 4,936건·관련도 11,771행, 기존 어댑터에서 1회, 학습률 5e-5 |

| 대표 실제 입력 | 현재 창 | 정답 긴급도/관련도 | 학습 전 출력 |
| --- | --- | --- | --- |
| 오늘 발송할 계약서의 금액이 견적과 다릅니다. 발송 전 수정해 주세요 | 집 누수 수리 접수 - Chrome | 4/1, PASS | 3/1, BLOCK |
| 회사 계정 로그인을 승인하시겠습니까? 본인이 요청하지 않았다면 거부하세요. | 발표 파일 압축하기 - 파일 탐색기 | 3/1, BLOCK | 4/1, PASS |
| 별항로 보드게임 규칙 외우다 내 머리도 우주로 떠난 듯ㅋㅋ | 별항로 보드게임 자원 교환 규칙 확인 - Chrome | 1/3, BLOCK | 1/4, PASS |

- 실행: `filtering_training/outputs/runs/2026-10-05-qwen06-continual-01/`. 원격 develop 추가 변경 없음, 준비 자료·부모 어댑터 해시 확인. 사용자 학습 지시로 실행 시작했고 최종 테스트는 사용하지 않음. 사전 검증 원문을 학습에 추가하지 않음.


## 2026-10-05 — 어댑터 누적 학습·검증 비교

| 항목 | 학습 전 | 학습 후 |
| --- | --- | --- |
| PASS/BLOCK 정답 | 55/74 | 66/74 |
| 기존 / 추가 검증 | 19/24 / 36/50 | 19/24 / 47/50 |
| 긴급도 / 관련도 점수 정답 | 62/74 / 50/74 | 62/74 / 65/74 |
| 긴급 알림 차단 | 2/17 | 2/17 |
| 불필요한 통과 | 7 | 4 |
| 긴급도 학습 건수·손실·시간 | — | 4936건 / 0.0597 / 31.3분 |
| 관련도 학습 건수·손실·시간 | — | 11771건 / 0.2124 / 88.3분 |

**대표 입출력 1** (v3_20_related)
- 알림: 오늘 발송할 계약서의 금액이 견적과 다릅니다. 발송 전 수정해 주세요
- 현재 창: 납품 계약 금액 확인 - Excel
- 최근 창: 없음
- 정답 긴급도/관련도: 4/5; 학습 전: {'urgency_score': 3, 'relevance_score': 4}; 학습 후: {'urgency_score': 3, 'relevance_score': 5}

**대표 입출력 2** (hist05_01_supporting)
- 알림: 견학 참가자에게 안전모 착용 위치를 설명하는 영문 예문을 보냈습니다. 안내 문구 작성에 참고하세요. 지금 답할 필요는 없습니다.
- 현재 창: 견학 안전 안내 영문 문구 작성 - Word
- 최근 창: chrome.exe: 견학 안전모 착용 지점 안내 - Chrome / AcroRd32.exe: 견학 이동 동선.pdf / ms-teams.exe: 견학 안내 문구 검토 - Teams
- 정답 긴급도/관련도: 2/4; 학습 전: {'urgency_score': 2, 'relevance_score': 3}; 학습 후: {'urgency_score': 2, 'relevance_score': 4}

**대표 입출력 3** (v3_36_unrelated)
- 알림: 회사 계정 로그인을 승인하시겠습니까? 본인이 요청하지 않았다면 거부하세요.
- 현재 창: 발표 파일 압축하기 - 파일 탐색기
- 최근 창: 없음
- 정답 긴급도/관련도: 3/1; 학습 전: {'urgency_score': 4, 'relevance_score': 1}; 학습 후: {'urgency_score': 5, 'relevance_score': 1}

- 결과: `filtering_training\outputs\runs\2026-10-05-qwen06-continual-01`. 동일한 검증 74행 사용, 최종 테스트 미사용.
- 최종 확인: 기존 어댑터·준비 자료 해시 유지. 두 어댑터 각각 112개 LoRA 텐서의 실제 변경 확인. 학습 원문·문맥 수와 1회 학습 기록 일치. 검증 74행은 독립 원문 31개의 문맥 변형임.
- 남은 오류: 계약 금액 오류 알림을 긴급도 3으로 판단해 무관·빈 창 2행을 계속 차단. 로그인 승인 요청은 긴급도 4→5로 과대평가. 12행의 정책 오류가 해결됐지만 장식용 머그컵 홍보 1행은 새로 오통과. 관련도 개선에 비해 긴급 판단의 핵심 문제는 남았으며 앱에는 적용하지 않음.

## 2026-10-05 — 부족 유형 보강 준비 재개

| 항목 | 상태 |
| --- | --- |
| 검수 후보 | 새 원문 8개·문맥 10행, `outputs/v6_targeted_review_01/` |
| 집중 유형 | 긴급 금전 오류/여유 있는 수정, 일반 인증/확인된 악용, 직접 유용한 자료/광고·무관 잡담, 최근 창 문맥 |
| 다음 규모 | 검수 후 신규 약 2,000원문 보강 및 기존 사례 20~30% 혼합 검토 |

- 기존 생성기 재사용. 점수는 미승인 제안이며 원문 중복·동일 알림 긴급도 일관성 검사 통과. 기존 데이터·가중치 보존, 학습·모델 평가 미실행.

## 2026-10-05 — 현재 창 우선 검수 반영

| 항목 | 수정 |
| --- | --- |
| 보강 검수 10번 | 관련도 1 → 4, 긴급도 2 유지, BLOCK → PASS |
| 관련도 기준 | 현재 창을 우선, 최근 3개 창은 보조. 최근 작업이 다르다는 이유만으로 직접 도움을 낮추지 않음 |

- 원래 후보·기존 학습 결과 보존, 같은 검수 폴더에 수정본·이력 기록. 나머지 9행의 승인은 추정하지 않음. 학습·모델 평가 없음.

## 2026-10-05 — 다섯 부족 유형 보강 후보 생성

| 항목 | 결과 |
| --- | --- |
| 신규 후보 | 원문 2,000개: 긴급 집중 1,000 / 관련 집중 1,000 |
| 문맥 | 점수 제안 10,000행 + 관련도 미정 2,000행 별도 보관 |
| 검사 | 원문 중복 없음, 기존 신규 검증 창 중복 없음, 동일 알림 긴급도·현재 창 우선 대비 일관성 통과 |
| 상태 | 10행 검수 제시, 전체는 합성 후보. 학습·모델 평가 미실행 |

- 기존 생성기를 확장하고 `outputs/v6_targeted_pool_01/`에 모음. 모두 모호한 창은 무관 점수를 붙이지 않음. 기존 작업 어휘·상태 틀의 조합이므로 독립 작성 2,000사례로 주장하지 않음. 다음은 검수 및 기존 점수·프롬프트 충돌 점검.

## 2026-10-05 — VS Code 파일명 문맥 보정

| 항목 | 결과 |
| --- | --- |
| 편집기 제목 | 설명형 제목 200행을 일반/단서 있는 파일명으로 보정 |
| 정보 부족 | 일반 파일명 + 다른 작업 기록 25행의 관련도 정답 제외 |
| 최신 후보 | 원문 2,000개, 점수 제안 9,975행 / 정보 부족 2,025행 |

- 같은 후보 폴더의 수정본 사용, 원래 후보·검수 이력 보존. 앱·파일 확장자만으로 작업을 추측하지 않음. 학습·모델 평가 없음.

## 2026-10-05 — 새 학습 방식 실행 준비

| 항목 | 설정 |
| --- | --- |
| 유지 | 직전 데이터·라벨·프롬프트, 최신 어댑터, 학습률 5e-5·LoRA r8/q,v |
| 변경 | 누적 배치 1→8, 실제 1 epoch, warmup 5%, 누적 loss 평균 처리 |
| 학습량 | 긴급도 4,936샘플/617회, 관련도 11,771샘플/1,472회 업데이트 |
| 확인 | 누적 8개·마지막 3개 gradient 검증, 관련 테스트 23개 통과 |

- 실행 폴더: `outputs/runs/2026-10-05-qwen06-accumulated-01/`. 사용자 요청으로 학습 전후 동일 검증 74행 비교를 실행하며, 결과와 대표 입출력은 완료 후 아래에 추가. 새 2천 원문 후보와 최종 테스트는 사용하지 않음.


## 2026-10-05 — 누적 배치 8·warmup 학습과 직전 결과 비교

| 항목 | 학습 전 | 학습 후 |
| --- | --- | --- |
| PASS/BLOCK 정답 | 66/74 | 66/74 |
| 긴급 알림 차단 | 2/17 | 2/17 |
| 불필요한 통과 | 4 | 4 |
| 기존 / 추가 검증 정답 | 19/24 / 47/50 | 19/24 / 47/50 |
| 긴급도 / 관련도 점수 정답 | 62/74 / 65/74 | 62/74 / 65/74 |

| 학습 방식 | 직전 학습 | 이번 학습 |
| --- | --- | --- |
| 배치 / gradient accumulation | 1 / 1 | 1 / 8 |
| 학습량 | 고정 step으로 전체 1회 | 실제 1 epoch, 샘플 처리 수 검증 |
| Warmup | 없음 | optimizer step의 5% |
| Loss 평균 | 샘플별 업데이트 | microbatch 평균을 누적, 마지막 부분 묶음 포함 |
| 유지 | 학습률 5e-5, linear, LoRA r8·q/v | 동일, 데이터·라벨·프롬프트도 동일 |
| 긴급도 학습 건수·손실·시간 | 4936건 / 0.0597 / 31.3분 | 4936건 / 0.0094 / 30.9분 |
| 긴급도 optimizer 업데이트 | 4936 | 617 |
| 관련도 학습 건수·손실·시간 | 11771건 / 0.2124 / 88.3분 | 11771건 / 0.0119 / 85.8분 |
| 관련도 optimizer 업데이트 | 11771 | 1472 |

**대표 입출력 1** (hist05_04_supporting)
- 알림: 별항로에서 탐사 카드를 쓴 뒤 이동할 수 있는지 설명한 공식 FAQ 링크야. 그 규칙 확인할 때 참고해. 급하게 답하지 않아도 돼.
- 현재 창: 별항로 탐사 카드 사용 후 이동 규칙 확인 - Chrome
- 최근 창: AcroRd32.exe: 별항로 탐사 카드 규칙.pdf / Notion.exe: 별항로 카드와 이동 질문 정리 - Notion / KakaoTalk.exe: 별항로 규칙 질문 모임
- 정답 긴급도/관련도: 2/5; 학습 전: {'urgency_score': 2, 'relevance_score': 4}; 학습 후: {'urgency_score': 2, 'relevance_score': 3}

**대표 입출력 2** (v3_20_unrelated)
- 알림: 오늘 발송할 계약서의 금액이 견적과 다릅니다. 발송 전 수정해 주세요
- 현재 창: 집 누수 수리 접수 - Chrome
- 최근 창: 없음
- 정답 긴급도/관련도: 4/1; 학습 전: {'urgency_score': 3, 'relevance_score': 1}; 학습 후: {'urgency_score': 3, 'relevance_score': 1}

**대표 입출력 3** (valpool05_02_reference_matching)
- 알림: 유약 시험편을 같은 조명에서 비교하는 촬영 방법을 정리했어요. 발색을 비교할 때 참고하세요. 나중에 확인해도 됩니다.
- 현재 창: 청색 유약 시험편 발색 비교 - Chrome
- 최근 창: chrome.exe: 청색 유약 시험편 발색 비교 - Chrome / Notion.exe: 도예 작업 대상과 진행 내용 - Notion / explorer.exe: 도예 작업 자료 - 파일 탐색기
- 정답 긴급도/관련도: 2/4; 학습 전: {'urgency_score': 2, 'relevance_score': 3}; 학습 후: {'urgency_score': 2, 'relevance_score': 4}

- 결과: `filtering_training\outputs\runs\2026-10-05-qwen06-accumulated-01`. 동일한 검증 74행 사용, 최종 테스트 미사용.
- 직전 가중치에 같은 데이터를 한 번 더 학습한 결과이며 추가 학습 효과도 포함. 설정 변경만의 효과로 단정하지 않음. 새 2천 원문 후보와 현재 창 우선 프롬프트는 이번 학습에 넣지 않음.
- 해석: 정책 개선 1행(유약 촬영 자료)·퇴행 1행(보드게임 FAQ)으로 전체 66/74 유지. 계약 금액 오류의 긴급 오차단은 그대로이며 로그인 긴급도는 5→4로 완화됐지만 정답 3에 미달. 학습 loss 감소만으로 실사용 성능 개선을 주장하지 않음.
- 최종 확인: 총 학습 116.6분(직전 119.7분). 데이터·라벨·프롬프트·기존 가중치 해시 유지, 두 어댑터 각각 112/112 LoRA 텐서 변경 확인. 관련도 마지막 3개를 포함해 1 epoch 처리 건수·업데이트 수 일치. 검증 74행은 원문 31개의 문맥 변형이며 최종 테스트는 보존.
