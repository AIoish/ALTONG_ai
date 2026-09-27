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
{"notification":{"id":"rapid_0001","app_name":"Slack","sender":"","title":"예약 서비스 요청 실패","body":"예약 서비스에 들어온 요청이 정상 처리되지 않습니다. 오늘 처리할 요청이 멈춰 빠른 확인이 필요합니다.","timestamp":"2026-09-27T12:00:23Z"},"context":{"active_process":"EXCEL.EXE","window_title":"개인 가계부 - Excel","last_updated":"2026-09-27T12:00:20Z","duration_seconds":65,"recent_processes":["EXCEL.EXE"]}}
```

목표 출력:

```json
{"urgency_score":4,"relevance_score":1,"category":"긴급 업무","ai_summary_reason":"현재 작업과 무관하며 현재 작업을 잠시 중단하고 확인할 가치가 있습니다."}
```

### 예시 2 — 개인 중요, 작업 정보 없음

입력:

```json
{"notification":{"id":"rapid_1571","app_name":"은행 앱","sender":"","title":"교통비 결제 처리 보류","body":"교통비 결제 처리가 보류돼 확인이 필요합니다. 오늘 안에 거래 내역을 확인해 주세요.","timestamp":"2026-09-27T22:02:13Z"},"context":{"active_process":"","window_title":"","last_updated":"2026-09-27T22:02:10Z","duration_seconds":0,"recent_processes":[]}}
```

목표 출력:

```json
{"urgency_score":3,"relevance_score":1,"category":"개인 중요","ai_summary_reason":"현재 작업 정보가 없어 연관성을 알 수 없으며 비교적 빠른 확인이 필요합니다."}
```

### 예시 3 — 광고/홍보, 현재 작업과 관련

입력:

```json
{"notification":{"id":"rapid_2601","app_name":"쇼핑 앱","sender":"","title":"휴대전화 케이스 쿠폰","body":"휴대전화 케이스 구매에 적용할 쿠폰이 도착했습니다. 이번 주 할인 상품을 확인해 보세요.","timestamp":"2026-09-28T04:37:03Z"},"context":{"active_process":"chrome.exe","window_title":"휴대전화 케이스 상품 비교 - Chrome","last_updated":"2026-09-28T04:37:00Z","duration_seconds":25,"recent_processes":["chrome.exe","explorer.exe","KakaoTalk.exe"]}}
```

목표 출력:

```json
{"urgency_score":1,"relevance_score":4,"category":"광고/홍보","ai_summary_reason":"현재 열어 둔 작업과 직접 관련이 있으며 언제 확인해도 큰 문제가 없습니다."}
```

동일 상황 유형이 학습·검증·평가에 중복되지 않도록 계보 파일의 `scenario`
기준으로 분할했다. 임시 분할은 학습 2,398건, 검증 302건, 평가 300건이다.
별도의 24건 독립 평가 세트는 여기에 넣지 않았다. 두 파일과 분할 결과는
`filtering_training/outputs/` 아래에 있으며 Git에서 제외된다.