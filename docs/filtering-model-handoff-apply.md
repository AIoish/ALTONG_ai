<!-- 용도: 앱 코드를 포함하지 않는 긴급도5점·관련도3점 ONNX 전달 파일의 적용 계약과 검증 방법.
생성일: 2026-11-17 (Asia/Seoul) -->

# ALTONG 필터 모델 적용 안내

이 패키지는 긴급도 전용 ONNX와 관련도 전용 ONNX, 토크나이저, 검증 자료만 제공합니다.
앱 소스 수정은 포함하지 않습니다. Python/PyTorch와 ONNX의 변환 일치 검사를 수행하며,
C# 토큰·점수 일치 및 실제 WPF 앱 동작은 앱 팀에서 아래 순서로 확인해야 합니다.
모델을 학습하거나 기존 체크포인트를 덮어쓰지 않습니다.

## 1. 제공 파일

```text
models/filtering/
  urgency-v1/
    model-fp32.onnx
    tokenizer/              tokenizer.json 및 tokenizer_config.json 등
    manifest.json
    verification.json
  relevance-v1/
    model-fp32.onnx
    tokenizer/
    manifest.json
    verification.json
verification-fixtures.json  공개 가능한 합성 입력·토큰·기대 모델 점수
verification-report.json    변환 일치 검사 결과
manifest.json               패키지 버전·평가 상태·파일 해시
checksums.json              패키지 내부 파일 SHA256
APPLY_GUIDE.md              본 안내
```

기존 모델 폴더를 덮어쓰지 않고 위 두 폴더를 앱 저장소에 추가하세요.
기존 `klue-base-safety-v1/model-fp32.onnx` 하나를 새 모델로 바꾸는 방식으로 적용할 수 없습니다.
두 ONNX는 다른 가중치를 사용하므로 독립 세션으로 로딩하고 종료 시 각각 해제해야 합니다.
각 모델은 약440MB FP32이며 함께 배포할 때 모델 가중치 공간과 메모리 사용이 증가합니다.
양자화나 인코더 공유는 이 전달에 포함되지 않습니다.

## 2. 모델과 점수

| 항목 | 긴급도 | 관련도 |
| --- | --- | --- |
| 출처 | 기존 안전 보강 모델의 긴급도 출력만 추출 | 교사 자료 비중5%, seed42, 선택2epoch |
| 입력 | 알림만 | 알림·현재 창·최근 창 |
| 점수 | 1~5 | 1~3 |
| ONNX 출력 | `logits`: float32 `[batch,5]` | `logits`: float32 `[batch,3]` |
| 계산 | 각 행 `i`마다 `argmax(logits[i,:]) + 1` | 동일 |

모두 `input_ids`, `attention_mask` 두 입력을 받으며 타입은 Int64 `[batch,sequence]`입니다.
`token_type_ids`는 전달하지 않습니다. 출력 이름은 두 파일 모두 `logits`입니다.
옛 출력 `[batch,2,5]` 및 head 인덱스 선택은 제거하고 각 세션의2차원 출력을 읽으세요.
동률이면 먼저 등장한 최댓값의 인덱스를 사용합니다.
manifest의 `decoding` 값 `argmax(logits[0,:]) + 1`은 batch1 기준 표기이며, batch가 2 이상이면 행마다 같은 방식으로 계산합니다.
관련도 manifest의 `unused_relevance_head_removed: false`는 처음부터 단일 관련도 출력이라 제거할 head가 없다는 뜻입니다.
ZIP에 포함된 2026-11-17 안내에는 위 두 설명이 없지만 모델·입력·출력 계약은 같습니다.

권장 통과 정책은 `urgency >= 4 || relevance == 3`입니다. 이는 이번 전달의 **정책 제안**이며,
앱 팀에서 채택 여부를 명시적으로 결정해 적용하세요. 기존 `relevance >= 4`는 새 범위에서 항상 거짓입니다.
관련도1~3을 임의로1~5로 환산하지 마세요. 0은 기존 앱의 미평가 표시로만 유지합니다.

## 3. 입력 JSON 계약

프롬프트나 설명 문장을 붙이지 않습니다. compact JSON 문자열 자체를 토큰화합니다.
UTF-8, 한글 등 비ASCII 문자는 그대로 유지하며 Python `ensure_ascii=False, separators=(',', ':')`와
키 순서·이스케이프까지 맞춰야 합니다. 입력 문자열이 달라지면 모델 입력 토큰도 달라집니다.
앱의 기존 `ModelInputBuilder.Quote()`를 재사용할 수 있지만 구조는 아래와 맞추세요.

긴급도 입력:

```json
{"notification":{"app_name":"Slack","sender":"검증 담당","title":"검증용 합성 알림","body":"수정한 합계표 공유했습니다.","timestamp":"2026-11-17T00:00:00Z"}}
```

관련도 입력:

```json
{"notification":{"app_name":"Slack","sender":"검증 담당","title":"검증용 합성 알림","body":"수정한 합계표 공유했습니다.","timestamp":"2026-11-17T00:00:00Z"},"current_window":{"active_process":"EXCEL.EXE","window_title":"분기 합계 검토.xlsx - Excel"},"recent_windows":[]}
```

- notification 키 순서: `app_name`, `sender`, `title`, `body`, `timestamp`.
- `id` 등 앱 내부 식별자와 별도 metadata는 입력에 넣지 않습니다.
- 앱에서 null 문자열을 빈 문자열로 정규화한 뒤 입력하세요. 위 명세와 공개 fixture는 문자열을 사용합니다.
- timestamp는 수신 시각의 UTC 초 단위 `yyyy-MM-ddTHH:mm:ssZ`입니다. 매 추론마다 현재 시각으로 바꾸지 마세요.
- 현재 창은 `active_process`, `window_title`만, 최근 창의 각 항목은 `app_name`, `window_title`만 포함합니다.
- 최근 창은 앱이 제공한 최신순을 유지합니다. 현재 창과 프로세스·제목이 모두 같은 항목을 먼저 제외하고 최대3개를 취합니다. 비교는 현재 앱처럼 대소문자를 구분합니다.
- `recent_processes`, 작업 지속 시간, 추가 점수·사유는 넣지 않습니다.
- 빈 현재 창/최근 창은 그대로 빈 문자열/배열로 전달하며 알림 내용으로 작업 맥락을 만들어 채우지 않습니다.
- 각 ONNX에 동봉된 tokenizer를 사용합니다. `[CLS]`·`[SEP]`를 포함해 최대512토큰이며 묵시적으로 자르지 않습니다.
- 배치 padding은 tokenizer의 padding ID, attention_mask는 실제 토큰1/padding0을 사용합니다.

## 4. 앱 팀 수정 범위

현재 develop은 아래 소스를 실제로 연결합니다. 문서에 남은 Qwen/초기 Small 모델 예시는 적용 기준이 아닙니다.

| 파일 | 적용할 변경 |
| --- | --- |
| `integration/csharp/Altong.Filtering/Input.cs` | `ContextInput.RecentProcesses`와 JSON의`recent_processes` 제거. 위 두 입력 직렬화 생성 |
| `integration/csharp/Altong.Filtering/KlueFilter.cs` | 한 세션/두 head를 두 세션/각 단일 출력으로 변경. 각 manifest·모델·tokenizer 해시 확인, 입력/출력 shape·타입 확인, 점수 범위 및 통과 정책 적용 |
| `integration/csharp/ClientAdapter/KlueFilterEngine.cs` | 두 모델 경로 및 tokenizer 경로 전달, 최근 프로세스 목록 생성 제거, `FilterResult`에 긴급도1~5/관련도1~3 반환 |
| `src/Altong.Client/App.xaml.cs` | 두 모델의 로딩 경로 설정 및 수명 관리 |
| `src/Altong.Client/Altong.Client.csproj` | 두 모델 폴더의 모든 파일을 빌드·publish 출력으로 복사 |
| `integration/csharp/Verification/Program.cs` 등 검사 | 옛 공동5점 모델의 shape/입력/기대 점수 대신 이 패키지의 fixture와 계약으로 검사 |

`ClientAdapter/KlueFilterEngine.cs`는 csproj에서 `Services/AI/Filtering/KlueFilterEngine.cs`로 링크됩니다.
같은 이름의 별도 엔진 구현을 새로 만들지 말고 실제 링크된 소스를 수정하세요.

`CurrentContext`는 이미 현재 창과 최근 창을 제공합니다. `FilterResult`와 DB 점수 컬럼은 정수이므로
새 점수 저장을 위한 타입 변경은 필수가 아닙니다. `NotificationPipelineCoordinator`는 엔진의
`IsPassed`를 사용하므로 통과 판단은 엔진에서 일관되게 계산하세요.
기존5점 관련도 기록과 새3점 기록은 모델/점수체계 버전을 함께 남겨 구분하는 것을 권장합니다.
옛 기록의 점수를 새 기준으로 재해석하거나 덮어쓰지 마세요.

기존의 백그라운드 로딩·요청 직렬화·최대 대기열·취소·시간초과·모델 누락 시 통과 처리를 유지하세요.
실패 점수를 임의5점으로 만들지 말고 기존 계약의0=미평가로 표시하세요.
두 세션 모두 `Microsoft.ML.OnnxRuntime` CPU로 실행할 수 있습니다. 검증 버전은 각 manifest에 있습니다.
앱 사용자에게 Python/PyTorch/CUDA 설치는 필요하지 않습니다.

## 5. 앱 팀 적용 검증

1. ZIP의 SHA256과 내부 `checksums.json`, 각 모델 manifest 해시를 확인합니다.
2. fixture의 `state`에서 두 JSON을 생성하여 `expected.<task>.input_text`와 문자열이 같은지 확인합니다.
3. 각 tokenizer의 `input_ids`, `attention_mask`가 fixture와 같은지 확인합니다. fixture의 recent_windows에는 현재 창 중복과4개 초과 사례도 있습니다.
4. 두 ONNX의 `argmax + 1`이 fixture의 `score`와 같은지 확인합니다. fixture 점수는 모델 예측이며 사람 정답이 아닙니다.
5. `dotnet build`, 기존 필터링 검사와 수정한 네이티브 검사를 실행합니다.
6. 실제 Windows 알림 수신→현재/최근 창 수집→점수→통과/차단→독 표시·DB 저장을 확인합니다.
7. 초기 로딩·시간초과·취소·긴 입력·모델 누락·집중 모드 전환·앱 종료도 확인합니다.

`verification-report.json`의 `csharp_parity_tested=false`, `full_app_tested=false`는 정상적인 현재 상태입니다.
이 패키지의 성공은 모델 변환 일치를 뜻하며 앱 통합 완료를 뜻하지 않습니다.
앱 팀에서 위 검증을 마친 뒤 실제 알림의 오탐·누락을 확인하세요.

## 6. 관련도 후보의 평가 상태

개발 검증1032건 중1012건(98.06%), 사용자 검수 실제47건 중40건(85.11%) 정답입니다.
실제3점20건 중14건을 맞혀 재현율70%, 실제3점 오탐1건입니다.
전체 기준은 통과했지만 기존 합성 검증 퇴행과 실제 새 오탐으로 추가 출처별 채택 기준은 보류됐습니다.
단일 시드의 개발 결과이며, 별도의 독립 실제 최종시험이나 정식 운영 승인 결과가 아닙니다.
기존 긴급도 판단을 보존하면서 새 관련도를 앱에서 검증할 **연동 후보**로 전달합니다.

원본 CSV·실제 알림 본문·학습 데이터·PyTorch 체크포인트·개인 경로는 공개 전달 파일에 포함하지 않습니다.
