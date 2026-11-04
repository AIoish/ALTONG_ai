<!-- 용도: 단독 자모 관련도 이상에 대한 앱 호출 경로·입력 불일치·모델 통제 실험을 기록한다.
생성일: 2026-11-04 -->

# 단독 자모 관련도 진단

> 이 문서는 입력 수정 전의 진단 기록이다. 이후 v13에서 현재 창을 제외하는 규칙을 학습·검증·평가에 적용했다.
> 현재 구현과 확인 결과는 [현재 학습 안내](filtering-windows-only-2026-11-04.md)를 따른다. 재학습과 실사용 증상 재현은 아직 완료하지 않았다.

사용자가 실사용 빌드는 ALTONG_client develop 그대로라고 확인했다.
확인한 커밋은 `803b55562b6dc7f4b3c97e53694ec1bb9c90502f`다.
해당 커밋은 긴급도5/관련도5 모델을 연결하며, 관련도3 모델이나 3단계 점수 변환은 연결되어 있지 않다.
이 문서의 실험 점수는 모두 기존 5단계 원시 점수다.

## 판단

점수를 덮어쓰는 룰베이스 연결이나 긴급도/관련도 헤드 혼동은 발견하지 못했다.
그러나 **앱과 학습 데이터의 최근 창 구성 불일치**와 **초단문 학습 공백**은 확인했다.
사용자의 실제 두 알림 입력은 확보하지 못했고, 글자만 바꾼 고정 입력 실험에서는 ㅇ/ㅁ 점수 차이를 재현하지 못했다.
따라서 해당 실사용 현상을 특정 코드 한 줄 또는 모델만의 문제로 확정하지 않는다.

## 실제 호출 경로

App.xaml.cs가 KlueFilterEngine을 만들어 NotificationPipelineCoordinator에 주입한다.
RuleBasedFilterEngine.cs는 존재하지만 이 경로에서 생성하거나 호출하지 않는다.
실패 시 규칙 엔진으로 전환하는 대신 점수0(미평가)과 통과 결과를 반환한다.

KlueFilter는 알림만 넣은 실행에서 긴급도 헤드0, 알림+창 맥락 실행에서 관련도 헤드1을 쓴다.
argmax+1로 1~5를 선택한다. 점수 후처리·3단계 변환·문자별 점수 규칙은 없다.
KlueFilterEngine.ToResult, 파이프라인, DB는 점수를 그대로 복사한다.
요약용 Qwen의 카테고리·요약·일정 추출은 별도 경로다.

- [실제 필터 등록](https://github.com/AIoish/ALTONG_client/blob/803b55562b6dc7f4b3c97e53694ec1bb9c90502f/src/Altong.Client/App.xaml.cs#L171)
- [ONNX 실행·점수 선택](https://github.com/AIoish/ALTONG_client/blob/803b55562b6dc7f4b3c97e53694ec1bb9c90502f/integration/csharp/Altong.Filtering/KlueFilter.cs#L50)
- [모델 맥락·결과 변환](https://github.com/AIoish/ALTONG_client/blob/803b55562b6dc7f4b3c97e53694ec1bb9c90502f/integration/csharp/ClientAdapter/KlueFilterEngine.cs#L28)

## 앱에서 입력이 달라질 수 있는 지점

관련도에는 발신자·방 외에 본문, 수신 시각, 현재 프로그램·창 제목, 직전 창 최대3개가 들어간다.
아래는 실제 두 알림에서 발생했다고 확정한 원인이 아니라 코드에서 확인한 경로다.

- 창을 500ms 간격으로 관찰하고 3초 안정화 후 갱신한다. 짧은 전환 중에는 직전 안정 맥락을 유지한다.
- 파이프라인은 CaptureNow()가 아닌 `_windowTracker.CurrentContext`를 읽는다.
- 카카오톡은 OCR 완료 뒤 알림 이벤트를 발행하므로 맥락은 최초 팝업 포착 시점과 다를 수 있다.
- OCR은 본문뿐 아니라 방·발신자도 읽는다. 줄 위치로 나누며 못 읽으면 대체 문구를 넣을 수 있다.
- 문자 보정 규칙은 있지만 원래의 ㅇ/ㅁ을 점수별로 다르게 취급하는 규칙은 없다. 두 문자 모두 유효 본문으로 허용한다.
- Paddle OCR은 신뢰도를 계산하지만 해당 호출에서는 RecognizeLine()의 문자열만 사용하고, 비어 있지 않으면 채택한다.

[맥락 읽기](https://github.com/AIoish/ALTONG_client/blob/803b55562b6dc7f4b3c97e53694ec1bb9c90502f/src/Altong.Client/Services/Notifications/NotificationPipelineCoordinator.cs#L62),
[OCR 이후 알림 발행](https://github.com/AIoish/ALTONG_client/blob/803b55562b6dc7f4b3c97e53694ec1bb9c90502f/src/Altong.Client/Services/Notifications/KakaoNotificationInterceptor.cs#L474),
[OCR 문자열 채택](https://github.com/AIoish/ALTONG_client/blob/803b55562b6dc7f4b3c97e53694ec1bb9c90502f/src/Altong.Client/Services/Notifications/LiveKakaoWindowOperator.cs#L841).

## 확인한 학습·실사용 입력 불일치

최신 앱 ToModelContext는 프로그램·제목이 현재 창과 모두 같은 항목을 recent_windows에서 제외한다.
그 뒤 남은 최대3개 창의 프로그램 이름으로 recent_processes를 재생성한다.
하지만 관련도 학습10,173행 중4,910행은 현재 창을 최근 창에도 넣는다.
이4,910행 중4,162행은 관련도4/5다. 높은 관련도 전체4,970행의 약83.7%가 이 중복을 가진다.
이는 중복이 정답 추측의 단서가 될 가능성을 보여주지만 내부 학습 원인의 증명은 아니다.

**v12에도 중복4,910행이 남아 있다.** v12에서 제거한 것은 recent_processes 필드뿐이다.
앞선 공통화에서는 필드 계약을 맞췄지만, 최근 창 목록의 값 구성까지 일치하는지는 놓쳤다.

원래 검증74행을 앱과 같은 규칙으로 바꿔 기존 모델에 넣으면 관련도가3행에서 달라진다.

| ID | 기존 관련도 | 앱 규칙 적용 후 | 실제 입력 차이 |
| --- | --- | --- | --- |
| v3_44_related | 5 | 1 | 최근 창 제목이 없는 옛 행의 프로세스 목록이 앱 규칙에서는 빈 목록이 됨 |
| valpool05_03_reference_matching | 3 | 1 | 최근 창에서 현재 창 중복을 빼고 프로그램 목록도 재생성 |
| valpool05_04_reference_matching | 2 | 1 | 위와 같음 |

5→1 사례를 현재 창 중복만의 영향으로 해석하면 안 된다. 옛 프로세스 목록이 사라지는 별도 차이다.
통과·차단 정답은68/74로 그대로였다. 긴급도가 최종 통과를 보완하므로 전체 정답률만으로는 관련도 변화를 놓친다.
같은 과거 사례를 이용한 진단이며 새로운 독립 평가나 사용자 사례 재현은 아니다.

## 모델의 학습 범위

관련도 학습10,173행에는 본문2글자 이하가0행이고 최단 본문도26자다.
단독 자모·짧은 대답은 길이만 보더라도 학습 예시 범위에서 크게 벗어나 있다.
학습 공백은 확인했지만 이것만으로 실제 두 알림 차이의 원인을 확정하지 않는다.

짧은 본문도 제목·창 단서에 따라 높은 점수를 낼 수 있었다.
제목에도 의미가 있으므로 한 글자면 항상1점이 정답이라고 단정하지는 않는다.
모델 입력에 이전 대화 본문·답장 대상은 없고, 불확실할 때 보류하는 출력도 없다.

## 통제 실험과 C# 비교

앱 manifest와 같은 모델 해시:
`315bb0778e47a84bfa2e5b705edb0daeb5164fa3057546109e425f749bc61a1a`

- 합성18조건: 조건 안에서 발신자·방·시각·맥락을 고정하고 본문만 ㅇ/ㅁ으로 변경. 점수 차이0건.
- 과거 검증74조건: 같은 방식으로 본문만 변경. 점수 차이0건.
- 같은 입력36쌍 반복: logits를 포함한 결과 일치.
- 별도 시각 변경8건: 해당 조건에서 클래스 차이 없음. 모든 입력에서 시각 영향이 없다는 뜻은 아님.
- 원격 앱의 Input.cs, KlueTokenizer.cs, KlueFilter.cs는 로컬 소스와 개행 정규화 후 동일.
- 기존 C# 검증 실행기에서 단문12건의 JSON·토큰·긴급도/관련도가 Python과 일치. 별도 단독 문자 등12건의 토큰도 일치.
- C#의 실패 시 통과·취소·긴 입력 거부·큐 제한 검사도 성공.

이 범위에서 C# 문자 처리·헤드 선택·점수 덮어쓰기 문제는 재현되지 않았다.
사용자의 실기기 OCR 결과, 실제 모델 입력, 실행 바이너리와 모델 파일 해시는 직접 확인하지 못했다.

## 다음 조치

1. 현재 창을 최근 창에서 제외하는 앱 정의에 맞춰 학습·검증·평가의 값 구성도 통일한다.
   새 데이터 버전으로 만들고 원본을 보존한다. 없어진 네 번째 과거 창을 추측해서 채우지 않는다.
2. 실제 점수가 갈린 두 건의 알림ID·모델 해시·OCR 후 전체 알림·안정화 맥락·최종JSON·토큰·raw logits·최종 점수를 같은ID로 기록한다.
   현재 DB는 알림 행에 평가 당시 창 스냅샷을 저장하지 않아 기존 점수 로그만으로 전체 입력을 복원할 수 없다.
3. 짧은 대답·무의미한 문자·약어·짧지만 구체적인 업무 메시지를 구분한 학습/별도 평가를 마련한다.
   문자만 보고 일괄적으로 점수를 주는 규칙을 덧붙이지 않는다.
4. 입력을 맞춘 뒤 공동 모델을 학습하고, 관련도 단독의 누락·불필요한 통과와 전체 정책을 따로 평가한다.

이번에는 조사만 했으며 앱·학습 데이터·기존 모델을 수정하거나 재학습하지 않았다.

## 로컬 근거

- outputs/runs/2026-11-04-relevance-short-input-diagnostic-01/: 합성 조건·시각 대조·전체 입력·logits
- outputs/runs/2026-11-04-relevance-short-input-diagnostic-02/: 검증 맥락의 단문 치환·C#fixture·학습 분포
- outputs/runs/2026-11-04-relevance-app-context-diagnostic-01/: 앱 규칙 전후 입력·점수·비교
- outputs/_maintenance/relevance-investigation-2026-11-04/client-src/: 조사 시점의 고정 커밋 관련 소스

위 경로는 filtering_training/ 기준이다. 모두 진단 자료이며 신규 학습셋이나 공식 독립 테스트셋이 아니다.
