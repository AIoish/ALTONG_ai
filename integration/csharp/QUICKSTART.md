<!-- 용도: 앱 개발자가 전달 ZIP을 받아 실제 차단·통과 기능을 연결하는 간단한 안내.
생성일: 2026-10-10 -->
# ALTONG 필터 모델 연결 가이드

**전달 파일:** `ALTONG_filtering_csharp_2026-10-10_actual.zip`

KLUE base가 긴급도·관련도를 각각 1~5점으로 판단한다. **집중 모드에서 긴급도 ≥4 또는 관련도 ≥4면 통과, 나머지는 차단**한다.
기본값은 실제 필터링인 `shadow: false`다. 개발 환경은 Windows x64 + .NET 9 SDK이며 Python 설치는 필요 없다.

## 1. ZIP을 앱 저장소 루트에 풀기

`ALTONG_client`의 `src`, `docs`가 있는 위치에 아래 폴더가 오도록 배치한다.

```text
ALTONG_client/
├─ src/
├─ integration/csharp/    ← C# 라이브러리·앱 어댑터
└─ models/filtering/klue-base-safety-v1/
   ├─ model-fp32.onnx
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   └─ manifest.json
```

모델·토크나이저·manifest는 함께 배포한다. 별도 토큰화 구현이나 모델 변환은 필요 없다.

## 2. 앱 프로젝트에 모듈 연결

`src/Altong.Client/Altong.Client.csproj`의 `<Project>` 안에 추가한다.

```xml
<ItemGroup>
  <ProjectReference Include="..\..\integration\csharp\Altong.Filtering\Altong.Filtering.csproj" />
  <Compile Include="..\..\integration\csharp\ClientAdapter\KlueFilterEngine.cs"
           Link="Services\AI\Filtering\KlueFilterEngine.cs" />
  <Content Include="..\..\models\filtering\klue-base-safety-v1\**\*">
    <Link>models\filtering\klue-base-safety-v1\%(RecursiveDir)%(Filename)%(Extension)</Link>
    <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    <CopyToPublishDirectory>PreserveNewest</CopyToPublishDirectory>
  </Content>
</ItemGroup>
```

ONNX Runtime 패키지는 라이브러리 참조를 통해 설치된다.

## 3. 기존 규칙 엔진을 KLUE 엔진으로 교체

`App.xaml.cs`의 `App` 클래스에 필드 하나를 추가한다.

```csharp
private KlueFilterEngine? _klueFilterEngine;
```

현재 `var filterEngine = new RuleBasedFilterEngine();` 부분을 아래로 교체한다.
`using Altong.Client.Services.Notifications;`와 `using System.IO;`를 사용한다.

```csharp
var modelDir = Path.Combine(AppContext.BaseDirectory,
    "models", "filtering", "klue-base-safety-v1");

_klueFilterEngine = new KlueFilterEngine(
    Path.Combine(modelDir, "model-fp32.onnx"),
    Path.Combine(modelDir, "tokenizer.json"),
    shadow: false);

IFilterEngine filterEngine = _klueFilterEngine;
```

기존 `NotificationPipelineCoordinator`에 `filterEngine`을 전달하는 부분은 그대로 사용한다.
`OnExit`에서는 기존 `NotificationPipeline.Dispose()` 처리 다음에 추가한다.

```csharp
_klueFilterEngine?.Dispose();
```

알림 수집·DB 저장·독 UI는 기존 파이프라인을 사용한다. 브리핑 기능은 변경할 필요 없다.

## 4. 빌드하고 집중 모드에서 사용

앱 저장소 루트에서 빌드한다.

```powershell
dotnet build src/Altong.Client/Altong.Client.csproj -c Release -r win-x64
```

빌드된 앱을 실행하고 **집중 모드를 켠다.** 통과한 알림은 기존 독에 표시되고 차단한 알림은 기존 집중 세션 DB에 저장된다.
집중 모드가 꺼져 있으면 앱의 기존 동작대로 차단하지 않는다.
배포할 때는 빌드/게시 결과의 모델 폴더와 네이티브 런타임 파일도 함께 포함한다.

## 연결할 때 알아둘 사항

- 모델이 정상 판단하면 실제 차단·통과를 적용한다. 로딩 중·파일 누락·추론 오류·시간 초과는 알림을 통과시킨다.
  이때 점수 **0은 미평가**이며 모델의 1~5점 판단이 아니다.
- 현재 앱에 없는 최근 창 제목 이력은 빈 배열로 전달한다. 현재 창과 최근 프로세스는 기존 값으로 전달한다.
- Windows 알림이 들어오지 않으면 WinRT 알림 접근 권한과 앱의 sparse package 등록을 확인한다.
  카카오톡 알림은 기존 전용 인터셉터를 사용한다.
- Python/C# 입력·토큰·점수 188건 일치는 완료했다. 모델의 기존 긴급 알림 누락 2건은 남아 있다.

세부 명세와 문제 해결은 ZIP 안의 `integration/csharp/README.md`를 참고한다.
