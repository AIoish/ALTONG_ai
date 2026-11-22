# Gemma Q4_0 프롬프트 캐시와 워밍업

ggml-org Gemma 4 E2B Q4_0로 같은 모의 알림 5묶음을 요약했습니다. 실행마다 새 llama-server 프로세스를 시작했습니다. 한 실행은 캐시를 껐고, 다른 실행은 첫 요청 전에 공통 시스템 프롬프트를 계산하고 같은 서버에서 캐시를 재사용했습니다. 두 실행 모두 동일한 시스템 프롬프트, 알림 순서, JSON 스키마를 사용했습니다.

## 측정 결과

| 알림 묶음 | 캐시·워밍업 끔 | 캐시·워밍업 켬 | 새로 처리한 입력 토큰 |
| --- | ---: | ---: | ---: |
| 회의 장소와 발표자 변경 | 26.22초 | 9.60초 | 1,618 → 420 |
| 배송 보류와 부분 환불 | 38.23초 | 10.01초 | 1,605 → 376 |
| 로그인 장애 복구와 후속 점검 | 36.99초 | 11.00초 | 1,600 → 371 |
| 제출 마감과 재제출 예외 | 26.90초 | 11.01초 | 1,549 → 320 |
| 일상 단톡의 농담과 잡담 | 25.95초 | 9.52초 | 1,583 → 354 |
| **합계** | **154.29초** | **51.14초** | |

사전 워밍업은 17.08초였습니다. 이를 포함한 캐시 사용 실행은 총 **68.22초**로 약 **56%** 짧았습니다. 첫 번째 요청은 입력 1,618토큰 중 1,198토큰을 재사용했습니다. 두 실행에서 각각 5개 응답이 알통 브리핑 형식 검증을 통과했고, 원본 응답은 5개 중 3개가 같았습니다.

위 시간은 모델 로딩, 프롬프트 렌더링, 토큰화, 앱 처리 및 화면 표시를 제외한 로컬 추론 요청 시간입니다. 캐시와 워밍업을 동시에 켜고 끈 두 조건을 비교했습니다. 클라이언트에서 첫 브리핑의 체감 시간이 줄어드는지는 별도 통합 확인이 필요합니다.

## 재현

- llama.cpp 빌드 b11541의 `llama-server`와 [ggml-org Q4_0 GGUF 모델](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/blob/main/gemma-4-E2B-it-Q4_0.gguf)을 준비합니다.
- 두 실행은 CPU 6스레드, 슬롯 1개, 컨텍스트 4,096토큰을 사용합니다.
- 저장소 루트에서 실행합니다. 아래 두 변수에는 자신의 실행 파일과 모델 파일 경로를 지정합니다.

```powershell
$llamaServer = "C:/path/to/llama-server.exe"
$gemmaModel = "C:/path/to/gemma-4-E2B-it-Q4_0.gguf"
$cases = "briefing_training/experiments/gemma-cache-warmup/cases.jsonl"

python -m briefing_training.run_gemma_gguf --llama-server $llamaServer --gguf-path $gemmaModel --cases $cases --output outputs/gemma-cache/off.jsonl --no-cache
python -m briefing_training.run_gemma_gguf --llama-server $llamaServer --gguf-path $gemmaModel --cases $cases --output outputs/gemma-cache/on.jsonl --warm-system-prompt
```

JSONL에는 사례별 응답, 형식 검증 결과, 소요 시간, 서버 타이밍이 기록됩니다. 첫 응답의 `timings.cache_n`은 캐시에서 재사용한 토큰 수이고 `timings.prompt_n`은 새로 처리한 토큰 수입니다. 워밍업 시간은 첫 응답의 `warmup_seconds`에서 확인할 수 있습니다. 다시 측정할 때는 새 출력 경로를 지정합니다.
