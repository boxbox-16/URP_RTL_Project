# CPU 후처리와 RTL 후처리 비교 실행 가이드

## 1. 비교 목적

이 실험은 같은 `test_video.mp4`를 사용해 다음 두 파이프라인을 비교합니다.

| 구분 | 전처리 | 추론 | 후처리 |
|---|---|---|---|
| SW 기준 | CPU/OpenCV | DPU | RTL-equivalent CPU golden model |
| RTL | CPU/OpenCV | DPU | FPGA RTL `postproc_top_0` |

확인할 내용은 다음과 같습니다.

1. 두 경로의 전처리 시간이 비슷한가?
2. 두 경로의 DPU 추론 시간이 비슷한가?
3. RTL 후처리가 CPU 후처리보다 빨라졌는가?
4. 두 후처리가 계산한 조향값이 비슷한가?

비교 실행 파일은 프로젝트 루트의 `compare_postprocessing.py`입니다.

## 2. 공정한 비교를 위해 적용된 조건

- 동일한 영상 프레임을 두 경로에 입력합니다.
- 두 경로가 동일한 전처리 구현과 DPU 모델을 사용합니다.
- CPU와 RTL 경로는 전처리와 DPU 추론을 각각 독립적으로 수행하고 시간을 측정합니다.
- 먼저 실행되는 경로에 유리한 영향을 줄이기 위해 실행 순서를 프레임마다 바꿉니다.
  - 짝수 프레임: `Baseline → RTL`
  - 홀수 프레임: `RTL → Baseline`
- 측정 전에 두 전처리 결과가 완전히 같은지 확인합니다.
- 동일 입력에 대한 두 DPU int8 출력이 완전히 같은지도 확인합니다.
- `auto_drive`와 `auto_drive_RTL`의 설정, 모델, `postprocessing.py`, 실행 환경 파일이 같은지 SHA-256으로 검사합니다.
- 영상 디코딩, 제어기 계산, 화면 출력, 결과 저장 시간은 단계별 성능 측정에서 제외합니다.
- 모터 액추에이터는 작동시키지 않습니다.

SW 기준 후처리는 현재 `postproc_top`과 같은 임계값 포함 조건, 5×5 morphology 경계 처리, 256×256 좌표계, 기준 행 및 valid 조건을 재현합니다. 두 경로 모두 RTL 비트스트림이 올라간 동일한 DPU 환경에서 실행하므로 후처리 구현만 비교할 수 있습니다.

공통 후처리 사양은 두 폴더의 동일한 `postprocessing.py`에 정의되어 있습니다. 두 파일의 해시가 다르면 비교를 시작하지 않습니다.

## 3. RTL 비트스트림 갱신

RTL 수정 사항을 반영한 뒤 다음 두 파일을 함께 갱신합니다.

```text
auto_drive_RTL/configs/dpu/dpu.bit
auto_drive_RTL/configs/dpu/dpu.hwh
```

비트스트림과 HWH가 서로 다른 빌드에서 만들어진 파일이면 IP 주소를 잘못 찾을 수 있으므로 반드시 같은 Vivado 빌드 결과를 사용합니다.

비교 스크립트는 HWH 정보에서 `postproc_top_0` 주소를 자동으로 찾습니다. 자동 탐지가 불가능하면 기존 노트북의 기본 주소인 `0x80010000`을 사용합니다.

## 4. 입력 영상 확인

기본 영상 위치는 프로젝트 루트의 다음 파일입니다.

```text
/home/xilinx/jupyter_notebooks/URP_HW/test_video.mp4
```

현재 영상 정보는 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| 해상도 | 640 × 480 |
| FPS | 30 |
| 프레임 수 | 2,949 |
| 재생 시간 | 약 98.3초 |

## 5. 실행 방법

### 5.1 프로젝트 디렉터리로 이동

```bash
cd /home/xilinx/jupyter_notebooks/URP_HW
```

### 5.2 기본 실행

프로젝트 루트의 `test_video.mp4` 전체를 비교하려면 인자 없이 실행합니다.

```bash
python3 compare_postprocessing.py
```

다른 영상을 비교하려면 영상 경로를 첫 번째 인자로 지정합니다.

```bash
python3 compare_postprocessing.py drive.mp4
```

결과는 별도 지정이 없으면 `comparison_results/<실행시각>/`에 자동 저장됩니다.

### 5.3 짧은 시험 실행

RTL 수정 후에는 먼저 30프레임만 실행해 DPU와 RTL IP가 정상 동작하는지 확인합니다.

```bash
python3 compare_postprocessing.py test_video.mp4 -n 30
```

결과 폴더를 직접 지정하려면 `-o`를 추가합니다.

```bash
python3 compare_postprocessing.py test_video.mp4 -n 30 -o comparison_results/rtl_v2_smoke
```

정상 실행되면 다음 메시지가 출력됩니다.

```text
공통 설정/xmodel 검증 완료
RTL postproc 주소: 0x........
공통 단계 값 검증: 전처리 출력 및 DPU int8 출력 완전 일치
```

동일 기능 검증에서는 `조향 MAE=0`, `lane_pixels 완전 일치=100%`,
`valid 일치=100%`가 모두 충족되어야 합니다. 하나라도 다르면 전체 영상 결과를
최종 가속 수치로 사용하지 말고 RTL 또는 golden model의 의미 차이를 먼저 수정합니다.

### 5.4 전체 영상 실행

시험 실행에 문제가 없으면 전체 영상을 처리합니다.

```bash
python3 compare_postprocessing.py
```

다른 영상 전체를 처리할 때는 해당 영상만 지정하면 됩니다.

```bash
python3 compare_postprocessing.py drive.mp4
```

`-n 0` 또는 `--frames 0`도 영상의 마지막 프레임까지 모두 처리한다는 뜻입니다.
출력 경로를 지정하려면 다음처럼 실행합니다.

```bash
python3 compare_postprocessing.py test_video.mp4 -o comparison_results/rtl_v2_full
```

자동 생성 경로의 예시는 다음과 같습니다.

```text
comparison_results/20260905T083000Z/
```

## 6. 주요 실행 옵션

| 인자/옵션 | 기본값 | 설명 |
|---|---:|---|
| `video` | `test_video.mp4` | 첫 번째 위치 인자로 지정하는 입력 영상 경로 |
| `-n`, `--frames` | `0` | 측정할 프레임 수. `0`이면 영상 끝까지 실행 |
| `-o`, `--output-dir` | 자동 생성 | CSV와 JSON을 저장할 디렉터리 |
| `--skip-frames` | `0` | 영상 앞부분에서 제외할 프레임 수 |
| `--warmup` | `10` | 통계에서 제외할 워밍업 횟수 |
| `--print-every` | `50` | 진행 상황을 출력할 프레임 간격 |
| `--steering-tolerance` | `0.0` | 두 조향 오차가 같다고 판단할 허용 범위 |
| `--timing-tolerance-pct` | `10.0` | 전처리·추론 시간이 유사하다고 판단할 차이 비율 |
| `--rtl-timeout-ms` | `50.0` | RTL 완료 신호 대기 제한 시간 |
| `--rtl-base` | 자동 탐지 | RTL IP 주소 수동 지정 |

기존 명령과의 호환성을 위해 `--video`와 `--frames`도 계속 지원하지만, 새 명령에서는 영상 위치 인자와 `-n` 사용을 권장합니다.

RTL IP 주소를 직접 지정해야 하는 경우에는 다음처럼 실행합니다.

```bash
python3 compare_postprocessing.py test_video.mp4 -n 30 --rtl-base 0x80010000
```

## 7. 저장되는 결과

실행이 끝나면 지정한 디렉터리에 두 파일이 만들어집니다.

```text
comparison_results/rtl_v2_full/
├── frames.csv
└── summary.json
```

### 7.1 `frames.csv`

각 영상 프레임의 상세 결과입니다. Excel, LibreOffice Calc, Pandas 등으로 분석할 수 있습니다.

주요 열은 다음과 같습니다.

| 열 | 의미 |
|---|---|
| `frame_id` | 영상 프레임 번호 |
| `execution_order` | 해당 프레임에서 두 경로가 실행된 순서 |
| `baseline_preprocess_ms` | Baseline 전처리 시간 |
| `rtl_preprocess_ms` | RTL 경로 전처리 시간 |
| `baseline_inference_ms` | Baseline DPU 추론 시간 |
| `rtl_inference_ms` | RTL 경로 DPU 추론 시간 |
| `baseline_postprocess_ms` | CPU 후처리 전체 시간 |
| `rtl_postprocess_ms` | 복사, RTL 실행, 결과 읽기를 포함한 RTL 후처리 전체 시간 |
| `baseline_pipeline_ms` | Baseline 전처리+추론+후처리 시간 |
| `rtl_pipeline_ms` | RTL 전처리+추론+후처리 시간 |
| `baseline_steering_error` | CPU 후처리가 계산한 정규화 조향 오차 |
| `rtl_steering_error` | RTL 후처리가 계산한 정규화 조향 오차 |
| `steering_error_abs_diff` | 두 조향 오차의 절댓값 차이 |
| `baseline_steering_cmd` | CPU 결과에 동일 PD 제어기를 적용한 조향 명령 |
| `rtl_steering_cmd` | RTL 결과에 동일 PD 제어기를 적용한 조향 명령 |
| `steering_cmd_abs_diff` | 두 조향 명령의 절댓값 차이 |
| `baseline_valid`, `rtl_valid` | 각 경로의 차선 검출 유효 판정 |
| `rtl_copy_ms` | DPU 출력을 RTL용 CMA 버퍼로 복사하고 flush한 시간 |
| `rtl_compute_ms` | RTL 시작부터 완료 신호까지 기다린 시간 |

### 7.2 `summary.json`

전체 프레임의 요약 통계와 실행 환경입니다.

다음 명령으로 보기 좋게 확인할 수 있습니다.

```bash
python3 -m json.tool comparison_results/rtl_v2_full/summary.json | less
```

`q`를 누르면 `less` 화면에서 나옵니다.

## 8. 결과 해석 방법

### 8.1 전처리와 추론 조건 확인

다음 항목이 `true`인지 확인합니다.

```text
criteria.preprocess_timing_similar
criteria.inference_timing_similar
```

기본 설정에서는 두 경로의 평균 시간 차이가 10% 이내이면 유사하다고 판단합니다.

### 8.2 RTL 후처리 가속 확인

핵심 가속비는 다음 항목입니다.

```text
timing_ms.postprocess_ms.speedup_baseline_over_rtl
```

계산식은 다음과 같습니다.

```text
CPU 후처리 평균 시간 ÷ RTL 후처리 평균 시간
```

| 값 | 의미 |
|---:|---|
| `1.0` | 속도가 같음 |
| `1.5` | RTL 후처리가 약 1.5배 빠름 |
| `2.0` | RTL 후처리가 약 2배 빠름 |
| 1보다 작음 | RTL 경로가 CPU보다 느림 |

다음 판정값도 함께 확인합니다.

```text
criteria.rtl_postprocess_faster
```

### 8.3 조향값 유사성 확인

이 실험에서 조향값은 실제 도(degree)가 아니라 `-1.0~+1.0` 범위의 정규화된 값입니다.

- `steering_error`: 후처리가 계산한 차선 중심 오차
- `steering_cmd`: `steering_error`에 동일한 PD+EMA 제어기를 적용한 명령

주요 지표는 다음과 같습니다.

| JSON 항목 | 의미 | 바람직한 방향 |
|---|---|---|
| `steering.error_mae` | 전체 프레임 평균 절대 오차 | 0에 가까울수록 좋음 |
| `steering.error_p95_abs` | 오차가 작은 순서로 95% 지점의 값 | 허용 오차 이하 |
| `steering.error_max_abs` | 가장 크게 차이 난 프레임 | 0에 가까울수록 좋음 |
| `steering.error_correlation` | 두 조향 변화의 상관계수 | 1에 가까울수록 좋음 |
| `steering.error_within_tolerance_ratio` | 허용 오차 안에 들어온 프레임 비율 | 1에 가까울수록 좋음 |
| `steering.direction_agreement_ratio` | 좌·우 조향 방향이 일치한 비율 | 1에 가까울수록 좋음 |
| `steering.valid_agreement_ratio` | 차선 유효 판정이 일치한 비율 | 1에 가까울수록 좋음 |
| `steering.command_mae` | 최종 조향 명령의 평균 절대 차이 | 0에 가까울수록 좋음 |

기본 조향 허용 오차는 `0.0`입니다. 동일한 정수 Q1.15 계산을 비교하므로 기본 판정은 완전 일치를 요구합니다.

```bash
python3 compare_postprocessing.py test_video.mp4 \
  --steering-tolerance 0.03 \
  --timing-tolerance-pct 10
```

### 8.4 전체 판정 확인

다음 항목은 모든 기본 조건을 만족했는지 나타냅니다.

```text
criteria.all_pass
```

`true`가 되려면 다음 조건이 모두 충족되어야 합니다.

1. 전처리 평균 시간 차이가 허용 범위 이내
2. DPU 추론 평균 시간 차이가 허용 범위 이내
3. RTL 후처리가 CPU 후처리보다 빠름
4. 조향 오차 MAE가 허용 범위 이내
5. 조향 오차 P95가 허용 범위 이내

## 9. 연구 결과에 기록하면 좋은 값

논문이나 보고서에는 최소한 다음 내용을 함께 기록하는 것이 좋습니다.

| 분류 | 기록 항목 |
|---|---|
| 입력 | 영상 SHA-256, 해상도, FPS, 프레임 수 |
| 실행 환경 | Python, OpenCV, Numpy 버전 |
| 공통 조건 | config와 xmodel SHA-256 |
| 하드웨어 | RTL bitstream SHA-256, RTL IP 주소 |
| 시간 | 각 단계 평균, 중앙값, P95, 표준편차 |
| 가속 | 후처리 가속비, 전체 파이프라인 가속비 |
| 정확도 | 조향 MAE, P95, 최대 오차, 상관계수, 방향 일치율 |
| 유효성 | Baseline/RTL valid 비율과 valid 일치율 |

이 값들은 `summary.json`의 `metadata`, `timing_ms`, `steering`, `criteria`에 자동으로 저장됩니다.

## 10. 실행 시 주의사항

- 결과 디렉터리는 기존 데이터를 덮어쓰지 않습니다. 같은 이름이 있으면 새 이름을 지정합니다.
- 워밍업 프레임은 측정 통계에서 제외됩니다.
- 실행 중 강제 종료하면 최종 CSV와 JSON이 만들어지지 않을 수 있으므로 전체 실행이 끝날 때까지 기다립니다.
- `postprocess_ms`는 실제 파이프라인 비용을 나타내며, RTL 경로에서는 CMA 복사와 MMIO 제어 비용까지 포함합니다.
- `rtl_compute_ms`만 사용하면 데이터 전달 비용을 제외하게 되므로 CPU 후처리와의 전체 성능 비교에는 `rtl_postprocess_ms`를 사용합니다.
- 조향 차이가 크면 먼저 `frames.csv`에서 `steering_error_abs_diff`가 큰 프레임과 `valid` 불일치 프레임을 확인합니다.
- 비트스트림을 수정할 때마다 새로운 결과 디렉터리를 사용해 이전 결과와 구분합니다.

### 현재 bitstream 검증 상태

2026-09-06의 RTL-equivalent 30프레임 smoke test에서는 SW 기준점보다 실제 RTL
기준점이 모든 프레임에서 정확히 64픽셀 오른쪽으로 계산됐습니다. 조향 MAE는
`0.5`, SW 후처리는 평균 약 `2.99 ms`, RTL 후처리는 약 `6.83 ms`였습니다.
따라서 현재 bitstream 결과는 최종 가속 수치로 사용하지 않습니다. RTL의
centerline BRAM read 정렬을 수정하고, 동일 기능 기준에서 RTL 실행 시간도
최적화한 뒤 다시 측정해야 합니다.

## 11. 권장 실행 명령

RTL 버전 이름이나 날짜를 결과 디렉터리에 포함하면 실험 결과를 구분하기 쉽습니다.

```bash
python3 compare_postprocessing.py test_video.mp4 \
  --warmup 10 \
  --print-every 100 \
  --steering-tolerance 0.0 \
  --timing-tolerance-pct 10 \
  -o comparison_results/rtl_버전명_날짜
```

예시:

```bash
python3 compare_postprocessing.py test_video.mp4 \
  --warmup 10 \
  --print-every 100 \
  -o comparison_results/rtl_v3_20260905
```
