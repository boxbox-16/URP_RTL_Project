# 소프트웨어 후처리와 RTL 후처리 비교 가이드

## 1. 목적

`compare_postprocessing.py`는 같은 주행 영상에 대해 다음 두 파이프라인을 순차적으로 측정합니다.

| 구분 | 비트스트림 | 전처리 | DPU 추론 | 후처리 |
|---|---|---|---|---|
| SW 기준 | `auto_drive/configs/default.yaml`의 `actuator.overlay_path` | CPU | 해당 bitstream의 DPU | `compare_postprocessing.py` 내부 CPU golden model |
| RTL | `auto_drive_RTL/configs/default.yaml`의 `actuator.overlay_path` | CPU | 해당 bitstream의 DPU | `postproc_top_0` RTL |

측정 목적은 다음과 같습니다.

- 전처리와 DPU 추론 조건이 두 비트스트림에서 동일한지 확인
- RTL 후처리의 전체 지연시간과 CPU 후처리 지연시간 비교
- 차선 픽셀 수, 기준점, 조향값, valid 판정의 결과 정합성 확인

실주행 주피터 노트북에는 비교 기능이 없습니다. 성능·정합성 비교는 이 스크립트에서만 수행합니다.

## 2. 비교 순서와 자원 관리

실행 순서는 프레임별 교대 방식이 아니라 완전한 2단계 직렬 방식입니다.

1. `auto_drive` 설정을 읽고 SW 비트스트림을 로드합니다.
2. SW 파이프라인을 영상 처음부터 끝까지 측정합니다.
3. SW의 영상 캡처, DPU runner, overlay 참조를 해제하고 garbage collection을 수행합니다.
4. `auto_drive_RTL` 설정을 읽고 RTL 비트스트림을 로드합니다.
5. 같은 영상을 처음부터 다시 읽어 RTL 파이프라인을 측정합니다.
6. 두 단계의 같은 프레임을 사후 대조해 결과를 저장합니다.

두 파이프라인은 동시에 실행되지 않습니다. 따라서 `frames.csv`의 `execution_order`는 모든 행에서 다음 값을 가집니다.

```text
baseline_overlay_pass->release->rtl_overlay_pass
```

이 방식은 실제로 서로 다른 두 bitstream을 각각 사용한다는 장점이 있습니다. 다만 SW 측정이 항상 먼저 실행되므로 온도·클럭 상태 편향을 줄이려면 실험을 여러 번 반복하거나 실행 순서를 바꾼 추가 측정이 필요합니다.

## 3. 공통 조건 검증

비트스트림을 제외한 다음 파일은 두 폴더에서 SHA-256을 비교합니다.

```text
configs/default.yaml
configs/camera.yaml
configs/model.yaml
configs/control.yaml
models/lane_segmentation.xmodel
setup.sh
```

비교 시작 전에 파일 내용이 다르면 중단합니다. 두 bitstream 자체는 서로 달라도 되며, 결과의 `metadata`에 각각의 경로와 SHA-256이 기록됩니다.

또한 모든 측정 프레임에 대해 다음 배열의 SHA-256을 대조합니다.

- 전처리 출력 배열
- DPU int8 출력 배열

따라서 후처리 성능 비교 전에 두 비트스트림이 같은 영상 프레임에 대해 같은 공통 단계 결과를 내는지 확인할 수 있습니다.

## 4. CPU golden model과 RTL 후처리 기준

CPU golden model은 `compare_postprocessing.py`의 `RTLEquivalentPostProcessor` 클래스입니다. 별도의 `postprocessing.py` 파일이나 주피터 외부 모듈을 사용하지 않습니다.

현재 기준은 다음과 같습니다.

```text
DPU int8 출력 256×256
    ↓
값 >= 0을 차선 픽셀로 판정
    ↓
5×5 erode → dilate → dilate → erode
    ↓
기준 행 y=179에 가장 가까운 유효 행 선택
    ↓
선택 행의 x 평균을 정수 나눗셈
    ↓
raw_error_q15 = (reference_x - 128) << 8
steering_error = raw_error_q15 / 32768
```

valid는 차선 픽셀이 존재하고 `min_lane_pixels` 이상일 때 true입니다. RTL 후처리는 DPU 출력 int8 버퍼를 DDR/CMA로 복사한 뒤 `postproc_top_0`의 결과 레지스터를 읽습니다. `postprocess_ms`에는 RTL용 버퍼 복사, MMIO 제어, RTL 완료 대기 시간이 모두 포함됩니다.

## 5. 비트스트림과 RTL 주소

각 설정 파일의 경로를 기준으로 bitstream을 찾습니다.

```yaml
# auto_drive/configs/default.yaml
actuator:
  overlay_path: configs/dpu/dpu.bit

# auto_drive_RTL/configs/default.yaml
actuator:
  overlay_path: configs/dpu/dpu.bit
```

따라서 실제로 읽히는 파일은 다음과 같습니다.

```text
auto_drive/configs/dpu/dpu.bit
auto_drive_RTL/configs/dpu/dpu.bit
```

RTL IP 주소는 overlay의 `ip_dict`에서 `postproc_top_0`를 자동 탐지합니다. 자동 탐지가 어려운 경우에만 다음 옵션으로 직접 지정합니다.

```bash
--rtl-base 0x80010000
```

비트스트림과 HWH는 반드시 같은 Vivado 빌드에서 생성된 쌍을 사용해야 합니다.

## 6. 실행 방법

프로젝트 루트에서 실행합니다.

```bash
cd /home/xilinx/jupyter_notebooks/URP_HW
```

짧은 smoke test:

```bash
python3 compare_postprocessing.py test_video.mp4 \
  -n 30 \
  --warmup 10 \
  --print-every 10 \
  -o comparison_results/smoke
```

전체 영상:

```bash
python3 compare_postprocessing.py test_video.mp4 \
  --warmup 10 \
  --print-every 250 \
  -o comparison_results/full
```

기본 영상은 `test_video.mp4`이며, 영상을 생략하면 프로젝트 루트의 기본 영상을 사용합니다.

```bash
python3 compare_postprocessing.py
python3 compare_postprocessing.py drive.mp4
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---:|---|
| `video` | `test_video.mp4` | 입력 영상 위치 인자 |
| `-n`, `--frames` | `0` | 측정 프레임 수. 0이면 끝까지 |
| `-o`, `--output-dir` | 자동 생성 | 결과 저장 폴더 |
| `--skip-frames` | `0` | 앞에서 건너뛸 프레임 수 |
| `--warmup` | `10` | 각 bitstream에서 통계 제외할 워밍업 횟수 |
| `--print-every` | `50` | 진행 출력 간격 |
| `--rtl-timeout-ms` | `50.0` | RTL 완료 대기 제한 |
| `--rtl-base` | 자동 탐지 | RTL IP 주소 수동 지정 |
| `--steering-tolerance` | `0.0` | 조향 오차 허용 범위 |
| `--timing-tolerance-pct` | `10.0` | 전처리/DPU 평균 시간 유사성 기준 |

## 7. 결과 파일

실행이 끝나면 다음 두 파일이 생성됩니다.

```text
comparison_results/<실행시각>/
├── frames.csv
└── summary.json
```

`frames.csv`의 주요 열은 다음과 같습니다.

| 열 | 의미 |
|---|---|
| `frame_id` | 원본 영상 프레임 번호 |
| `execution_order` | 두 overlay pass의 실행 순서 |
| `baseline_preprocess_ms`, `rtl_preprocess_ms` | 전처리 시간 |
| `baseline_inference_ms`, `rtl_inference_ms` | DPU 추론 시간 |
| `baseline_postprocess_ms`, `rtl_postprocess_ms` | 후처리 전체 시간 |
| `baseline_pipeline_ms`, `rtl_pipeline_ms` | 전처리+추론+후처리 시간 |
| `baseline_steering_error`, `rtl_steering_error` | 정규화 조향 오차 |
| `baseline_steering_cmd`, `rtl_steering_cmd` | 동일 PD 제어기 적용 후 조향 명령 |
| `baseline_lane_pixels`, `rtl_lane_pixels` | 차선 픽셀 수 |
| `baseline_reference_x`, `rtl_reference_x` | 기준점 x 좌표 |
| `baseline_raw_error_q15`, `rtl_raw_error_q15` | 정수 Q1.15 조향값 |
| `steering_error_abs_diff` | 조향 오차 절댓값 차이 |
| `steering_cmd_abs_diff` | 조향 명령 절댓값 차이 |
| `rtl_copy_ms` | RTL 입력 버퍼 복사/flush 시간 |
| `rtl_compute_ms` | RTL 시작부터 완료까지의 시간 |

`summary.json`에는 다음 정보가 저장됩니다.

- SW/RTL bitstream 경로와 SHA-256
- RTL IP 주소 및 탐지 방식
- 모든 프레임의 전처리/DPU 출력 해시 일치 여부
- 단계별 평균, 중앙값, P95, 표준편차, 최소/최대 시간
- 조향 MAE, P95, 최대 오차, 상관계수, 방향 일치율
- lane pixel 및 valid 일치율
- 전체 판정 결과

## 8. 결과 해석

후처리 가속비는 다음과 같이 계산합니다.

```text
SW 후처리 평균 시간 / RTL 후처리 평균 시간
```

예를 들어 `3.25x`이면 RTL 후처리가 SW 후처리보다 약 3.25배 빠르다는 의미입니다. 전체 파이프라인 가속비는 `pipeline_ms` 평균으로 별도로 계산합니다.

다음 판정 항목을 확인합니다.

```text
criteria.preprocess_timing_similar
criteria.inference_timing_similar
criteria.rtl_postprocess_faster
criteria.steering_mae_within_tolerance
criteria.steering_p95_within_tolerance
criteria.lane_pixels_exact
criteria.valid_exact
criteria.all_pass
```

기본 조향 허용 오차는 `0.0`이므로, 현재 RTL과 CPU golden model이 같은 Q1.15 결과를 내는지 엄격하게 확인합니다.

## 9. 현재 전체 측정 예시

2026-09-07에 서로 다른 두 bitstream으로 2,949프레임을 측정한 결과입니다.

| 단계 | SW | RTL | 결과 |
|---|---:|---:|---:|
| 전처리 | 5.780 ms | 5.758 ms | 평균 차이 0.37% |
| DPU 추론 | 14.792 ms | 14.751 ms | 평균 차이 0.28% |
| 후처리 | 2.979 ms | 0.917 ms | 3.248배 |
| 합계 | 23.601 ms | 21.474 ms | 1.099배 |

정합성 결과:

- 전처리 출력 해시: 2,949/2,949 일치
- DPU int8 출력 해시: 2,949/2,949 일치
- 조향 MAE/P95/max: 모두 0
- lane pixel 완전 일치: 100%
- valid 일치: 100%

결과 파일:

```text
comparison_results/sequential_overlay_full_2949/summary.json
comparison_results/sequential_overlay_full_2949/frames.csv
```

## 10. 논문·보고서 작성 시 주의사항

- `3.248배`는 후처리 단독 가속비입니다.
- 전체 파이프라인 `1.099배`는 영상 디코딩, 제어기, 화면 출력, 파일 저장을 제외한 전처리+추론+후처리 기준입니다.
- 두 bitstream을 순차 실행하므로 반복 측정과 실행 순서 역전 실험을 추가하는 것이 좋습니다.
- 측정에 사용한 영상 SHA-256, bitstream SHA-256, xmodel, 클럭, 보드, 라이브러리 버전을 함께 기록해야 합니다.
- RTL이 전체 마스크를 외부로 반환하지 않는 경우, “모든 마스크 픽셀이 일치했다”고 쓰지 말고 조향값, 기준점, lane pixel 수, valid가 일치했다고 기술합니다.
- 실제 카메라 주행은 입력 프레임이 동일하지 않으므로 성능 비교는 녹화 영상으로 수행하고, 실주행은 기능 검증 사례로 구분합니다.
