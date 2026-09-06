# 자율주행 파이프라인 정리

## 프로젝트 개요

Ultra96V2 (PYNQ) + DPU 기반 차선 추종 자율주행.  
모든 실행 코드는 `autonomous_driving.ipynb` 단일 파일에 인라인으로 정의되어 있다.

---

## 파일 구조

```
autonomous_driving_baseline/
├── autonomous_driving.ipynb   ← 전체 파이프라인 (유일한 실행 파일)
├── setup.sh                   ← 스레드 수 환경변수 (OMP_NUM_THREADS=2 등)
├── configs/
│   ├── default.yaml           ← 실행 모드, 액추에이터 설정
│   ├── camera.yaml            ← 해상도, FPS, ROI
│   ├── model.yaml             ← xmodel 경로, 입력 크기, 정규화
│   ├── control.yaml           ← kp, deadband, 속도 범위
│   └── dpu/
│       ├── dpu.bit            ← FPGA 비트스트림 (DPU + 6채널 PWM)
│       ├── dpu.hwh            ← PYNQ 하드웨어 설명 (MMIO 주소 맵)
│       └── dpu.xclbin         ← Vitis AI 런타임 바이너리
└── models/
    └── lane_segmentation.xmodel  ← 차선 세그멘테이션 DPU 모델
```

---

## 파이프라인 흐름

```
카메라 프레임 (640×480 BGR)
      │
      ▼
 [1] 전처리 (Preprocessing)
      │  ROI 크롭 → 256×256 리사이즈 → BGR→RGB → float32 정규화 (/255)
      ▼
 [2] DPU 추론 (Inference)
      │  float32 → int8 양자화 (×64) → DPU 실행 → int8 → float32 역양자화 (×0.25)
      │  출력: (1, 256, 256, 1) float32 logits
      ▼
 [3] 후처리 (Postprocessing)
      │  logits → binary 마스크 (threshold=0.0)
      │  → Morphology OPEN+CLOSE (노이즈 제거)
      │  → 256×256 → ROI 크기 복원
      │  → 행별 차선 중심 x좌표 추출 (centerline)
      │  → 기준점 선택 (reference_row_ratio=0.7 높이)
      │  → steering_error = (ref_x - width/2) / (width/2)   ← [-1.0, +1.0]
      ▼
 [4] P 제어기 (PController)
      │  steering_cmd = kp × steering_error   (kp=0.4)
      │  speed_cmd    = base_speed × max(0.4, 1 - |steering_cmd|)  (코너 감속)
      ▼
 [5] 액추에이터 (PynqMMIOActuator)
      │  steering_effort 계산 → PWM duty 계산 → MMIO 레지스터 쓰기
      ▼
 FPGA PWM → 모터 드라이버 → 12V DC 모터
```

---

## 각 단계 상세

### [1] 전처리

| 단계 | 입력 → 출력 | 비고 |
|------|------------|------|
| ROI 크롭 | 640×480 → 640×480 | roi_top_ratio=0.0 (크롭 없음) |
| 리사이즈 | 640×480 → 256×256 | INTER_LINEAR |
| 색상 변환 | BGR → RGB | 모델 학습 순서 맞춤 |
| 정규화 | uint8 → float32 | ÷255, mean=0, std=1 |

### [2] DPU 추론

- 모델: `lane_segmentation.xmodel` (차선 이진 세그멘테이션)
- 입력: `(1, 256, 256, 3)` float32
- 출력: `(1, 256, 256, 1)` float32 (fix_point=2 → ×0.25)
- 양자화: fix_point=6 → ×64 후 int8 클리핑

### [3] 후처리

```
logits > 0.0  →  binary mask (0 or 255)
    ↓
Morphology OPEN (5×5)  →  작은 노이즈 제거
    ↓
Morphology CLOSE (5×5)  →  구멍 메우기
    ↓
connected components 필터  →  min_area=80 미만 blob 제거
    ↓
256×256 → ROI 크기(480×640) 복원  →  INTER_NEAREST
    ↓
행별 중심 x 추출 (5픽셀 간격, 행당 최소 3픽셀)
    ↓
기준점: y = height × 0.7 에 가장 가까운 중심선 포인트
    ↓
steering_error = (ref_x - 320) / 320
```

**유효(valid) 판정 조건** (모두 만족해야 true):
- 기준점이 존재
- 차선 픽셀 수 ≥ 20
- 중심선 포인트 수 ≥ 5

### [4] P 제어기

```python
steering_cmd = clamp(kp × steering_error, -1.0, +1.0)
speed_cmd    = clamp(base_speed × max(0.4, 1 - |steering_cmd|), min_speed, max_speed)
```

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| kp | 0.4 | 조향 비례 게인 |
| deadband | 0.02 | 이 미만 오차는 0으로 처리 |
| base_speed | 0.7 | 직진 기준 속도 명령 |
| min_speed | 0.10 | 속도 하한 |
| max_speed | 1.0 | 속도 상한 |

### [5] 액추에이터 (PWM MMIO 제어)

**FPGA PWM IP 레지스터 맵 (채널당):**

| 오프셋 | 이름 | 역할 |
|--------|------|------|
| 0x00 | REG_PERIOD | PWM 주기 클럭 수 (600600 = 500Hz @ 300MHz) |
| 0x04 | REG_DUTY | PWM on 구간 클럭 수 (0 ~ PERIOD) |
| 0x08 | REG_VALID | 출력 활성화 (1=ON, 0=OFF) |

**MMIO 주소 → 물리 핀 매핑 (dpu.bit 기준):**

| 주소 | PWM 핀 | 채널 역할 |
|------|--------|---------|
| 0xA0000000 | PWM_out_0 | rear_right_fwd |
| 0xA0010000 | PWM_out_4 | **steering_right** |
| 0xA0020000 | PWM_out_3 | rear_left_fwd |
| 0xA0030000 | PWM_out_2 | rear_left_bwd |
| 0xA0040000 | PWM_out_1 | rear_right_bwd |
| 0xA0050000 | PWM_out_5 | **steering_left** |

> HW_setting_test.bit는 주소 배치가 다름: 0xA0040000 = steering_right

**조향 duty 계산:**

```
조향 중:
  effective = min_duty + (1 - min_duty) × |effort|
  duty = period × steering_duty_percent × effective

중앙 유지 (|effort| ≤ deadband):
  steering_right, steering_left 양쪽에 center_hold_duty 동시 인가
  → 균등한 양방향 힘 → 기구적 중앙 복귀
```

**3가지 조향 상태:**

| 상태 | steering_right | steering_left |
|------|----------------|---------------|
| 우조향 | duty ON | OFF |
| 좌조향 | OFF | duty ON |
| 중앙 유지 | center_hold_duty ON | center_hold_duty ON |

---

## 주요 설정 위치

| 항목 | 파일 | 키 |
|------|------|----|
| 조향 게인 | `configs/control.yaml` | `kp` |
| 직진 속도 | `configs/control.yaml` | `base_speed` |
| 조향 최소 duty | `configs/default.yaml` | `steering_min_duty_percent` |
| 중앙 유지 duty | `configs/default.yaml` | `steering_center_hold_percent` |
| 모터 켜기/끄기 | `configs/default.yaml` | `use_actuator` |
| 실행 프레임 수 | `configs/default.yaml` | `max_frames` (0=무한) |
| 카메라 인덱스 | `configs/camera.yaml` | `camera_index` |
| ROI 범위 | `configs/camera.yaml` | `roi_top_ratio` 등 |

---

## 루프 타이밍 (실측 평균, Ultra96V2)

| 단계 | 평균 |
|------|------|
| 캡처 | ~16 ms |
| 전처리 | ~14 ms |
| DPU 추론 | ~16 ms |
| 후처리 | ~8 ms |
| 제어 + 액추에이터 | ~2 ms |
| **전체** | **~56 ms (≈18fps)** |

`target_fps: 10` 설정 시 남는 시간은 sleep으로 소비.

---

## 이슈 이력

| 증상 | 원인 | 해결 |
|------|------|------|
| 조향 전혀 안 됨 | kp=0.4 × 소오차 → duty 4~12%, DC 모터 기동 토크 부족 | `steering_min_duty_percent: 0.6` 추가 |
| 직진 시 조향 유지 안 됨 | deadband 진입 시 PWM 전원 차단 → 모터 프리휠 | `steering_center_hold_percent: 0.3` 추가 |
| HW_setting_test는 되는데 파이프라인은 안 됨 | 두 비트스트림 간 MMIO 주소 배치 상이 (동일 물리 핀, 다른 주소) | dpu.bit hwh 검증으로 default.yaml 주소 정확성 확인 |
