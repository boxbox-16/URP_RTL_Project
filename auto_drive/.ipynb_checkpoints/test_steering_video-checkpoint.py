"""
조향 모터 동작 확인 테스트 (영상 파이프라인 + 직접 하드웨어)

[테스트 1] 직접 하드웨어 스윕
  steering_cmd를 0 → +1 → -1 → 0 으로 보내서 조향 모터가 물리적으로 움직이는지 확인.
  MMIO period/duty/valid 레지스터 readback 포함.

[테스트 2] 영상 파이프라인 트레이스
  VIDEO_PATH 영상을 프레임 단위로 처리하면서 각 단계의 조향 관련 값을 출력.
  어느 단계에서 조향이 0이 되는지 확인:
    steering_error(후처리) → steering_cmd(제어기) → steering_effort(액추에이터)

실행:
  python3 -m tests.test_steering_video
  python3 -m tests.test_steering_video --video test.mp4 --frames 60 --skip-sweep
  python3 -m tests.test_steering_video --skip-sweep --steer-only --stream         # 브라우저 실시간 확인
  python3 -m tests.test_steering_video --skip-sweep --steer-only --stream --port 5001
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

# 프로젝트 루트에서 실행 (python3 -m tests.test_steering_video)
from utils.io_utils import load_yaml
from utils.image_utils import make_debug_overlay
from core.preprocess import preprocess_frame
from core.postprocess import postprocess_output
from core.controller import PController
from tools.stream_server import MJPEGServer


# ── 기본 경로 ────────────────────────────────────────────────────────────────
VIDEO_PATH   = "test.mp4"
CONFIG_DIR   = "configs"
SWEEP_PAUSE  = 1.5   # 각 단계 유지 시간 (초)


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _draw_steering_gauge(
    frame: np.ndarray,
    steering_cmd: float,
    steering_effort: float,
    duty: int,
    channel: str,
    period: int,
) -> None:
    """프레임 하단에 조향 게이지(방향 바 + 텍스트)를 in-place로 그린다."""
    h, w = frame.shape[:2]
    bar_y  = h - 50
    bar_h  = 20
    bar_x0 = w // 6
    bar_x1 = w * 5 // 6
    bar_cx = (bar_x0 + bar_x1) // 2
    bar_w  = bar_x1 - bar_x0

    # 배경 바 (회색)
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (60, 60, 60), -1)
    # 중심선 (흰색)
    cv2.line(frame, (bar_cx, bar_y - 4), (bar_cx, bar_y + bar_h + 4), (200, 200, 200), 1)

    # 조향 방향 바 (steering_cmd 기반)
    fill_w = int(abs(steering_cmd) * bar_w // 2)
    if steering_cmd > 0:  # 우회전 → 오른쪽으로
        color = (0, 140, 255)  # 주황
        cv2.rectangle(frame, (bar_cx, bar_y), (bar_cx + fill_w, bar_y + bar_h), color, -1)
        arrow_tip = (bar_cx + fill_w + 8, bar_y + bar_h // 2)
        cv2.arrowedLine(frame, (bar_cx, bar_y + bar_h // 2), arrow_tip, color, 2, tipLength=0.4)
    elif steering_cmd < 0:  # 좌회전 → 왼쪽으로
        color = (255, 100, 0)  # 파랑
        cv2.rectangle(frame, (bar_cx - fill_w, bar_y), (bar_cx, bar_y + bar_h), color, -1)
        arrow_tip = (bar_cx - fill_w - 8, bar_y + bar_h // 2)
        cv2.arrowedLine(frame, (bar_cx, bar_y + bar_h // 2), arrow_tip, color, 2, tipLength=0.4)

    # duty 퍼센트 계산
    duty_pct = duty / period * 100 if period else 0

    # 텍스트 라인
    ch_text   = channel if channel != "없음" else "NEUTRAL"
    info_text = f"cmd:{steering_cmd:+.3f}  effort:{steering_effort:+.3f}  duty:{duty}({duty_pct:.1f}%)  [{ch_text}]"
    cv2.putText(frame, info_text, (bar_x0, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # L / R 레이블
    cv2.putText(frame, "L", (bar_x0 - 18, bar_y + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "R", (bar_x1 + 4, bar_y + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)


def _load_configs() -> dict:
    cfg = {
        "default": load_yaml(f"{CONFIG_DIR}/default.yaml"),
        "camera":  load_yaml(f"{CONFIG_DIR}/camera.yaml"),
        "model":   load_yaml(f"{CONFIG_DIR}/model.yaml"),
        "control": load_yaml(f"{CONFIG_DIR}/control.yaml"),
    }
    return cfg


def _dump_mmio(actuator, channels: list[str], label: str = "") -> None:
    """지정 채널의 MMIO 레지스터(period/duty/valid)를 읽어 출력한다."""
    if not hasattr(actuator, "mmios"):
        return
    if label:
        print(f"  [{label}] MMIO readback")
    for name in channels:
        mmio = actuator.mmios.get(name)
        if mmio is None:
            print(f"    {name}: (채널 없음 - base_addrs 확인 필요)")
            continue
        p = mmio.read(0x00)
        d = mmio.read(0x04)
        v = mmio.read(0x08)
        ratio = d / p * 100 if p else 0
        status = "ON" if v else "OFF"
        print(f"    {name}: period={p}  duty={d} ({ratio:.1f}%)  valid={v} [{status}]")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 1: 직접 조향 하드웨어 스윕
# ─────────────────────────────────────────────────────────────────────────────

def test_steering_sweep(actuator, cfg: dict) -> None:
    """steering_cmd를 수동으로 보내 조향 서보가 물리적으로 움직이는지 확인한다."""
    print("\n" + "=" * 60)
    print("[테스트 1] 직접 조향 하드웨어 스윕")
    print("=" * 60)

    act_cfg    = cfg["default"]["actuator"]
    motor_cfg  = act_cfg.get("motors", {})
    steer_r    = motor_cfg.get("steering_right", "steering_right")
    steer_l    = motor_cfg.get("steering_left",  "steering_left")
    period     = act_cfg.get("period_size", 600600)
    duty_pct   = act_cfg.get("steering_duty_percent", 0.25)
    deadband   = act_cfg.get("steering_feedback_deadband", 0.05)

    print(f"\n설정 확인:")
    print(f"  period_size              = {period}")
    print(f"  steering_duty_percent    = {duty_pct}  → 최대 duty = {int(period * duty_pct)}")
    print(f"  steering_feedback_deadband = {deadband}  → steering_cmd > {deadband} 일 때만 동작")
    print(f"  steering_right 채널      = {steer_r}")
    print(f"  steering_left  채널      = {steer_l}")

    steps = [
        ("중립 (0.0)",    {"steering_cmd": 0.0,  "speed_cmd": 0.0, "mode": "drive"}),
        ("우회전 (+1.0)", {"steering_cmd": 1.0,  "speed_cmd": 0.0, "mode": "drive"}),
        ("중립 (0.0)",    {"steering_cmd": 0.0,  "speed_cmd": 0.0, "mode": "drive"}),
        ("좌회전 (-1.0)", {"steering_cmd": -1.0, "speed_cmd": 0.0, "mode": "drive"}),
        ("중립 (0.0)",    {"steering_cmd": 0.0,  "speed_cmd": 0.0, "mode": "drive"}),
        # deadband 경계값 테스트
        ("경계+ (0.06)",  {"steering_cmd": 0.06, "speed_cmd": 0.0, "mode": "drive"}),
        ("경계- (0.04)",  {"steering_cmd": 0.04, "speed_cmd": 0.0, "mode": "drive"}),
        ("STOP",          {"steering_cmd": 0.0,  "speed_cmd": 0.0, "mode": "stop"}),
    ]

    print()
    for label, cmd in steps:
        print(f"\n>>> {label}  steering_cmd={cmd['steering_cmd']}")
        act_result = actuator.apply_control(cmd)
        effort  = act_result.get("steering_effort", "N/A")
        duty    = act_result.get("steering_duty",   "N/A")
        channel = act_result.get("steering_channel","없음")
        mode    = act_result.get("mode", "?")
        print(f"    effort={effort}  steering_duty={duty}  활성채널={channel}  mode={mode}")
        _dump_mmio(actuator, [steer_r, steer_l])

        if label != "STOP":
            time.sleep(SWEEP_PAUSE)

    actuator.stop()
    print("\n[테스트 1 완료] 위 결과에서 우회전/좌회전 시 duty>0 & valid=1 이면 하드웨어 정상")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 2: 영상 파이프라인 조향 트레이스
# ─────────────────────────────────────────────────────────────────────────────

def _run_mock_dpu(input_image: np.ndarray, model_cfg: dict) -> np.ndarray:
    """DPU 없이 동작하는 Mock 추론기: 이미지 중앙 절반 폭을 차선으로 가정."""
    h = model_cfg.get("input_height", 224)
    w = model_cfg.get("input_width",  224)
    mask = np.zeros((1, h, w, 1), dtype=np.float32)
    # 화면 중앙 40~60% 열에 차선이 있다고 가정 (steering_error≈0 기대)
    cx = w // 2
    mask[0, :, cx - w // 10 : cx + w // 10, 0] = 1.0
    return mask


class _MockDPU:
    """DPU를 사용할 수 없을 때 대체하는 Mock 객체."""
    def __init__(self, model_cfg: dict):
        self._mcfg = model_cfg

    def run(self, image: np.ndarray) -> np.ndarray:
        return _run_mock_dpu(image, self._mcfg)


def test_video_pipeline(dpu_runner, actuator, cfg: dict,
                        video_path: str = VIDEO_PATH,
                        max_frames: int = 100,
                        steer_only: bool = False,
                        stream_server: MJPEGServer | None = None) -> None:
    """영상의 각 프레임에 대해 파이프라인을 실행하고 조향 관련 값을 출력한다."""
    print("\n" + "=" * 60)
    print(f"[테스트 2] 영상 파이프라인 트레이스: {video_path}")
    if steer_only:
        print("  [조향 전용 모드] 뒷바퀴 speed_cmd=0 강제 적용")
    if stream_server is not None:
        print(f"  [스트리밍 ON] 브라우저에서 확인하세요")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 영상 파일을 열 수 없습니다: {video_path}")
        return

    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps           = cap.get(cv2.CAP_PROP_FPS)
    frame_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n영상 정보: {frame_w}x{frame_h}  {fps:.1f}fps  총 {total_frames}프레임")
    print(f"처리 프레임 수: {min(max_frames, total_frames)}\n")

    camera_cfg  = cfg["camera"]
    model_cfg   = cfg["model"]
    control_cfg = cfg["control"]
    controller  = PController(control_cfg)

    act_cfg   = cfg["default"]["actuator"]
    motor_cfg = act_cfg.get("motors", {})
    steer_r   = motor_cfg.get("steering_right", "steering_right")
    steer_l   = motor_cfg.get("steering_left",  "steering_left")
    deadband  = act_cfg.get("steering_feedback_deadband", 0.05)
    kp        = control_cfg.get("kp", 0.4)

    # 헤더 출력
    header = (
        f"{'프레임':>6} | {'s_error':>8} | {'s_cmd':>7} | "
        f"{'effort':>7} | {'duty':>6} | {'채널':<14} | 판정"
    )
    print(header)
    print("-" * len(header))

    stats = {"total": 0, "lane_ok": 0, "steer_nonzero_error": 0, "steer_activated": 0}
    issues = []  # 오류 케이스 수집

    for frame_id in range(min(max_frames, total_frames)):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        # 전처리
        input_image, pre_meta = preprocess_frame(frame_bgr, camera_cfg, model_cfg)

        # DPU 추론
        raw_output = dpu_runner.run(input_image)

        # 후처리
        post = postprocess_output(raw_output, pre_meta, model_cfg, control_cfg)

        # 제어기
        s_error = post["steering_error"]
        ctrl    = controller.compute(s_error)
        s_cmd   = ctrl["steering_cmd"]

        # 조향 전용 모드: 뒷바퀴 정지, 조향만 동작
        if steer_only:
            ctrl = dict(ctrl, speed_cmd=0.0)

        # 액추에이터
        act = actuator.apply_control(ctrl)
        effort  = act.get("steering_effort", 0.0)
        duty    = act.get("steering_duty",   0)
        channel = act.get("steering_channel") or "없음"

        # 스트리밍: 디버그 오버레이 + 조향 게이지 합성 후 push
        if stream_server is not None:
            debug_frame = make_debug_overlay(
                frame_bgr=frame_bgr,
                preprocess_meta=pre_meta,
                postprocess_result=post,
                control_command=ctrl,
                actuator_status=act,
            )
            _draw_steering_gauge(
                debug_frame,
                steering_cmd=float(s_cmd),
                steering_effort=float(effort),
                duty=int(duty) if duty else 0,
                channel=channel,
                period=act_cfg.get("period_size", 600600),
            )
            stream_server.push_frame(debug_frame)
            stream_server.push_telemetry({
                "frame_id":       frame_id,
                "steering_error": round(float(s_error), 4),
                "steering_cmd":   round(float(s_cmd), 4),
                "speed_cmd":      round(float(ctrl.get("speed_cmd", 0.0)), 4),
                "steering_effort":round(float(effort), 4),
                "steering_duty":  duty,
                "steering_channel": channel,
                "valid":          bool(post["valid"]),
                "reference_x":    post["reference_point"][0],
                "reference_y":    post["reference_point"][1],
                "heading_error":  round(float(post["heading_error"]), 4),
                "lane_pixels":    post.get("lane_pixels", 0),
                "mode":           act.get("mode", "unknown"),
            })

        stats["total"] += 1
        if post["valid"]:
            stats["lane_ok"] += 1
        if abs(s_error) >= 0.01:
            stats["steer_nonzero_error"] += 1
        if duty and duty > 0:
            stats["steer_activated"] += 1

        # 판정 문자열 (문제 있는 경우 강조)
        note = "OK"
        if abs(s_error) > 0.1 and duty == 0:
            note = "*** 오류: 오차있으나 duty=0 ***"
            issues.append((frame_id, s_error, s_cmd, effort, duty, channel))
        elif not post["valid"]:
            note = "차선미검출"

        row = (
            f"{frame_id:>6} | {s_error:>8.4f} | {s_cmd:>7.4f} | "
            f"{effort:>7.4f} | {duty:>6} | {channel:<14} | {note}"
        )
        print(row)

    cap.release()
    actuator.stop()

    # ── 요약 출력 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[요약]")
    print(f"  처리 프레임   : {stats['total']}")
    print(f"  차선 검출 성공: {stats['lane_ok']} ({stats['lane_ok']/max(stats['total'],1)*100:.1f}%)")
    print(f"  조향오차 ≥0.01: {stats['steer_nonzero_error']}")
    print(f"  조향 duty>0   : {stats['steer_activated']}")

    print("\n[단계별 진단]")

    # steering_error 분포로 후처리 이상 여부 확인
    if stats["lane_ok"] == 0:
        print("  [X] postprocess: 차선이 한 번도 검출되지 않음 → 모델/ROI 설정 확인 필요")
    elif stats["steer_nonzero_error"] == 0:
        print("  [X] postprocess: 차선은 검출됐으나 steering_error가 항상 0 → 차선이 항상 중앙에 있음")
    else:
        print(f"  [O] postprocess: {stats['steer_nonzero_error']}개 프레임에서 비제로 조향 오차 발생")

    # controller 이상 여부
    if stats["steer_nonzero_error"] > 0 and stats["steer_activated"] == 0:
        print(f"  [X] controller/actuator: 오차가 있는데 조향이 0번 활성화됨")
        print(f"      → kp={kp} × steering_error 가 deadband({deadband}) 이하인지 확인")
        print(f"      → steering_error가 {deadband/kp:.3f} 이하이면 조향 발동 안됨 (kp={kp}, deadband={deadband})")
    elif stats["steer_activated"] == 0:
        print("  [?] actuator: 조향이 한 번도 활성화되지 않음 (오차 자체가 없어서일 수 있음)")
    else:
        print(f"  [O] actuator: {stats['steer_activated']}개 프레임에서 조향 duty 활성화됨")

    if issues:
        print(f"\n  [경고] 오차 있으나 duty=0인 프레임 {len(issues)}개 (처음 5개):")
        for fid, se, sc, ef, d, ch in issues[:5]:
            print(f"    frame={fid}  s_error={se:.4f}  s_cmd={sc:.4f}  effort={ef:.4f}  duty={d}  ch={ch}")
            # 원인 추론
            if abs(sc) < deadband:
                needed_error = deadband / kp
                print(f"      → s_cmd={sc:.4f} < deadband({deadband}). 조향 발동하려면 |s_error| > {needed_error:.3f} 필요")
                print(f"         현재 kp={kp}이 너무 낮거나 deadband가 너무 큼 → control.yaml의 kp 값 올리거나")
                print(f"         default.yaml의 steering_feedback_deadband 낮추는 것 고려")

    # MMIO 최종 상태 확인
    print()
    _dump_mmio(actuator, [steer_r, steer_l], "최종 MMIO 상태")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="조향 모터 동작 확인 테스트")
    p.add_argument("--video",       default=VIDEO_PATH, help=f"영상 경로 (기본: {VIDEO_PATH})")
    p.add_argument("--frames",      type=int, default=9999, help="처리할 최대 프레임 수 (기본: 전체)")
    p.add_argument("--skip-sweep",  action="store_true",   help="하드웨어 스윕 테스트 건너뜀")
    p.add_argument("--skip-video",  action="store_true",   help="영상 파이프라인 테스트 건너뜀")
    p.add_argument("--steer-only",  action="store_true",   help="영상 테스트 시 뒷바퀴 끄고 조향만 동작")
    p.add_argument("--stream",      action="store_true",   help="브라우저 실시간 스트리밍 활성화")
    p.add_argument("--port",        type=int, default=5000, help="스트리밍 서버 포트 (기본: 5000)")
    p.add_argument("--mock-dpu",    action="store_true",   help="DPU 없이 Mock 추론기 사용")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg  = _load_configs()

    # ── DPU / 오버레이 초기화 ─────────────────────────────────────────────
    dpu_runner = None
    if not args.mock_dpu:
        try:
            from pynq_dpu import DpuOverlay
            from core.dpu_runner import DPURunner

            bit_path = cfg["default"]["actuator"].get("overlay_path", "configs/dpu/dpu.bit")
            print(f"FPGA 오버레이 로드 중: {bit_path}")
            DpuOverlay(bit_path)
            print("FPGA 오버레이 로드 완료")

            dpu_runner = DPURunner(cfg["model"])
            dpu_runner.load_model()
            dpu_runner.create_runner()
            print(f"DPU 로드 완료  input={dpu_runner.get_input_shape()}  output={dpu_runner.get_output_shape()}")
        except Exception as exc:
            print(f"[경고] DPU 초기화 실패: {exc}")
            print("       --mock-dpu 플래그로 Mock DPU를 사용합니다.")
            dpu_runner = None

    if dpu_runner is None:
        print("[Mock DPU 사용] 실제 차선 추론 없이 중앙 차선을 가정합니다.")
        dpu_runner = _MockDPU(cfg["model"])

    # ── 액추에이터 초기화 ─────────────────────────────────────────────────
    try:
        from core.actuator import PynqMMIOActuator
        actuator = PynqMMIOActuator(cfg["default"]["actuator"])
        actuator.initialize()
        print("PynqMMIOActuator 초기화 완료 (실제 MMIO 제어)")
    except Exception as exc:
        print(f"[경고] PynqMMIOActuator 초기화 실패: {exc}")
        print("       DryRunActuator로 대체합니다 (MMIO 쓰기 없음).")
        from core.actuator import DryRunActuator
        actuator = DryRunActuator(cfg["default"]["actuator"])
        actuator.initialize()

    # ── 스트리밍 서버 초기화 ─────────────────────────────────────────────
    stream_server = None
    if args.stream:
        stream_server = MJPEGServer(port=args.port, quality=60)
        stream_server.start()
        board_ip = None
        try:
            import socket
            board_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            pass
        print(f"\n[스트리밍 서버 시작]")
        for url in stream_server.urls():
            print(f"  브라우저에서 접속: {url}")
        print()

    # ── 테스트 실행 ──────────────────────────────────────────────────────
    try:
        if not args.skip_sweep:
            test_steering_sweep(actuator, cfg)

        if not args.skip_video:
            test_video_pipeline(dpu_runner, actuator, cfg,
                                video_path=args.video,
                                max_frames=args.frames,
                                steer_only=args.steer_only,
                                stream_server=stream_server)
    finally:
        actuator.stop()
        actuator.close()
        if stream_server is not None:
            stream_server.stop()
        print("\n[테스트 종료] 모든 모터 정지 완료")


if __name__ == "__main__":
    main()
