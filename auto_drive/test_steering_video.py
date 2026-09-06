"""
조향 모터 동작 확인 테스트 (영상 파이프라인 + 직접 하드웨어)

[테스트 1] 직접 하드웨어 스윕
  steering_cmd를 0 → +1 → -1 → 0 으로 보내서 조향 모터가 물리적으로 움직이는지 확인.
  MMIO period/duty/valid 레지스터 readback 포함.

[테스트 2] 영상 파이프라인 트레이스
  VIDEO_PATH 영상을 프레임 단위로 처리하면서 각 단계의 조향 관련 값을 출력.
  어느 단계에서 조향이 0이 되는지 확인:
    steering_error(후처리) → steering_cmd(제어기) → steering_effort(액추에이터)

실행 (프로젝트 루트에서):
  python3 test_steering_video.py
  python3 test_steering_video.py --video test.mp4 --frames 60 --skip-sweep
  python3 test_steering_video.py --skip-sweep --steer-only
  python3 test_steering_video.py --skip-sweep --steer-only --stream --port 5000
  python3 test_steering_video.py --mock-dpu --skip-sweep
                                                                                                                                                                   
  처음엔 Mock DPU로 파이프라인만 확인:                                                                                                                             
  python3 test_steering_video.py --mock-dpu --skip-sweep                                                                                                                                                                                                                                       
  조향 하드웨어만 테스트 (모터 직접 움직임):                                                                                                                       
  python3 test_steering_video.py --skip-video                                                                                                                      
                                                                                                                                                                   
  실제 DPU + 영상, 조향만 (뒷바퀴 정지):                                                                                                                           
  python3 test_steering_video.py --skip-sweep --steer-only                                                                                                         
                                                                                                                                                                   
  브라우저로 실시간 확인하면서:                                                                                                                                    
  python3 test_steering_video.py --skip-sweep --steer-only --stream --port 5000                                                                                    
  # → http://<보드IP>:5000/stream 접속  
"""

from __future__ import annotations

import argparse
import io
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── 기본 경로 ────────────────────────────────────────────────────────────────
VIDEO_PATH  = "test.mp4"
CONFIG_DIR  = "configs"
SWEEP_PAUSE = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_configs() -> dict:
    return {
        "default": load_yaml(f"{CONFIG_DIR}/default.yaml"),
        "camera":  load_yaml(f"{CONFIG_DIR}/camera.yaml"),
        "model":   load_yaml(f"{CONFIG_DIR}/model.yaml"),
        "control": load_yaml(f"{CONFIG_DIR}/control.yaml"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 전처리
# ─────────────────────────────────────────────────────────────────────────────

def _compute_roi(frame_bgr: np.ndarray, cfg_camera: dict):
    h, w = frame_bgr.shape[:2]
    top    = max(0, min(int(h * cfg_camera["roi_top_ratio"]),    h - 1))
    bottom = max(top + 1, min(int(h * cfg_camera["roi_bottom_ratio"]), h))
    left   = max(0, min(int(w * cfg_camera["roi_left_ratio"]),   w - 1))
    right  = max(left + 1, min(int(w * cfg_camera["roi_right_ratio"]),  w))
    return top, bottom, left, right


def preprocess_frame(frame_bgr: np.ndarray, camera_cfg: dict, model_cfg: dict):
    h, w = frame_bgr.shape[:2]
    top, bottom, left, right = _compute_roi(frame_bgr, camera_cfg)
    roi_bgr = frame_bgr[top:bottom, left:right]
    meta = {
        "orig_h": h, "orig_w": w,
        "roi_top": top, "roi_bottom": bottom,
        "roi_left": left, "roi_right": right,
        "roi_h": roi_bgr.shape[0], "roi_w": roi_bgr.shape[1],
        "input_h": model_cfg["input_height"],
        "input_w": model_cfg["input_width"],
        "channel_order": model_cfg.get("channel_order", "RGB"),
    }
    iw, ih = model_cfg["input_width"], model_cfg["input_height"]
    resized = cv2.resize(roi_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
    if model_cfg.get("channel_order", "RGB").upper() == "RGB":
        converted = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        converted = resized
    out = converted.astype(np.float32)
    if model_cfg.get("normalize", True):
        out *= (1.0 / 255.0)
    mean = model_cfg.get("mean")
    std  = model_cfg.get("std")
    if mean is not None:
        out -= np.array(mean, dtype=np.float32)
    if std is not None:
        out /= np.array(std, dtype=np.float32)
    return out, meta


# ─────────────────────────────────────────────────────────────────────────────
# 후처리
# ─────────────────────────────────────────────────────────────────────────────

_morph_kernels: dict = {}


def _get_kernel(size: int) -> np.ndarray:
    if size not in _morph_kernels:
        _morph_kernels[size] = np.ones((size, size), np.uint8)
    return _morph_kernels[size]


def _logits_to_mask(raw_output: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    arr = raw_output
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise RuntimeError(f"지원하지 않는 output shape: {raw_output.shape}")
    return ((arr > threshold).astype(np.uint8)) * 255


def postprocess_output(raw_output: np.ndarray, pre_meta: dict,
                       model_cfg: dict, control_cfg: dict) -> dict:
    mask = _logits_to_mask(raw_output, threshold=model_cfg["threshold"])

    kernel = _get_kernel(control_cfg["morph_kernel_size"])
    filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    min_area = int(control_cfg.get("min_component_area", 0))
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered, connectivity=8)
        if num_labels > 1:
            keep = np.zeros(num_labels, dtype=np.uint8)
            keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_area).astype(np.uint8)
            filtered = (keep[labels] * 255).astype(np.uint8)

    roi_mask = cv2.resize(filtered, (pre_meta["roi_w"], pre_meta["roi_h"]),
                          interpolation=cv2.INTER_NEAREST)

    row_step = int(control_cfg.get("centerline_row_step", 5))
    min_pix  = int(control_cfg.get("min_pixels_per_sample_row", 1))
    h, w     = roi_mask.shape[:2]
    ys       = np.arange(h - 1, -1, -max(1, row_step))
    rows     = roi_mask[ys] > 0
    row_sums = rows.sum(axis=1)
    valid_rows = row_sums >= max(1, min_pix)
    cl_pts: list = []
    if valid_rows.any():
        xs     = np.arange(w, dtype=np.int32)
        x_sums = (rows * xs).sum(axis=1)
        cxs    = np.where(valid_rows, x_sums // row_sums.clip(1), 0)
        cl_pts = [(int(cxs[i]), int(ys[i])) for i in range(len(ys)) if valid_rows[i]]

    target_y = int(h * control_cfg["reference_row_ratio"])
    ref_pt = min(cl_pts, key=lambda p: abs(p[1] - target_y)) if cl_pts else (None, None)

    ref_x = ref_pt[0]
    steer_err = float((ref_x - w / 2.0) / (w / 2.0)) if ref_x is not None else 0.0

    if len(cl_pts) >= 2:
        p1, p2  = cl_pts[0], cl_pts[-1]
        dx, dy  = p2[0] - p1[0], p1[1] - p2[1]
        heading_err = float(dx / dy) if dy != 0 else 0.0
    else:
        heading_err = 0.0

    lane_pixels = int(np.count_nonzero(roi_mask))
    valid = (
        ref_x is not None
        and lane_pixels >= int(control_cfg.get("min_lane_pixels", 1))
        and len(cl_pts) >= int(control_cfg.get("min_centerline_points", 1))
    )
    return {
        "roi_mask":          roi_mask,
        "centerline_points": cl_pts,
        "reference_point":   ref_pt,
        "steering_error":    steer_err,
        "heading_error":     heading_err,
        "lane_pixels":       lane_pixels,
        "valid":             valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# P 제어기
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PController:
    def __init__(self, cfg: dict):
        self.kp           = cfg["kp"]
        self.deadband     = cfg["deadband"]
        self.max_steering = cfg["max_steering"]
        self.min_steering = cfg["min_steering"]
        self.base_speed   = cfg["base_speed"]
        self.min_speed    = cfg["min_speed"]
        self.max_speed    = cfg["max_speed"]

    def compute(self, steering_error: float) -> dict:
        if abs(steering_error) < self.deadband:
            steering_error = 0.0
        steering_cmd = _clamp(self.kp * steering_error, self.min_steering, self.max_steering)
        speed_cmd    = _clamp(
            self.base_speed * max(0.4, 1.0 - abs(steering_cmd)),
            self.min_speed, self.max_speed,
        )
        return {"steering_cmd": float(steering_cmd), "speed_cmd": float(speed_cmd), "mode": "drive"}


# ─────────────────────────────────────────────────────────────────────────────
# 액추에이터
# ─────────────────────────────────────────────────────────────────────────────

REG_PERIOD = 0x00
REG_DUTY   = 0x04
REG_VALID  = 0x08

try:
    from pynq import MMIO  # type: ignore
except Exception:
    MMIO = None


class DryRunActuator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.initialized       = False
        self._current_steering = 0.0

    def initialize(self):
        self.initialized = True
        print("[DryRun] 액추에이터 초기화")

    def apply_control(self, control_command: dict) -> dict:
        steer = _clamp(float(control_command.get("steering_cmd", 0.0)), -1.0, 1.0)
        speed = _clamp(float(control_command.get("speed_cmd",    0.0)), -1.0, 1.0)
        mode  = control_command.get("mode", "drive")
        if mode == "stop":
            steer = speed = 0.0
        self._current_steering = steer
        return {
            "steering_effort":  steer,
            "steering_duty":    0,
            "steering_channel": None,
            "applied_speed":    speed,
            "mode":             "stop" if mode == "stop" else "dry_run",
        }

    def stop(self):
        self._current_steering = 0.0

    def close(self):
        self.stop()
        self.initialized = False


class PynqMMIOActuator:
    def __init__(self, cfg: dict | None = None):
        self.cfg         = cfg or {}
        self.mmios: dict = {}
        self.initialized = False
        self._watchdog_timer = None
        self._watchdog_lock  = threading.RLock()

    @staticmethod
    def _parse_addr(addr) -> int:
        return int(addr, 0) if isinstance(addr, str) else int(addr)

    def initialize(self):
        if MMIO is None:
            raise RuntimeError("pynq.MMIO import 실패. PYNQ 보드 환경 필요")
        motor_cfg  = self.cfg.get("motors", {})
        base_addrs = motor_cfg.get("base_addrs", {})
        mmio_range = int(self.cfg.get("mmio_range", 0x10000))
        if not base_addrs:
            raise RuntimeError("모터 주소 설정 없음")
        self.mmios = {
            name: MMIO(self._parse_addr(addr), mmio_range)
            for name, addr in base_addrs.items()
        }
        self._period             = int(self.cfg.get("period_size", 600600))
        self._drive_duty_pct     = float(self.cfg.get("drive_duty_percent", 1.0))
        self._steer_duty_pct     = float(self.cfg.get("steering_duty_percent", 1.0))
        self._steer_min_duty_pct = float(self.cfg.get("steering_min_duty_percent", 0.0))
        center_pct               = float(self.cfg.get("steering_center_hold_percent", 0.0))
        self._steer_center_duty  = int(self._period * self._steer_duty_pct * center_pct)
        self._fwd_channels       = motor_cfg.get("drive_channels", [])
        self._bwd_channels       = motor_cfg.get("reverse_channels", [])
        self._r_name             = motor_cfg.get("steering_right")
        self._l_name             = motor_cfg.get("steering_left")
        self._steer_deadband     = float(self.cfg.get("steering_feedback_deadband", 0.05))
        self._timeout_sec        = float(self.cfg.get("command_timeout_sec", 1.0))
        for name in self.mmios:
            self.mmios[name].write(REG_PERIOD, self._period)
            self.mmios[name].write(REG_DUTY,   self._period)
            self.mmios[name].write(REG_VALID,  0)
        self.initialized = True
        print(f"[MMIO] 액추에이터 초기화 완료 (Period: {self._period})")

    def _write_duty(self, name, value):
        if name in self.mmios:
            self.mmios[name].write(REG_DUTY, int(value))

    def _write_valid(self, name, enable: bool):
        if name in self.mmios:
            if enable:
                self.mmios[name].write(REG_PERIOD, self._period)
            self.mmios[name].write(REG_VALID, 1 if enable else 0)

    def _steer_duty(self, effort_abs: float) -> int:
        min_pct   = self._steer_min_duty_pct
        effective = min_pct + (1.0 - min_pct) * _clamp(effort_abs, 0.0, 1.0)
        return int(self._period * self._steer_duty_pct * effective)

    def _cancel_watchdog(self):
        with self._watchdog_lock:
            t = self._watchdog_timer
            self._watchdog_timer = None
            if t is not None and t is not threading.current_thread():
                t.cancel()

    def _arm_watchdog(self):
        if self._timeout_sec <= 0:
            return
        with self._watchdog_lock:
            self._cancel_watchdog()
            t = threading.Timer(self._timeout_sec, self.stop)
            t.daemon = True
            t.start()
            self._watchdog_timer = t

    def apply_control(self, control_command: dict) -> dict:
        steer_cmd = _clamp(float(control_command.get("steering_cmd", 0.0)), -1.0, 1.0)
        speed_cmd = _clamp(float(control_command.get("speed_cmd",    0.0)), -1.0, 1.0)
        mode      = control_command.get("mode", "drive")

        if mode == "stop":
            self.stop()
            return {"steering_effort": 0.0, "steering_duty": 0,
                    "steering_channel": None, "mode": "stop"}

        effort = steer_cmd  # no feedback sensor in this test
        duty   = 0
        channel = None

        if abs(effort) < self._steer_deadband:
            if self._steer_center_duty > 0:
                self._write_duty(self._r_name, self._steer_center_duty)
                self._write_duty(self._l_name, self._steer_center_duty)
                self._write_valid(self._r_name, True)
                self._write_valid(self._l_name, True)
            else:
                self._write_valid(self._r_name, False)
                self._write_valid(self._l_name, False)
        elif effort > 0:
            duty = self._steer_duty(effort)
            channel = self._r_name
            self._write_valid(self._l_name, False)
            self._write_duty(self._r_name, duty)
            self._write_valid(self._r_name, True)
        else:
            duty = self._steer_duty(-effort)
            channel = self._l_name
            self._write_valid(self._r_name, False)
            self._write_duty(self._l_name, duty)
            self._write_valid(self._l_name, True)

        # drive channels
        drive_duty = int(self._period * self._drive_duty_pct * abs(speed_cmd))
        for name in self._fwd_channels + self._bwd_channels:
            self._write_valid(name, False)
        if abs(speed_cmd) >= 0.05:
            for name in (self._fwd_channels if speed_cmd > 0 else self._bwd_channels):
                self._write_duty(name, drive_duty)
                self._write_valid(name, True)

        self._arm_watchdog()
        return {
            "steering_effort":  float(effort),
            "steering_duty":    duty,
            "steering_channel": channel,
            "mode":             "drive",
        }

    def stop(self):
        for name in self.mmios:
            self.mmios[name].write(REG_VALID, 0)

    def close(self):
        self._cancel_watchdog()
        self.stop()
        self.initialized = False


# ─────────────────────────────────────────────────────────────────────────────
# Mock DPU
# ─────────────────────────────────────────────────────────────────────────────

def _run_mock_dpu(input_image: np.ndarray, model_cfg: dict) -> np.ndarray:
    h = model_cfg.get("input_height", 256)
    w = model_cfg.get("input_width",  256)
    mask = np.zeros((1, h, w, 1), dtype=np.float32)
    cx = w // 2
    mask[0, :, cx - w // 10: cx + w // 10, 0] = 1.0
    return mask


class _MockDPU:
    def __init__(self, model_cfg: dict):
        self._mcfg = model_cfg

    def run(self, image: np.ndarray) -> np.ndarray:
        return _run_mock_dpu(image, self._mcfg)


# ─────────────────────────────────────────────────────────────────────────────
# 디버그 오버레이 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_debug_overlay(
    frame_bgr: np.ndarray,
    preprocess_meta: dict,
    postprocess_result: dict,
    control_command: dict,
    actuator_status: dict,
) -> np.ndarray:
    """원본 프레임에 차선 마스크·중심선·기준점·조향 화살표를 합성한다."""
    out = frame_bgr.copy()
    post = postprocess_result

    roi_top  = preprocess_meta.get("roi_top", 0)
    roi_left = preprocess_meta.get("roi_left", 0)

    roi_mask = post.get("roi_mask")
    if roi_mask is not None:
        green = np.zeros_like(out)
        green[roi_top: roi_top + roi_mask.shape[0],
              roi_left: roi_left + roi_mask.shape[1], 1] = roi_mask
        out = np.where(green > 0,
                       (out * 0.5 + green * 0.5).clip(0, 255).astype(np.uint8),
                       out)

    for px, py in post.get("centerline_points", [])[::3]:
        cv2.circle(out, (px + roi_left, py + roi_top), 2, (255, 140, 0), -1)

    ref_pt = post.get("reference_point", (None, None))
    if ref_pt[0] is not None:
        cv2.circle(out, (ref_pt[0] + roi_left, ref_pt[1] + roi_top), 7, (0, 0, 255), -1)

    h, w = out.shape[:2]
    steer = control_command.get("steering_cmd", 0.0)
    cx, cy = w // 2, h - 20
    end_x  = cx + int(steer * w * 0.3)
    cv2.arrowedLine(out, (cx, cy), (end_x, cy), (0, 255, 255), 2, tipLength=0.3)

    err   = post.get("steering_error", 0.0)
    valid = post.get("valid", False)
    speed = control_command.get("speed_cmd", 0.0)
    label = "OK" if valid else "INVALID"
    txt   = f"err:{err:+.3f} steer:{steer:+.3f} spd:{speed:.2f} [{label}]"
    cv2.putText(out, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG 스트리밍 서버 (브라우저에서 http://<ip>:<port>/stream 접속)
# ─────────────────────────────────────────────────────────────────────────────

class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # 콘솔 로그 억제

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = self.server.mjpeg_server.get_frame()
                    if jpeg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


class MJPEGServer:
    def __init__(self, port: int = 5000, quality: int = 60):
        self._port    = port
        self._quality = quality
        self._lock    = threading.Lock()
        self._frame: bytes | None = None
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def push_frame(self, frame_bgr: np.ndarray):
        ok, jpeg = cv2.imencode(".jpg", frame_bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if ok:
            with self._lock:
                self._frame = jpeg.tobytes()

    def push_telemetry(self, data: dict):
        pass

    def start(self):
        server = HTTPServer(("", self._port), _MJPEGHandler)
        server.mjpeg_server = self  # handler가 참조할 수 있도록
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()

    def urls(self) -> list[str]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "localhost"
        return [f"http://{ip}:{self._port}/stream", f"http://localhost:{self._port}/stream"]


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
    h, w   = frame.shape[:2]
    bar_y  = h - 50
    bar_h  = 20
    bar_x0 = w // 6
    bar_x1 = w * 5 // 6
    bar_cx = (bar_x0 + bar_x1) // 2
    bar_w  = bar_x1 - bar_x0

    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (60, 60, 60), -1)
    cv2.line(frame, (bar_cx, bar_y - 4), (bar_cx, bar_y + bar_h + 4), (200, 200, 200), 1)

    fill_w = int(abs(steering_cmd) * bar_w // 2)
    if steering_cmd > 0:
        color = (0, 140, 255)
        cv2.rectangle(frame, (bar_cx, bar_y), (bar_cx + fill_w, bar_y + bar_h), color, -1)
        cv2.arrowedLine(frame, (bar_cx, bar_y + bar_h // 2),
                        (bar_cx + fill_w + 8, bar_y + bar_h // 2), color, 2, tipLength=0.4)
    elif steering_cmd < 0:
        color = (255, 100, 0)
        cv2.rectangle(frame, (bar_cx - fill_w, bar_y), (bar_cx, bar_y + bar_h), color, -1)
        cv2.arrowedLine(frame, (bar_cx, bar_y + bar_h // 2),
                        (bar_cx - fill_w - 8, bar_y + bar_h // 2), color, 2, tipLength=0.4)

    duty_pct  = duty / period * 100 if period else 0
    ch_text   = channel if channel and channel != "없음" else "NEUTRAL"
    info_text = f"cmd:{steering_cmd:+.3f}  effort:{steering_effort:+.3f}  duty:{duty}({duty_pct:.1f}%)  [{ch_text}]"
    cv2.putText(frame, info_text, (bar_x0, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "L", (bar_x0 - 18, bar_y + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "R", (bar_x1 + 4, bar_y + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)


def _dump_mmio(actuator, channels: list[str], label: str = "") -> None:
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
        ratio  = d / p * 100 if p else 0
        status = "ON" if v else "OFF"
        print(f"    {name}: period={p}  duty={d} ({ratio:.1f}%)  valid={v} [{status}]")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 1: 직접 조향 하드웨어 스윕
# ─────────────────────────────────────────────────────────────────────────────

def test_steering_sweep(actuator, cfg: dict) -> None:
    print("\n" + "=" * 60)
    print("[테스트 1] 직접 조향 하드웨어 스윕")
    print("=" * 60)

    act_cfg   = cfg["default"]["actuator"]
    motor_cfg = act_cfg.get("motors", {})
    steer_r   = motor_cfg.get("steering_right", "steering_right")
    steer_l   = motor_cfg.get("steering_left",  "steering_left")
    period    = act_cfg.get("period_size", 600600)
    duty_pct  = act_cfg.get("steering_duty_percent", 0.25)
    deadband  = act_cfg.get("steering_feedback_deadband", 0.05)

    print(f"\n설정 확인:")
    print(f"  period_size              = {period}")
    print(f"  steering_duty_percent    = {duty_pct}  → 최대 duty = {int(period * duty_pct)}")
    print(f"  steering_feedback_deadband = {deadband}  → |steering_cmd| > {deadband} 일 때만 동작")
    print(f"  steering_right 채널      = {steer_r}")
    print(f"  steering_left  채널      = {steer_l}")

    steps = [
        ("중립 (0.0)",    {"steering_cmd": 0.0,  "speed_cmd": 0.0, "mode": "drive"}),
        ("우회전 (+1.0)", {"steering_cmd": 1.0,  "speed_cmd": 0.0, "mode": "drive"}),
        ("중립 (0.0)",    {"steering_cmd": 0.0,  "speed_cmd": 0.0, "mode": "drive"}),
        ("좌회전 (-1.0)", {"steering_cmd": -1.0, "speed_cmd": 0.0, "mode": "drive"}),
        ("중립 (0.0)",    {"steering_cmd": 0.0,  "speed_cmd": 0.0, "mode": "drive"}),
        ("경계+ (0.06)",  {"steering_cmd": 0.06, "speed_cmd": 0.0, "mode": "drive"}),
        ("경계- (0.04)",  {"steering_cmd": 0.04, "speed_cmd": 0.0, "mode": "drive"}),
        ("STOP",          {"steering_cmd": 0.0,  "speed_cmd": 0.0, "mode": "stop"}),
    ]

    for label, cmd in steps:
        print(f"\n>>> {label}  steering_cmd={cmd['steering_cmd']}")
        act_result = actuator.apply_control(cmd)
        effort  = act_result.get("steering_effort", "N/A")
        duty    = act_result.get("steering_duty",   "N/A")
        channel = act_result.get("steering_channel", "없음")
        mode    = act_result.get("mode", "?")
        print(f"    effort={effort}  steering_duty={duty}  활성채널={channel}  mode={mode}")
        _dump_mmio(actuator, [steer_r, steer_l])
        if label != "STOP":
            time.sleep(SWEEP_PAUSE)

    actuator.stop()
    print("\n[테스트 1 완료] 우회전/좌회전 시 duty>0 & valid=1 이면 하드웨어 정상")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 2: 영상 파이프라인 조향 트레이스
# ─────────────────────────────────────────────────────────────────────────────

def test_video_pipeline(
    dpu_runner,
    actuator,
    cfg: dict,
    video_path: str = VIDEO_PATH,
    max_frames: int = 9999,
    steer_only: bool = False,
    stream_server: MJPEGServer | None = None,
) -> None:
    print("\n" + "=" * 60)
    print(f"[테스트 2] 영상 파이프라인 트레이스: {video_path}")
    if steer_only:
        print("  [조향 전용 모드] 뒷바퀴 speed_cmd=0 강제 적용")
    if stream_server is not None:
        print("  [스트리밍 ON] 브라우저에서 확인하세요")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 영상 파일을 열 수 없습니다: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n영상 정보: {frame_w}x{frame_h}  {fps:.1f}fps  총 {total_frames}프레임")
    n_frames = min(max_frames, total_frames)
    print(f"처리 프레임 수: {n_frames}\n")

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

    header = (
        f"{'프레임':>6} | {'s_error':>8} | {'s_cmd':>7} | "
        f"{'effort':>7} | {'duty':>6} | {'채널':<14} | 판정"
    )
    print(header)
    print("-" * len(header))

    stats  = {"total": 0, "lane_ok": 0, "steer_nonzero_error": 0, "steer_activated": 0}
    issues = []

    for frame_id in range(n_frames):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        input_image, pre_meta = preprocess_frame(frame_bgr, camera_cfg, model_cfg)
        raw_output = dpu_runner.run(input_image)
        post = postprocess_output(raw_output, pre_meta, model_cfg, control_cfg)

        s_error = post["steering_error"]
        ctrl    = controller.compute(s_error)
        s_cmd   = ctrl["steering_cmd"]

        if steer_only:
            ctrl = dict(ctrl, speed_cmd=0.0)

        act     = actuator.apply_control(ctrl)
        effort  = act.get("steering_effort", 0.0)
        duty    = act.get("steering_duty",   0)
        channel = act.get("steering_channel") or "없음"

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
                "frame_id":        frame_id,
                "steering_error":  round(float(s_error), 4),
                "steering_cmd":    round(float(s_cmd), 4),
                "speed_cmd":       round(float(ctrl.get("speed_cmd", 0.0)), 4),
                "steering_effort": round(float(effort), 4),
                "steering_duty":   duty,
                "steering_channel": channel,
                "valid":           bool(post["valid"]),
                "reference_x":     post["reference_point"][0],
                "reference_y":     post["reference_point"][1],
                "heading_error":   round(float(post["heading_error"]), 4),
                "lane_pixels":     post.get("lane_pixels", 0),
                "mode":            act.get("mode", "unknown"),
            })

        stats["total"] += 1
        if post["valid"]:
            stats["lane_ok"] += 1
        if abs(s_error) >= 0.01:
            stats["steer_nonzero_error"] += 1
        if duty and duty > 0:
            stats["steer_activated"] += 1

        note = "OK"
        if abs(s_error) > 0.1 and duty == 0:
            note = "*** 오류: 오차있으나 duty=0 ***"
            issues.append((frame_id, s_error, s_cmd, effort, duty, channel))
        elif not post["valid"]:
            note = "차선미검출"

        print(
            f"{frame_id:>6} | {s_error:>8.4f} | {s_cmd:>7.4f} | "
            f"{effort:>7.4f} | {duty:>6} | {channel:<14} | {note}"
        )

    cap.release()
    actuator.stop()

    print("\n" + "=" * 60)
    print("[요약]")
    print(f"  처리 프레임   : {stats['total']}")
    print(f"  차선 검출 성공: {stats['lane_ok']} ({stats['lane_ok']/max(stats['total'],1)*100:.1f}%)")
    print(f"  조향오차 ≥0.01: {stats['steer_nonzero_error']}")
    print(f"  조향 duty>0   : {stats['steer_activated']}")

    print("\n[단계별 진단]")
    if stats["lane_ok"] == 0:
        print("  [X] postprocess: 차선이 한 번도 검출되지 않음 → 모델/ROI 설정 확인 필요")
    elif stats["steer_nonzero_error"] == 0:
        print("  [X] postprocess: 차선은 검출됐으나 steering_error가 항상 0")
    else:
        print(f"  [O] postprocess: {stats['steer_nonzero_error']}개 프레임에서 비제로 조향 오차 발생")

    if stats["steer_nonzero_error"] > 0 and stats["steer_activated"] == 0:
        print(f"  [X] controller/actuator: 오차가 있는데 조향이 0번 활성화됨")
        print(f"      → kp={kp} × steering_error 가 deadband({deadband}) 이하인지 확인")
        print(f"      → |s_error| > {deadband/kp:.3f} 이어야 조향 발동 (kp={kp}, deadband={deadband})")
    elif stats["steer_activated"] == 0:
        print("  [?] actuator: 조향이 한 번도 활성화되지 않음 (오차 자체가 없어서일 수 있음)")
    else:
        print(f"  [O] actuator: {stats['steer_activated']}개 프레임에서 조향 duty 활성화됨")

    if issues:
        print(f"\n  [경고] 오차 있으나 duty=0인 프레임 {len(issues)}개 (처음 5개):")
        for fid, se, sc, ef, d, ch in issues[:5]:
            print(f"    frame={fid}  s_error={se:.4f}  s_cmd={sc:.4f}  effort={ef:.4f}  duty={d}  ch={ch}")
            if abs(sc) < deadband:
                needed = deadband / kp
                print(f"      → s_cmd={sc:.4f} < deadband({deadband}). 조향 발동하려면 |s_error| > {needed:.3f} 필요")

    print()
    _dump_mmio(actuator, [steer_r, steer_l], "최종 MMIO 상태")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="조향 모터 동작 확인 테스트")
    p.add_argument("--video",      default=VIDEO_PATH, help=f"영상 경로 (기본: {VIDEO_PATH})")
    p.add_argument("--frames",     type=int, default=9999, help="처리할 최대 프레임 수")
    p.add_argument("--skip-sweep", action="store_true", help="하드웨어 스윕 테스트 건너뜀")
    p.add_argument("--skip-video", action="store_true", help="영상 파이프라인 테스트 건너뜀")
    p.add_argument("--steer-only", action="store_true", help="뒷바퀴 끄고 조향만 동작")
    p.add_argument("--stream",     action="store_true", help="브라우저 MJPEG 스트리밍 활성화")
    p.add_argument("--port",       type=int, default=5000, help="스트리밍 서버 포트 (기본: 5000)")
    p.add_argument("--mock-dpu",   action="store_true", help="DPU 없이 Mock 추론기 사용")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg  = _load_configs()

    # ── DPU 초기화 ────────────────────────────────────────────────────────────
    dpu_runner = None
    if not args.mock_dpu:
        try:
            from pynq_dpu import DpuOverlay   # type: ignore
            import xir, vart                  # type: ignore

            bit_path = cfg["default"]["actuator"].get("overlay_path", "configs/dpu/dpu.bit")
            print(f"FPGA 오버레이 로드 중: {bit_path}")
            DpuOverlay(bit_path)
            print("FPGA 오버레이 로드 완료")

            graph  = xir.Graph.deserialize(str(Path(cfg["model"]["xmodel_path"])))
            root   = graph.get_root_subgraph()
            dpu_sg = next(
                c for c in root.toposort_child_subgraph()
                if c.has_attr("device") and c.get_attr("device").upper() == "DPU"
            )
            runner = vart.Runner.create_runner(dpu_sg, "run")
            in_t   = runner.get_input_tensors()[0]
            out_t  = runner.get_output_tensors()[0]

            def _fp(t):
                return t.get_attr("fix_point") if t.has_attr("fix_point") else 0

            in_scale  = float(2 **  _fp(in_t))
            out_scale = float(2 ** -_fp(out_t))
            in_buf    = [np.empty(tuple(in_t.dims),  dtype=np.int8)]
            out_buf   = [np.empty(tuple(out_t.dims), dtype=np.int8)]

            class _RealDPU:
                def run(self, image: np.ndarray) -> np.ndarray:
                    np.clip(image * in_scale, -128, 127, out=in_buf[0][0])
                    runner.execute_async(in_buf, out_buf)
                    runner.wait(0)
                    return out_buf[0].astype(np.float32) * out_scale

            dpu_runner = _RealDPU()
            print(f"DPU 로드 완료  input={tuple(in_t.dims)}  output={tuple(out_t.dims)}")
        except Exception as exc:
            print(f"[경고] DPU 초기화 실패: {exc}")
            dpu_runner = None

    if dpu_runner is None:
        print("[Mock DPU 사용] 중앙 차선을 가정합니다.")
        dpu_runner = _MockDPU(cfg["model"])

    # ── 액추에이터 초기화 ──────────────────────────────────────────────────────
    try:
        actuator = PynqMMIOActuator(cfg["default"]["actuator"])
        actuator.initialize()
        print("PynqMMIOActuator 초기화 완료")
    except Exception as exc:
        print(f"[경고] PynqMMIOActuator 초기화 실패: {exc}")
        print("       DryRunActuator로 대체합니다.")
        actuator = DryRunActuator(cfg["default"]["actuator"])
        actuator.initialize()

    # ── 스트리밍 서버 초기화 ───────────────────────────────────────────────────
    stream_server = None
    if args.stream:
        stream_server = MJPEGServer(port=args.port, quality=60)
        stream_server.start()
        print(f"\n[스트리밍 서버 시작]")
        for url in stream_server.urls():
            print(f"  브라우저에서 접속: {url}")
        print()

    # ── 테스트 실행 ────────────────────────────────────────────────────────────
    try:
        if not args.skip_sweep:
            test_steering_sweep(actuator, cfg)

        if not args.skip_video:
            test_video_pipeline(
                dpu_runner, actuator, cfg,
                video_path=args.video,
                max_frames=args.frames,
                steer_only=args.steer_only,
                stream_server=stream_server,
            )
    finally:
        actuator.stop()
        actuator.close()
        if stream_server is not None:
            stream_server.stop()
        print("\n[테스트 종료] 모든 모터 정지 완료")


if __name__ == "__main__":
    main()
