#!/usr/bin/env python3
"""같은 영상에서 RTL-equivalent CPU 후처리와 실제 RTL 후처리를 비교한다.

먼저 auto_drive 설정의 비트스트림으로 소프트웨어 파이프라인을 측정하고 자원을
해제한다. 그 다음 auto_drive_RTL 설정의 비트스트림을 로드해 RTL 파이프라인을
측정한다. 두 단계는 같은 영상을 처음부터 다시 읽고 프레임별 전처리/DPU 출력
해시를 대조한다. 영상 디코딩, 제어기 계산, 파일 출력은 단계별 시간에서 제외된다.

기본 사용법:
    python3 compare_postprocessing.py
    python3 compare_postprocessing.py drive.mp4
    python3 compare_postprocessing.py drive.mp4 -n 1000

인자를 생략하면 프로젝트 루트의 test_video.mp4 전체를 비교한다. 결과는
comparison_results/<실행시각>/에 CSV와 JSON으로 자동 저장된다.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 두 기존 setup.sh와 같은 스레드 조건을 Python 라이브러리 import 전에 적용한다.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "auto_drive"
RTL_DIR = ROOT / "auto_drive_RTL"
DEFAULT_VIDEO = ROOT / "test_video.mp4"
CONFIG_NAMES = ("default.yaml", "camera.yaml", "model.yaml", "control.yaml")

POSTPROC_BASE = 0x80010000
POSTPROC_RANGE = 0x1000
PP_CTRL = 0x00
PP_ADDR_LO = 0x04
PP_RESULT_ERR = 0x08
PP_RESULT_VLD = 0x0C
PP_RESULT_PX = 0x10
PP_ADDR_HI = 0x18


class RTLEquivalentPostProcessor:
    """현재 ``postproc_top`` RTL의 CPU golden model.

    비교 기준을 한 파일에 고정하기 위해 이 구현은 실주행 노트북과 분리한다.
    입력은 256x256 DPU 출력이며 RTL과 같이 0을 포함한 비음수 값을 차선으로
    판정하고 5x5 morphology 네 pass 뒤 Q1.15 조향값을 계산한다.
    """

    def __init__(self, model_cfg: dict[str, Any], control_cfg: dict[str, Any]):
        self.height = int(model_cfg["input_height"])
        self.width = int(model_cfg["input_width"])
        self.threshold = float(model_cfg["threshold"])
        self.threshold_inclusive = bool(model_cfg.get("threshold_inclusive", False))
        kernel_size = int(control_cfg["morph_kernel_size"])
        if (self.height, self.width) != (256, 256):
            raise ValueError("현재 postproc_top은 256x256 출력만 지원합니다")
        if self.threshold != 0.0 or not self.threshold_inclusive:
            raise ValueError("현재 postproc_top은 0을 포함한 비음수 값을 차선으로 판정합니다")
        if kernel_size != 5:
            raise ValueError("현재 postproc_top의 morphology kernel은 5x5로 고정되어 있습니다")

        self.target_row = int(self.height * float(control_cfg["reference_row_ratio"]))
        self.min_lane_pixels = int(control_cfg.get("min_lane_pixels", 1))
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        self.binary = np.empty((self.height, self.width), dtype=np.bool_)
        self.mask = np.empty((self.height, self.width), dtype=np.uint8)
        self.stage_a = np.empty_like(self.mask)
        self.stage_b = np.empty_like(self.mask)

    @staticmethod
    def _as_2d(raw_output: np.ndarray) -> np.ndarray:
        arr = raw_output
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise RuntimeError(f"지원하지 않는 DPU output shape: {raw_output.shape}")
        return arr

    def run(self, raw_output: np.ndarray) -> dict[str, Any]:
        arr = self._as_2d(raw_output)
        if arr.shape != (self.height, self.width):
            raise RuntimeError(
                f"DPU output 크기 불일치: {arr.shape} != {(self.height, self.width)}"
            )

        np.greater_equal(arr, self.threshold, out=self.binary)
        np.copyto(self.mask, self.binary, casting="unsafe")
        border = {"borderType": cv2.BORDER_CONSTANT, "borderValue": 0}
        cv2.erode(self.mask, self.kernel, dst=self.stage_a, **border)
        cv2.dilate(self.stage_a, self.kernel, dst=self.stage_b, **border)
        cv2.dilate(self.stage_b, self.kernel, dst=self.stage_a, **border)
        cv2.erode(self.stage_a, self.kernel, dst=self.stage_b, **border)

        lane_pixels = min(int(np.count_nonzero(self.stage_b)), 0xFFFF)
        row_counts = np.count_nonzero(self.stage_b, axis=1)
        valid_rows = np.flatnonzero(row_counts)
        hardware_valid = bool(valid_rows.size)
        ref_x: int | None = None
        ref_y: int | None = None
        if hardware_valid:
            distances = np.abs(valid_rows - self.target_row)
            ref_y = int(valid_rows[int(np.argmin(distances))])
            xs = np.flatnonzero(self.stage_b[ref_y])
            ref_x = int(xs.sum(dtype=np.int64) // xs.size)

        raw_error_q15 = 0 if ref_x is None else (ref_x - self.width // 2) << 8
        valid = hardware_valid and lane_pixels >= self.min_lane_pixels
        return {
            "steering_error": float(raw_error_q15 / 32768.0),
            "heading_error": 0.0,
            "lane_pixels": lane_pixels,
            "valid": valid,
            "hardware_valid": hardware_valid,
            "reference_point": (ref_x, ref_y),
            "raw_error_q15": raw_error_q15,
        }


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int) -> float:
    return (now_ns() - start_ns) / 1_000_000.0


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML 최상위 값이 dict가 아닙니다: {path}")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_common_inputs() -> dict[str, str]:
    """후처리 외 조건에 쓰이는 파일이 두 디렉터리에서 같은지 강제한다."""
    checked: dict[str, str] = {}
    paths = [
        *(Path("configs") / name for name in CONFIG_NAMES),
        Path("models/lane_segmentation.xmodel"),
        Path("setup.sh"),
    ]
    mismatches: list[str] = []
    for relative in paths:
        base_path = BASE_DIR / relative
        rtl_path = RTL_DIR / relative
        if not base_path.is_file() or not rtl_path.is_file():
            mismatches.append(f"누락: {base_path} 또는 {rtl_path}")
            continue
        base_hash = sha256(base_path)
        rtl_hash = sha256(rtl_path)
        checked[str(relative)] = base_hash
        if base_hash != rtl_hash:
            mismatches.append(f"내용 불일치: {relative}")
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            "후처리를 제외한 공통 입력 파일이 동일하지 않습니다. 비교를 중단합니다."
            f"\n  - {details}"
        )
    return checked


def load_config(pipeline_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "default": load_yaml(pipeline_dir / "configs/default.yaml"),
        "camera": load_yaml(pipeline_dir / "configs/camera.yaml"),
        "model": load_yaml(pipeline_dir / "configs/model.yaml"),
        "control": load_yaml(pipeline_dir / "configs/control.yaml"),
    }


class Preprocessor:
    """두 비교 경로가 각각 사용하는 동일 구현의 사전 할당 전처리기."""

    def __init__(self, frame_shape: tuple[int, ...], camera_cfg: dict, model_cfg: dict):
        frame_h, frame_w = frame_shape[:2]
        top = max(0, min(int(frame_h * camera_cfg["roi_top_ratio"]), frame_h - 1))
        bottom = max(top + 1, min(int(frame_h * camera_cfg["roi_bottom_ratio"]), frame_h))
        left = max(0, min(int(frame_w * camera_cfg["roi_left_ratio"]), frame_w - 1))
        right = max(left + 1, min(int(frame_w * camera_cfg["roi_right_ratio"]), frame_w))
        self.roi = (top, bottom, left, right)
        self.roi_h = bottom - top
        self.roi_w = right - left
        self.input_h = int(model_cfg["input_height"])
        self.input_w = int(model_cfg["input_width"])
        self.channel_order = str(model_cfg.get("channel_order", "RGB")).upper()
        self.resized = np.empty((self.input_h, self.input_w, 3), dtype=np.uint8)
        self.converted = np.empty_like(self.resized)
        self.normalized = np.empty((self.input_h, self.input_w, 3), dtype=np.float32)
        self.scale = np.float32(1.0 / 255.0 if model_cfg.get("normalize", True) else 1.0)
        self.mean = np.asarray(model_cfg.get("mean") or [0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.asarray(model_cfg.get("std") or [1.0, 1.0, 1.0], dtype=np.float32)
        self.need_mean = bool(np.any(self.mean != 0.0))
        self.need_std = bool(np.any(self.std != 1.0))
        self.meta = {"roi_h": self.roi_h, "roi_w": self.roi_w}

    def run(self, frame_bgr: np.ndarray) -> np.ndarray:
        top, bottom, left, right = self.roi
        roi_bgr = frame_bgr[top:bottom, left:right]
        cv2.resize(
            roi_bgr,
            (self.input_w, self.input_h),
            dst=self.resized,
            interpolation=cv2.INTER_LINEAR,
        )
        if self.channel_order == "RGB":
            cv2.cvtColor(self.resized, cv2.COLOR_BGR2RGB, dst=self.converted)
        else:
            np.copyto(self.converted, self.resized)
        np.multiply(self.converted, self.scale, out=self.normalized)
        if self.need_mean:
            np.subtract(self.normalized, self.mean, out=self.normalized)
        if self.need_std:
            np.divide(self.normalized, self.std, out=self.normalized)
        return self.normalized


class DPURunner:
    """각 오버레이 단계에서 생성하는 VART runner와 공통 양/역양자화 구현."""

    def __init__(self, model_path: Path):
        import xir  # type: ignore
        import vart  # type: ignore

        graph = xir.Graph.deserialize(str(model_path))
        root = graph.get_root_subgraph()
        dpu_subgraphs = [
            child
            for child in root.toposort_child_subgraph()
            if child.has_attr("device") and child.get_attr("device").upper() == "DPU"
        ]
        if len(dpu_subgraphs) != 1:
            raise RuntimeError(f"DPU subgraph가 정확히 1개여야 합니다: {len(dpu_subgraphs)}개")
        self.graph = graph
        self.runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
        self.input_tensor = self.runner.get_input_tensors()[0]
        self.output_tensor = self.runner.get_output_tensors()[0]
        self.input_shape = tuple(self.input_tensor.dims)
        self.output_shape = tuple(self.output_tensor.dims)
        self.in_f32 = np.empty(self.input_shape, dtype=np.float32)
        self.in_buf = np.empty(self.input_shape, dtype=np.int8)
        self.out_buf = np.empty(self.output_shape, dtype=np.int8)
        self.out_f32 = np.empty(self.output_shape, dtype=np.float32)
        in_fp = self.input_tensor.get_attr("fix_point") if self.input_tensor.has_attr("fix_point") else 0
        out_fp = self.output_tensor.get_attr("fix_point") if self.output_tensor.has_attr("fix_point") else 0
        self.in_scale = float(2**in_fp)
        self.out_scale = float(2 ** (-out_fp))

    def run(self, image: np.ndarray) -> np.ndarray:
        np.multiply(image, self.in_scale, out=self.in_f32[0])
        np.clip(self.in_f32[0], -128, 127, out=self.in_f32[0])
        np.copyto(self.in_buf[0], self.in_f32[0], casting="unsafe")
        job_id = self.runner.execute_async([self.in_buf], [self.out_buf])
        self.runner.wait(job_id)
        np.multiply(self.out_buf, self.out_scale, out=self.out_f32)
        return self.out_f32

    def close(self) -> None:
        self.runner = None
        self.input_tensor = None
        self.output_tensor = None
        self.graph = None


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def resolve_overlay_path(pipeline_dir: Path, config: dict[str, dict[str, Any]]) -> Path:
    configured = config["default"].get("actuator", {}).get("overlay_path")
    if not configured:
        raise KeyError(f"{pipeline_dir}/configs/default.yaml의 actuator.overlay_path가 필요합니다")
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = pipeline_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"설정된 비트스트림이 없습니다: {path}")
    return path


class LegacyCPUPostProcessor:
    """이전 auto_drive 후처리. 회귀 분석용이며 성능 비교에는 사용하지 않는다."""

    def __init__(self, model_cfg: dict, control_cfg: dict, meta: dict):
        self.threshold = float(model_cfg["threshold"])
        self.control = control_cfg
        self.roi_h = int(meta["roi_h"])
        self.roi_w = int(meta["roi_w"])
        kernel_size = int(control_cfg["morph_kernel_size"])
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def run(self, raw_output: np.ndarray) -> dict[str, Any]:
        arr = raw_output
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise RuntimeError(f"지원하지 않는 DPU output shape: {raw_output.shape}")

        mask = ((arr > self.threshold).astype(np.uint8)) * 255
        filtered = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, self.kernel)
        min_area = int(self.control.get("min_component_area", 0))
        if min_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered, connectivity=8)
            if num_labels > 1:
                keep = np.zeros(num_labels, dtype=np.uint8)
                keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_area).astype(np.uint8)
                filtered = (keep[labels] * 255).astype(np.uint8)

        roi_mask = cv2.resize(
            filtered, (self.roi_w, self.roi_h), interpolation=cv2.INTER_NEAREST
        )
        row_step = max(1, int(self.control.get("centerline_row_step", 5)))
        min_pix = max(1, int(self.control.get("min_pixels_per_sample_row", 1)))
        ys = np.arange(self.roi_h - 1, -1, -row_step)
        rows = roi_mask[ys] > 0
        row_sums = rows.sum(axis=1)
        valid_rows = row_sums >= min_pix
        points: list[tuple[int, int]] = []
        if valid_rows.any():
            xs = np.arange(self.roi_w, dtype=np.int32)
            x_sums = (rows * xs).sum(axis=1)
            center_xs = np.where(valid_rows, x_sums // row_sums.clip(1), 0)
            points = [
                (int(center_xs[index]), int(ys[index]))
                for index in range(len(ys))
                if valid_rows[index]
            ]
        target_y = int(self.roi_h * float(self.control["reference_row_ratio"]))
        ref_point = min(points, key=lambda point: abs(point[1] - target_y)) if points else (None, None)
        ref_x = ref_point[0]
        steering_error = (
            float((ref_x - self.roi_w / 2.0) / (self.roi_w / 2.0))
            if ref_x is not None
            else 0.0
        )
        lane_pixels = int(np.count_nonzero(roi_mask))
        valid = (
            ref_x is not None
            and lane_pixels >= int(self.control.get("min_lane_pixels", 1))
            and len(points) >= int(self.control.get("min_centerline_points", 1))
        )
        return {
            "steering_error": steering_error,
            "lane_pixels": lane_pixels,
            "valid": bool(valid),
        }


class RTLPostProcessor:
    def __init__(self, dpu: DPURunner, base_addr: int, timeout_ms: float, control_cfg: dict):
        from pynq import MMIO, allocate  # type: ignore

        self.dpu = dpu
        self.timeout_ms = float(timeout_ms)
        self.min_lane_pixels = int(control_cfg.get("min_lane_pixels", 1))
        self.mmio = MMIO(int(base_addr), POSTPROC_RANGE)
        self.out_cma = allocate(shape=dpu.output_shape, dtype=np.int8)

    def run(self, _raw_output: np.ndarray) -> dict[str, Any]:
        copy_start = now_ns()
        np.copyto(self.out_cma, self.dpu.out_buf)
        self.out_cma.flush()
        copy_ms = elapsed_ms(copy_start)

        physical = int(self.out_cma.physical_address)
        self.mmio.write(PP_ADDR_LO, physical & 0xFFFFFFFF)
        self.mmio.write(PP_ADDR_HI, (physical >> 32) & 0xFF)
        status = self.mmio.read(PP_CTRL)
        if (status >> 1) & 1:
            deadline = time.perf_counter() + self.timeout_ms / 1000.0
            while time.perf_counter() < deadline and ((self.mmio.read(PP_CTRL) >> 1) & 1):
                time.sleep(0.0002)

        rtl_start = now_ns()
        self.mmio.write(PP_CTRL, 0x1)
        deadline = time.perf_counter() + self.timeout_ms / 1000.0
        done = 0
        while time.perf_counter() < deadline:
            status = self.mmio.read(PP_CTRL)
            done = (status >> 2) & 1
            if done:
                break
            time.sleep(0.0002)
        rtl_ms = elapsed_ms(rtl_start)
        if not done:
            raise TimeoutError(f"postproc_top_0 timeout: STATUS=0x{status:08X}")

        raw_error = self.mmio.read(PP_RESULT_ERR)
        signed_error = raw_error if raw_error < 0x80000000 else raw_error - 0x100000000
        hardware_valid = self.mmio.read(PP_RESULT_VLD) & 1
        lane_pixels = self.mmio.read(PP_RESULT_PX) & 0xFFFF
        ref_x = signed_error // 256 + 128 if hardware_valid else None
        return {
            "steering_error": float(signed_error / 32768.0),
            "lane_pixels": int(lane_pixels),
            "valid": bool(hardware_valid) and lane_pixels >= self.min_lane_pixels,
            "hardware_valid": bool(hardware_valid),
            "reference_point": (ref_x, None),
            "raw_error_q15": int(signed_error),
            "copy_ms": copy_ms,
            "rtl_ms": rtl_ms,
        }

    def close(self) -> None:
        if self.out_cma is not None:
            self.out_cma.freebuffer()
            self.out_cma = None


class PDController:
    """두 기존 노트북과 같은 PD+EMA 제어기. 비교 경로별 상태는 분리한다."""

    def __init__(self, cfg: dict):
        self.kp = float(cfg["kp"])
        self.kd = float(cfg.get("kd", 0.5))
        self.deadband = float(cfg["deadband"])
        self.minimum = float(cfg["min_steering"])
        self.maximum = float(cfg["max_steering"])
        self.previous_error = 0.0
        self.previous_command = 0.0
        self.alpha = 0.6

    def compute(self, error: float, valid: bool) -> float:
        if not valid:
            return 0.0
        if abs(error) < self.deadband:
            error = 0.0
        difference = error - self.previous_error
        self.previous_error = error
        target = min(self.maximum, max(self.minimum, self.kp * error + self.kd * difference))
        command = self.alpha * target + (1.0 - self.alpha) * self.previous_command
        self.previous_command = command
        return float(command)


def detect_rtl_base(overlay: Any, requested: int | None) -> tuple[int, str]:
    if requested is not None:
        return int(requested), "command line"
    ip_dict = getattr(overlay, "ip_dict", {})
    matches = [name for name in ip_dict if name.split("/")[-1] == "postproc_top_0"]
    if len(matches) == 1:
        info = ip_dict[matches[0]]
        address = info.get("phys_addr", info.get("base_address"))
        if address is not None:
            return int(address), f"overlay.ip_dict[{matches[0]}]"
    return POSTPROC_BASE, "notebook fallback"


def run_path(
    frame: np.ndarray,
    preprocessor: Preprocessor,
    dpu: DPURunner,
    postprocessor: Any,
) -> tuple[dict[str, Any], dict[str, float]]:
    total_start = now_ns()
    stage_start = now_ns()
    image = preprocessor.run(frame)
    preprocess_ms = elapsed_ms(stage_start)
    stage_start = now_ns()
    raw_output = dpu.run(image)
    inference_ms = elapsed_ms(stage_start)
    stage_start = now_ns()
    result = postprocessor.run(raw_output)
    postprocess_ms = elapsed_ms(stage_start)
    timings = {
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "pipeline_ms": elapsed_ms(total_start),
    }
    return result, timings


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {key: math.nan for key in ("mean", "median", "p95", "std", "min", "max")}
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "std": statistics.pstdev(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def relative_difference_pct(a: float, b: float) -> float:
    denominator = (abs(a) + abs(b)) / 2.0
    return abs(a - b) / denominator * 100.0 if denominator else 0.0


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    return float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])


def make_summary(
    rows: list[dict[str, Any]],
    steering_tolerance: float,
    timing_tolerance_pct: float,
) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    for stage in ("preprocess_ms", "inference_ms", "postprocess_ms", "pipeline_ms"):
        base_stats = describe([float(row[f"baseline_{stage}"]) for row in rows])
        rtl_stats = describe([float(row[f"rtl_{stage}"]) for row in rows])
        timing[stage] = {
            "baseline": base_stats,
            "rtl": rtl_stats,
            "mean_relative_difference_pct": relative_difference_pct(base_stats["mean"], rtl_stats["mean"]),
            "speedup_baseline_over_rtl": base_stats["mean"] / rtl_stats["mean"] if rtl_stats["mean"] else None,
        }

    base_errors = [float(row["baseline_steering_error"]) for row in rows]
    rtl_errors = [float(row["rtl_steering_error"]) for row in rows]
    error_diffs = [abs(a - b) for a, b in zip(base_errors, rtl_errors)]
    base_commands = [float(row["baseline_steering_cmd"]) for row in rows]
    rtl_commands = [float(row["rtl_steering_cmd"]) for row in rows]
    command_diffs = [abs(a - b) for a, b in zip(base_commands, rtl_commands)]
    valid_agreements = [row["baseline_valid"] == row["rtl_valid"] for row in rows]
    lane_pixel_diffs = [
        abs(int(row["baseline_lane_pixels"]) - int(row["rtl_lane_pixels"])) for row in rows
    ]
    within = [difference <= steering_tolerance for difference in error_diffs]
    direction_agreements = [
        (abs(a) <= steering_tolerance and abs(b) <= steering_tolerance) or (a * b > 0)
        for a, b in zip(base_errors, rtl_errors)
    ]
    post_speedup = timing["postprocess_ms"]["speedup_baseline_over_rtl"]
    error_p95 = describe(error_diffs)["p95"]
    checks = {
        "preprocess_timing_similar": timing["preprocess_ms"]["mean_relative_difference_pct"] <= timing_tolerance_pct,
        "inference_timing_similar": timing["inference_ms"]["mean_relative_difference_pct"] <= timing_tolerance_pct,
        "rtl_postprocess_faster": post_speedup is not None and post_speedup > 1.0,
        "steering_mae_within_tolerance": statistics.fmean(error_diffs) <= steering_tolerance,
        "steering_p95_within_tolerance": error_p95 <= steering_tolerance,
        "lane_pixels_exact": not any(lane_pixel_diffs),
        "valid_exact": all(valid_agreements),
    }
    return {
        "frames": len(rows),
        "timing_ms": timing,
        "steering": {
            "tolerance": steering_tolerance,
            "error_mae": statistics.fmean(error_diffs),
            "error_rmse": math.sqrt(statistics.fmean([value * value for value in error_diffs])),
            "error_max_abs": max(error_diffs),
            "error_p95_abs": error_p95,
            "error_correlation": correlation(base_errors, rtl_errors),
            "error_within_tolerance_ratio": statistics.fmean(within),
            "direction_agreement_ratio": statistics.fmean(direction_agreements),
            "command_mae": statistics.fmean(command_diffs),
            "command_max_abs": max(command_diffs),
            "valid_agreement_ratio": statistics.fmean(valid_agreements),
            "lane_pixels_mae": statistics.fmean(lane_pixel_diffs),
            "lane_pixels_max_abs": max(lane_pixel_diffs),
            "lane_pixels_exact_ratio": statistics.fmean(
                [difference == 0 for difference in lane_pixel_diffs]
            ),
            "baseline_valid_ratio": statistics.fmean([bool(row["baseline_valid"]) for row in rows]),
            "rtl_valid_ratio": statistics.fmean([bool(row["rtl_valid"]) for row in rows]),
        },
        "criteria": {
            "stage_mean_similarity_tolerance_pct": timing_tolerance_pct,
            **checks,
            "all_pass": all(checks.values()),
        },
    }


def write_results(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    timing = summary["timing_ms"]
    steering = summary["steering"]
    criteria = summary["criteria"]
    print("\n=== 비교 결과 ===")
    print(f"측정 프레임: {summary['frames']}")
    print("단계                 SW 기준 mean      RTL mean     차이/가속")
    for stage, label in (
        ("preprocess_ms", "전처리"),
        ("inference_ms", "DPU 추론"),
        ("postprocess_ms", "후처리"),
        ("pipeline_ms", "합계"),
    ):
        item = timing[stage]
        suffix = (
            f"{item['speedup_baseline_over_rtl']:.3f}x"
            if stage in ("postprocess_ms", "pipeline_ms")
            else f"{item['mean_relative_difference_pct']:.2f}%"
        )
        print(f"{label:<18} {item['baseline']['mean']:>10.3f} ms {item['rtl']['mean']:>10.3f} ms {suffix:>11}")
    print(
        f"조향 오차: MAE={steering['error_mae']:.6f}, "
        f"P95={steering['error_p95_abs']:.6f}, max={steering['error_max_abs']:.6f}, "
        f"허용오차 내={steering['error_within_tolerance_ratio'] * 100:.2f}%"
    )
    corr = steering["error_correlation"]
    print(
        f"조향 상관계수={'N/A' if corr is None else f'{corr:.6f}'}, "
        f"방향 일치={steering['direction_agreement_ratio'] * 100:.2f}%, "
        f"valid 일치={steering['valid_agreement_ratio'] * 100:.2f}%"
    )
    print(
        f"lane_pixels: MAE={steering['lane_pixels_mae']:.3f}, "
        f"max={steering['lane_pixels_max_abs']}, "
        f"완전 일치={steering['lane_pixels_exact_ratio'] * 100:.2f}%"
    )
    print(
        "판정: "
        f"전처리 유사={criteria['preprocess_timing_similar']}, "
        f"추론 유사={criteria['inference_timing_similar']}, "
        f"RTL 후처리 가속={criteria['rtl_postprocess_faster']}, "
        f"조향 MAE 허용범위={criteria['steering_mae_within_tolerance']}, "
        f"조향 P95 허용범위={criteria['steering_p95_within_tolerance']}, "
        f"픽셀 수 일치={criteria['lane_pixels_exact']}, "
        f"valid 일치={criteria['valid_exact']}"
    )
    print(f"결과 저장: {output_dir}")


def parse_address(value: str) -> int:
    return int(value, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="같은 주행 영상으로 RTL-equivalent SW와 실제 RTL의 속도/결과를 비교합니다.",
        epilog=(
            "예:\n"
            "  python3 compare_postprocessing.py\n"
            "  python3 compare_postprocessing.py drive.mp4\n"
            "  python3 compare_postprocessing.py drive.mp4 -n 1000\n\n"
            "인자 없이 실행하면 test_video.mp4 전체를 비교합니다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="비교할 주행 영상 (기본: test_video.mp4)",
    )
    parser.add_argument(
        "-n",
        "--frames",
        type=int,
        default=0,
        help="비교할 프레임 수 (기본: 0, 영상 끝까지)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="결과 폴더 (기본: comparison_results/<실행시각>)",
    )

    advanced = parser.add_argument_group("고급 옵션")
    advanced.add_argument("--skip-frames", type=int, default=0, help="영상 앞에서 제외할 프레임 수")
    advanced.add_argument("--warmup", type=int, default=10, help="통계에서 제외할 워밍업 반복 수")
    advanced.add_argument("--rtl-base", type=parse_address, default=None, help="RTL IP 주소(예: 0x80010000)")
    advanced.add_argument("--rtl-timeout-ms", type=float, default=50.0)
    advanced.add_argument("--steering-tolerance", type=float, default=0.0)
    advanced.add_argument("--timing-tolerance-pct", type=float, default=10.0)
    advanced.add_argument("--print-every", type=int, default=50)

    # 예전 실행 명령과의 호환성을 위한 숨김 옵션이다. 새 명령에서는 위치 인자를 쓴다.
    parser.add_argument("--video", dest="legacy_video", type=Path, help=argparse.SUPPRESS)
    return parser


def measure_pipeline(
    name: str,
    label: str,
    pipeline_dir: Path,
    bit_path: Path,
    model_path: Path,
    config: dict[str, dict[str, Any]],
    video_path: Path,
    args: argparse.Namespace,
    overlay_class: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """한 비트스트림의 파이프라인을 독립 측정한 뒤 모든 HW 자원을 해제한다."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV가 영상을 열지 못했습니다: {video_path}")
    overlay = None
    dpu = None
    postprocessor = None
    records: list[dict[str, Any]] = []
    details: dict[str, Any] = {"bitstream": str(bit_path), "bitstream_sha256": sha256(bit_path)}
    try:
        for _ in range(args.skip_frames):
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError("--skip-frames가 영상 길이보다 큽니다")
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError("영상에서 첫 측정 프레임을 읽지 못했습니다")

        print(f"\n[{label}] 오버레이 로드: {bit_path}")
        overlay = overlay_class(str(bit_path))
        dpu = DPURunner(model_path)
        expected_input = (
            1,
            int(config["model"]["input_height"]),
            int(config["model"]["input_width"]),
            int(config["model"]["input_channels"]),
        )
        expected_output = (1, expected_input[1], expected_input[2], 1)
        if dpu.input_shape != expected_input:
            raise RuntimeError(f"{label} DPU 입력 shape 불일치: {dpu.input_shape} != {expected_input}")
        if dpu.output_shape != expected_output:
            raise RuntimeError(f"{label} DPU 출력 shape 불일치: {dpu.output_shape} != {expected_output}")

        preprocessor = Preprocessor(first_frame.shape, config["camera"], config["model"])
        if name == "baseline":
            postprocessor = RTLEquivalentPostProcessor(config["model"], config["control"])
        else:
            rtl_base, source = detect_rtl_base(overlay, args.rtl_base)
            details["rtl_base_address"] = f"0x{rtl_base:08X}"
            details["rtl_base_source"] = source
            print(f"[{label}] postproc 주소: 0x{rtl_base:08X} ({source})")
            postprocessor = RTLPostProcessor(dpu, rtl_base, args.rtl_timeout_ms, config["control"])
        controller = PDController(config["control"])

        print(f"[{label}] 워밍업: {args.warmup}회 (통계 제외)")
        for _ in range(args.warmup):
            run_path(first_frame, preprocessor, dpu, postprocessor)

        frame = first_frame
        measured_index = 0
        while frame is not None and (args.frames == 0 or measured_index < args.frames):
            output, timings = run_path(frame, preprocessor, dpu, postprocessor)
            command = controller.compute(float(output["steering_error"]), bool(output["valid"]))
            ref_x, ref_y = output["reference_point"]
            record: dict[str, Any] = {
                "frame_id": args.skip_frames + measured_index,
                "preprocess_sha256": array_sha256(preprocessor.normalized),
                "dpu_int8_sha256": array_sha256(dpu.out_buf),
                "steering_error": round(float(output["steering_error"]), 9),
                "steering_cmd": round(command, 9),
                "lane_pixels": int(output["lane_pixels"]),
                "valid": bool(output["valid"]),
                "reference_x": ref_x,
                "reference_y": ref_y,
                "raw_error_q15": int(output["raw_error_q15"]),
                **{key: round(value, 6) for key, value in timings.items()},
            }
            if name == "rtl":
                record["copy_ms"] = round(float(output["copy_ms"]), 6)
                record["rtl_ms"] = round(float(output["rtl_ms"]), 6)
            records.append(record)
            measured_index += 1
            if args.print_every > 0 and (measured_index == 1 or measured_index % args.print_every == 0):
                print(
                    f"[{label}] frame {measured_index:5d}: "
                    f"pre={timings['preprocess_ms']:.3f} ms, "
                    f"DPU={timings['inference_ms']:.3f} ms, "
                    f"post={timings['postprocess_ms']:.3f} ms"
                )
            ok, next_frame = capture.read()
            frame = next_frame if ok and next_frame is not None else None

        if not records:
            raise RuntimeError(f"{label}에서 측정된 프레임이 없습니다")
        return records, details
    finally:
        capture.release()
        if isinstance(postprocessor, RTLPostProcessor):
            postprocessor.close()
        if dpu is not None:
            dpu.close()
        postprocessor = None
        dpu = None
        overlay = None
        gc.collect()
        print(f"[{label}] 자원 해제 완료")


def merge_passes(
    baseline_records: list[dict[str, Any]], rtl_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(baseline_records) != len(rtl_records):
        raise RuntimeError(
            f"두 단계의 프레임 수가 다릅니다: SW={len(baseline_records)}, RTL={len(rtl_records)}"
        )
    rows: list[dict[str, Any]] = []
    for baseline, rtl in zip(baseline_records, rtl_records):
        frame_id = baseline["frame_id"]
        if frame_id != rtl["frame_id"]:
            raise RuntimeError(f"프레임 ID 불일치: SW={frame_id}, RTL={rtl['frame_id']}")
        if baseline["preprocess_sha256"] != rtl["preprocess_sha256"]:
            raise RuntimeError(f"frame {frame_id}: 전처리 출력 불일치")
        if baseline["dpu_int8_sha256"] != rtl["dpu_int8_sha256"]:
            raise RuntimeError(f"frame {frame_id}: 두 비트스트림의 DPU int8 출력 불일치")

        row: dict[str, Any] = {
            "frame_id": frame_id,
            "execution_order": "baseline_overlay_pass->release->rtl_overlay_pass",
        }
        for prefix, record in (("baseline", baseline), ("rtl", rtl)):
            for key in (
                "preprocess_ms", "inference_ms", "postprocess_ms", "pipeline_ms",
                "steering_error", "steering_cmd", "lane_pixels", "valid",
                "reference_x", "reference_y", "raw_error_q15",
            ):
                row[f"{prefix}_{key}"] = record[key]
        row["steering_error_abs_diff"] = round(
            abs(row["baseline_steering_error"] - row["rtl_steering_error"]), 9
        )
        row["steering_cmd_abs_diff"] = round(
            abs(row["baseline_steering_cmd"] - row["rtl_steering_cmd"]), 9
        )
        row["rtl_copy_ms"] = rtl["copy_ms"]
        row["rtl_compute_ms"] = rtl["rtl_ms"]
        rows.append(row)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    if args.video is not None and args.legacy_video is not None:
        raise ValueError("영상은 위치 인자 또는 --video 중 하나로만 지정하세요")
    selected_video = args.video or args.legacy_video or DEFAULT_VIDEO
    video_path = selected_video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(
            f"영상이 없습니다: {video_path}\n"
            "영상 경로를 첫 번째 인자로 지정하세요. 예: python3 compare_postprocessing.py drive.mp4"
        )
    if args.frames < 0 or args.skip_frames < 0 or args.warmup < 0:
        raise ValueError("--frames, --skip-frames, --warmup은 0 이상이어야 합니다")
    if args.steering_tolerance < 0 or args.timing_tolerance_pct < 0:
        raise ValueError("허용 오차는 0 이상이어야 합니다")

    common_hashes = verify_common_inputs()
    baseline_config = load_config(BASE_DIR)
    rtl_config = load_config(RTL_DIR)
    baseline_bit = resolve_overlay_path(BASE_DIR, baseline_config)
    rtl_bit = resolve_overlay_path(RTL_DIR, rtl_config)

    probe = cv2.VideoCapture(str(video_path))
    if not probe.isOpened():
        raise RuntimeError(f"OpenCV가 영상을 열지 못했습니다: {video_path}")
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(probe.get(cv2.CAP_PROP_FPS))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    from pynq_dpu import DpuOverlay  # type: ignore

    print(f"공통 설정/xmodel 검증 완료: {len(common_hashes)}개 파일 동일")
    print(f"영상: {video_path} ({width}x{height}, {video_fps:.3f} fps, {total_frames} frames)")
    print(f"1단계 SW bitstream: {baseline_bit}")
    print(f"2단계 RTL bitstream: {rtl_bit}")

    baseline_records, baseline_details = measure_pipeline(
        "baseline", "1/2 SW 파이프라인", BASE_DIR, baseline_bit,
        BASE_DIR / "models/lane_segmentation.xmodel", baseline_config,
        video_path, args, DpuOverlay,
    )
    rtl_records, rtl_details = measure_pipeline(
        "rtl", "2/2 RTL 파이프라인", RTL_DIR, rtl_bit,
        RTL_DIR / "models/lane_segmentation.xmodel", rtl_config,
        video_path, args, DpuOverlay,
    )
    rows = merge_passes(baseline_records, rtl_records)
    print(f"공통 단계 값 검증: {len(rows)}개 프레임의 전처리 및 DPU int8 출력 완전 일치")

    summary = make_summary(rows, args.steering_tolerance, args.timing_tolerance_pct)
    run_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else ROOT / "comparison_results" / run_time
    )
    summary["metadata"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path),
        "video_sha256": sha256(video_path),
        "video_fps": video_fps,
        "video_total_frames": total_frames,
        "skip_frames": args.skip_frames,
        "warmup_iterations_per_overlay": args.warmup,
        "common_stage_value_check": f"all {len(rows)} frame preprocess/DPU int8 SHA-256 matched",
        "software_postprocess_spec": "RTLEquivalentPostProcessor embedded in compare_postprocessing.py",
        "execution_order": "complete baseline overlay pass, release resources, complete RTL overlay pass",
        "timed_scope": "preprocess + DPU inference + postprocess; video decode/control/hash/output excluded",
        "baseline": baseline_details,
        "rtl": rtl_details,
        "common_file_sha256": common_hashes,
        "baseline_config": baseline_config,
        "rtl_config": rtl_config,
        "python": sys.version,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    write_results(output_dir, rows, summary)
    print_summary(summary, output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n사용자 중단", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
