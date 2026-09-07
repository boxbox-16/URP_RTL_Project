"""현재 ``postproc_top`` RTL과 동일한 CPU 후처리 참조 구현.

이 모듈은 성능 비교용 golden model이다. 입력은 DPU의 256x256 int8 출력이며,
RTL과 같이 0을 포함한 비음수 값을 차선으로 판정하고 5x5 morphology를 네 번
수행한 뒤 256x256 좌표계에서 조향값을 계산한다.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class RTLEquivalentPostProcessor:
    """``postproc_top``의 관측 가능한 결과를 CPU에서 재현한다."""

    def __init__(self, model_cfg: dict[str, Any], control_cfg: dict[str, Any]):
        self.height = int(model_cfg["input_height"])
        self.width = int(model_cfg["input_width"])
        self.threshold = float(model_cfg["threshold"])
        self.threshold_inclusive = bool(model_cfg.get("threshold_inclusive", False))
        kernel_size = int(control_cfg["morph_kernel_size"])
        if (self.height, self.width) != (256, 256):
            raise ValueError("현재 postproc_top은 256x256 출력만 지원합니다")
        if self.threshold != 0.0:
            raise ValueError("현재 postproc_top의 임계값은 0으로 고정되어 있습니다")
        if not self.threshold_inclusive:
            raise ValueError("현재 postproc_top은 threshold 값 자체도 차선에 포함합니다")
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

        # RTL의 lane_bit = ~pix_data[7]과 동일하게 0도 차선에 포함한다.
        np.greater_equal(arr, self.threshold, out=self.binary)
        np.copyto(self.mask, self.binary, casting="unsafe")

        # RTL은 네 pass 모두 영상 밖을 0으로 취급한다.
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
            # RTL은 위에서 아래로 검색하고 거리가 같은 경우 위쪽 행을 유지한다.
            distances = np.abs(valid_rows - self.target_row)
            ref_y = int(valid_rows[int(np.argmin(distances))])
            xs = np.flatnonzero(self.stage_b[ref_y])
            ref_x = int(xs.sum(dtype=np.int64) // xs.size)

        raw_error_q15 = 0 if ref_x is None else (ref_x - self.width // 2) << 8
        steering_error = float(raw_error_q15 / 32768.0)
        valid = hardware_valid and lane_pixels >= self.min_lane_pixels
        return {
            "steering_error": steering_error,
            "heading_error": 0.0,
            "lane_pixels": lane_pixels,
            "valid": valid,
            "hardware_valid": hardware_valid,
            "reference_point": (ref_x, ref_y),
            "raw_error_q15": raw_error_q15,
        }
