"""
board_pwm_server.py — Ultra96V2 보드에서 실행
==============================================
USB 시리얼로 PC의 명령 수신 → MMIO PWM 레지스터 쓰기

PC에서 보내는 프로토콜: "S{steer:+.3f}V{speed:.3f}\n"
예: "S+0.250V0.220\n"  "S+0.000V0.000\n"

보드에서 실행:
  python3 board_pwm_server.py
  python3 board_pwm_server.py --port /dev/ttyUSB1
"""

import argparse
import sys
import time
import serial
import threading

# ── PYNQ MMIO ────────────────────────────────────────────────────────────────
try:
    from pynq import MMIO, Overlay
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("[경고] pynq 없음 - 드라이런 모드")

# PWM 레지스터 오프셋
REG_PERIOD = 0x00
REG_DUTY   = 0x04
REG_VALID  = 0x08

# default.yaml 기준 MMIO 주소 맵
MOTOR_ADDRS = {
    "rear_right_fwd": 0xA0000000,
    "rear_right_bwd": 0xA0010000,
    "rear_left_fwd":  0xA0030000,
    "rear_left_bwd":  0xA0020000,
    "steering_right": 0xA0040000,
    "steering_left":  0xA0050000,
}

# default.yaml 기준 파라미터
PERIOD              = 200000     # 500Hz @ 100MHz (HW_setting_test_bit 기준)
DRIVE_DUTY_PCT      = 1.0
DRIVE_MIN_PCT       = 0.6
STEERING_DUTY_PCT   = 1.0
STEERING_MIN_PCT    = 0.70
STEERING_CENTER_PCT = 0.50
STEERING_DEADBAND   = 0.05
MMIO_RANGE          = 65536
TIMEOUT_SEC         = 1.0        # 이 시간 내 명령 없으면 정지


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PWMController:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.mmios: dict = {}
        self._last_cmd_time = time.time()
        self._watchdog_thread = None
        self._running = True

        if not dry_run and PYNQ_AVAILABLE:
            # 비트스트림 로드
            try:
                print("[FPGA] 비트스트림 로드 중...")
                Overlay("/home/xilinx/jupyter_notebooks/pynq-dpu/Motor/HW_setting_test_bit/HW_setting_test.bit")
                print("[FPGA] 로드 완료")
            except Exception as e:
                print(f"[경고] 비트스트림 로드 실패: {e}")

            # MMIO 초기화
            for name, addr in MOTOR_ADDRS.items():
                try:
                    m = MMIO(addr, MMIO_RANGE)
                    m.write(REG_PERIOD, PERIOD)
                    m.write(REG_DUTY,   PERIOD)
                    m.write(REG_VALID,  0)
                    self.mmios[name] = m
                except Exception as e:
                    print(f"[경고] {name} MMIO 초기화 실패: {e}")

            print(f"[PWM] 초기화 완료: {list(self.mmios.keys())}")
        else:
            print("[드라이런] MMIO 없이 시뮬레이션")

    def _write_duty(self, name, value):
        if name in self.mmios:
            self.mmios[name].write(REG_DUTY, int(value))

    def _write_valid(self, name, enable: bool):
        if name in self.mmios:
            if enable:
                self.mmios[name].write(REG_PERIOD, PERIOD)
            self.mmios[name].write(REG_VALID, 1 if enable else 0)

    def apply(self, steer: float, speed: float):
        self._last_cmd_time = time.time()
        steer = _clamp(steer, -1.0, 1.0)
        speed = _clamp(speed,  0.0, 1.0)

        if self.dry_run:
            print(f"  [드라이런] steer={steer:+.3f} speed={speed:.3f}")
            return

        # ── 조향: 비례 duty (최소 STEERING_MIN_PCT ~ 최대 STEERING_DUTY_PCT) ──
        abs_steer = abs(steer)
        if abs_steer > STEERING_DEADBAND:
            # 오차 크기에 비례해 duty 증가
            t = _clamp((abs_steer - STEERING_DEADBAND) / (1.0 - STEERING_DEADBAND), 0.0, 1.0)
            steer_duty = int(PERIOD * (STEERING_MIN_PCT + (STEERING_DUTY_PCT - STEERING_MIN_PCT) * t))
            if steer > 0:
                self._write_duty("steering_right", steer_duty)
                self._write_valid("steering_right", True)
                self._write_valid("steering_left",  False)
            else:
                self._write_duty("steering_left", steer_duty)
                self._write_valid("steering_left",  True)
                self._write_valid("steering_right", False)
        else:
            self._write_valid("steering_right", False)
            self._write_valid("steering_left",  False)

        # ── 구동: 급회전 시 속도 감소 ─────────────────────────────────────
        # 조향이 클수록 속도를 줄여 차선 이탈 방지
        speed_scale = max(0.4, 1.0 - abs_steer * 0.8)
        effective_speed = DRIVE_MIN_PCT + (1.0 - DRIVE_MIN_PCT) * _clamp(speed * speed_scale, 0.0, 1.0)
        drive_duty = int(PERIOD * DRIVE_DUTY_PCT * effective_speed)
        if speed >= 0.05:
            for name in ["rear_right_fwd", "rear_left_fwd"]:
                self._write_duty(name, drive_duty)
                self._write_valid(name, True)
            for name in ["rear_right_bwd", "rear_left_bwd"]:
                self._write_valid(name, False)
        else:
            for name in ["rear_right_fwd", "rear_left_fwd",
                         "rear_right_bwd", "rear_left_bwd"]:
                self._write_valid(name, False)

    def stop(self):
        if self.dry_run:
            print("  [드라이런] 정지")
            return
        for name in self.mmios:
            self.mmios[name].write(REG_VALID, 0)

    def start_watchdog(self):
        """TIMEOUT_SEC 동안 명령 없으면 자동 정지."""
        def _watch():
            while self._running:
                if time.time() - self._last_cmd_time > TIMEOUT_SEC:
                    self.stop()
                time.sleep(0.1)
        self._watchdog_thread = threading.Thread(target=_watch, daemon=True)
        self._watchdog_thread.start()

    def close(self):
        self._running = False
        self.stop()


# =============================================================================
# 시리얼 수신 루프
# =============================================================================

def parse_cmd(line: str):
    """
    "S+0.250V0.220" → (0.250, 0.220)
    파싱 실패 시 None 반환
    """
    try:
        line = line.strip()
        s_idx = line.index('S')
        v_idx = line.index('V')
        steer = float(line[s_idx+1:v_idx])
        speed = float(line[v_idx+1:])
        return steer, speed
    except Exception:
        return None


def run_server(port: str, baud: int, pwm: PWMController):
    while True:
        print(f"[시리얼] {port} @ {baud}bps 연결 시도...")
        try:
            ser = serial.Serial(port, baud, timeout=0.5)
        except Exception as e:
            print(f"[시리얼] 열기 실패: {e} — 2초 후 재시도")
            time.sleep(2)
            continue

        print(f"[시리얼] 연결 완료. 명령 수신 대기...\n")
        buf = ""

        try:
            while True:
                data = ser.read(64).decode("utf-8", errors="ignore")
                if not data:
                    continue
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    result = parse_cmd(line)
                    if result:
                        steer, speed = result
                        pwm.apply(steer, speed)
                        print(f"  S={steer:+.3f}  V={speed:.3f}")
                    elif 'S' not in line:
                        buf = ""  # 잘린 메시지 버퍼 초기화
        except KeyboardInterrupt:
            print("\n[종료]")
            pwm.stop()
            ser.close()
            return
        except Exception as e:
            print(f"[시리얼] 끊김: {e} — 재연결 시도")
            pwm.stop()
            try:
                ser.close()
            except Exception:
                pass
            buf = ""
            time.sleep(1)


# =============================================================================
# 진입점
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Ultra96V2 PWM 시리얼 서버")
    ap.add_argument("--port",     default="/dev/ttyPS0",
                    help="시리얼 포트 (기본: /dev/ttyPS0)")
    ap.add_argument("--baud",     type=int, default=115200)
    ap.add_argument("--dry-run",  action="store_true",
                    help="MMIO 쓰기 없이 시뮬레이션")
    args = ap.parse_args()

    pwm = PWMController(dry_run=args.dry_run or not PYNQ_AVAILABLE)
    pwm.start_watchdog()

    try:
        run_server(args.port, args.baud, pwm)
    finally:
        pwm.close()


if __name__ == "__main__":
    main()
