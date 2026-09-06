#!/usr/bin/env python3
"""
자율주행 디버그 뷰어 — ZMQ SUB 수신 + OpenCV 표시
사용법:
  pip install pyzmq opencv-python
  python debug_viewer.py --ip 172.20.10.6 --port 5556
"""
import argparse
import sys
import numpy as np

try:
    import zmq
except ImportError:
    sys.exit("pyzmq 없음. 설치: pip install pyzmq")

try:
    import cv2
except ImportError:
    sys.exit("opencv 없음. 설치: pip install opencv-python")


def main():
    parser = argparse.ArgumentParser(description="Autonomous driving debug viewer")
    parser.add_argument("--ip",   default="172.20.10.6", help="보드 IP 주소")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB 포트")
    args = parser.parse_args()

    addr = f"tcp://{args.ip}:{args.port}"
    print(f"[Viewer] 연결 중: {addr}")
    print("  ESC 또는 Q → 종료")
    print("  프레임이 오지 않으면 보드에서 debug_stream.enabled: true 확인")

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(addr)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 2000)   # 2초 타임아웃

    frame_count = 0
    win_name    = f"Debug Stream  [{args.ip}:{args.port}]"

    try:
        while True:
            try:
                data = sub.recv()
                arr  = np.frombuffer(data, dtype=np.uint8)
                img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    frame_count += 1
                    cv2.imshow(win_name, img)
            except zmq.Again:
                print(f"[Viewer] 대기 중... (수신 {frame_count}프레임)")

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[Viewer] 종료. 총 수신: {frame_count}프레임")
        cv2.destroyAllWindows()
        sub.close()
        ctx.term()


if __name__ == "__main__":
    main()
