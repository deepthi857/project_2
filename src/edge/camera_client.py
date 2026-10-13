import os
import sys
import time

import cv2
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SOURCE = os.getenv("CAMERA_SOURCE", "data/raw/NEU-DET/validation/images/crazing")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "50"))
FRAME_DELAY_SECONDS = float(os.getenv("FRAME_DELAY_SECONDS", "0.2"))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def frame_generator(source):
    if source.isdigit():
        source = int(source)
    if isinstance(source, int) or os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
    elif os.path.isdir(source):
        for name in sorted(os.listdir(source)):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                frame = cv2.imread(os.path.join(source, name))
                if frame is not None:
                    yield frame
    else:
        raise FileNotFoundError(f"Source not found: {source}")


def main():
    sent = 0
    defective = 0
    total_ms = 0.0

    for frame in frame_generator(SOURCE):
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        started = time.perf_counter()
        try:
            requests.post(f"{API_URL}/camera/heartbeat", timeout=5)
            response = requests.post(
                f"{API_URL}/detect",
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=30,
            )
        except requests.exceptions.ConnectionError:
            print(f"Cannot reach the API at {API_URL}. Start it with: uvicorn src.api.main:app")
            sys.exit(1)
        total_ms += (time.perf_counter() - started) * 1000
        sent += 1

        if response.status_code != 200:
            print(f"frame {sent:3d}: API error {response.status_code} {response.text}")
            continue

        body = response.json()
        if body["count"]:
            defective += 1
        model_ms = body["inference_time_seconds"] * 1000
        print(f"frame {sent:3d}: {body['action']:6} defects={body['count']}  model={model_ms:.0f} ms")

        if sent >= MAX_FRAMES:
            break
        time.sleep(FRAME_DELAY_SECONDS)

    print("\n=== Summary ===")
    print("Frames sent:", sent)
    print("Frames with defects:", defective)
    if sent:
        print(f"Average round trip: {total_ms / sent:.0f} ms per frame")


if __name__ == "__main__":
    main()
