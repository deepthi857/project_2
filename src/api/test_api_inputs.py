import os
import sys

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import cv2
from fastapi.testclient import TestClient

from src.api.main import app

SAMPLE_IMAGE = "data/raw/NEU-DET/train/images/crazing/crazing_1.jpg"


def encode(image, ext):
    ok, buffer = cv2.imencode(ext, image)
    return buffer.tobytes()


def build_cases():
    image = cv2.imread(SAMPLE_IMAGE)
    gray = cv2.imread(SAMPLE_IMAGE, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Sample image not found: {SAMPLE_IMAGE}")
        sys.exit(1)

    return [
        ("JPEG image", "sample.jpg", encode(image, ".jpg"), "image/jpeg", (200,)),
        ("PNG image", "sample.png", encode(image, ".png"), "image/png", (200,)),
        ("BMP image", "sample.bmp", encode(image, ".bmp"), "image/bmp", (200,)),
        ("Grayscale JPEG", "gray.jpg", encode(gray, ".jpg"), "image/jpeg", (200,)),
        ("Tiny image 50x50", "tiny.jpg", encode(cv2.resize(image, (50, 50)), ".jpg"), "image/jpeg", (200,)),
        ("Large image 1600x1600", "large.jpg", encode(cv2.resize(image, (1600, 1600)), ".jpg"), "image/jpeg", (200,)),
        ("Text file renamed .jpg", "fake.jpg", b"this is not an image", "image/jpeg", (400,)),
        ("Empty file", "empty.jpg", b"", "image/jpeg", (400, 422)),
        ("Oversized file 11 MB", "big.jpg", os.urandom(11 * 1024 * 1024), "image/jpeg", (413,)),
    ]


def main():
    cases = build_cases()
    passed = 0
    total = 0

    with TestClient(app, raise_server_exceptions=False) as client:
        print("=== Endpoint checks ===")
        health = client.get("/health")
        total += 1
        ok = health.status_code == 200
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] GET /health -> {health.status_code} {health.json()}")

        print("\n=== /detect input handling ===")
        for name, filename, data, ctype, expected in cases:
            response = client.post("/detect", files={"file": (filename, data, ctype)})
            total += 1
            ok = response.status_code in expected
            passed += ok
            extra = ""
            if response.status_code == 200:
                extra = f" (detections: {response.json()['count']})"
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: expected {expected}, got {response.status_code}{extra}")

        response = client.post("/detect")
        total += 1
        ok = response.status_code == 422
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] No file field at all: expected (422,), got {response.status_code}")

        print("\n=== /metrics check ===")
        metrics_text = client.get("/metrics").text
        for metric in [
            "model_inference_latency_seconds",
            "images_processed_total",
            "defective_images_total",
            "invalid_requests_total",
            "service_uptime_seconds",
        ]:
            total += 1
            ok = metric in metrics_text
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] metric present: {metric}")

    print(f"\n{passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
