import os

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

ONNX_PATH = "yolov8n.onnx"
TEST_IMAGE = "data/raw/NEU-DET/train/images/crazing/crazing_1.jpg"
INPUT_SIZE = 640
TIMED_RUNS = 20


def report_hardware():
    providers = ort.get_available_providers()
    print("=== Hardware and runtime report ===")
    print("PyTorch CUDA available:", torch.cuda.is_available())
    print("ONNX Runtime providers:", providers)
    print("CUDA provider available:", "CUDAExecutionProvider" in providers)
    print("TensorRT provider available:", "TensorrtExecutionProvider" in providers)
    if "TensorrtExecutionProvider" in providers:
        print("Result: TensorRT can be used on this machine.")
    else:
        print("Result: TensorRT needs an NVIDIA GPU with CUDA and the TensorRT libraries.")
        print("        Falling back to ONNX Runtime on CPU for edge inference.")
    return providers


def preprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)


def benchmark_onnx(onnx_path, providers):
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path} (run src/edge/export_onnx.py first)")
        return

    preferred = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") if p in providers]
    session = ort.InferenceSession(onnx_path, providers=preferred + ["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    batch = preprocess(TEST_IMAGE)

    print("\n=== ONNX Runtime test ===")
    print("Model:", onnx_path)
    print("Providers in use:", session.get_providers())
    print("Input shape:", batch.shape)

    outputs = session.run(None, {input_name: batch})
    shape = outputs[0].shape
    print("Output shape:", shape)
    print("Classes in this model:", shape[1] - 4)
    print("  (6 = defect model, 80 = generic COCO model)")

    start = time.perf_counter()
    for _ in range(TIMED_RUNS):
        session.run(None, {input_name: batch})
    avg_ms = (time.perf_counter() - start) / TIMED_RUNS * 1000
    print(f"Average inference time over {TIMED_RUNS} runs: {avg_ms:.1f} ms")
    print(f"Approximate throughput: {1000 / avg_ms:.1f} FPS")


if __name__ == "__main__":
    available = report_hardware()
    benchmark_onnx(ONNX_PATH, available)
