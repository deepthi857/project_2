import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = Path("models/best.onnx")

IMAGE_DIR = Path("data/raw/NEU-DET/validation/images")

IMAGE_SIZE = 320

WARMUP_RUNS = 5


# =========================================================
# LOAD ONNX MODEL
# =========================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nONNX model not found: {MODEL_PATH}\n"
            "Place the exported best.onnx file inside the models folder."
        )

    print("=" * 60)
    print("LOADING ONNX MODEL")
    print("=" * 60)

    print(f"Model path: {MODEL_PATH}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    print("ONNX model loaded successfully.")

    return session


# =========================================================
# FIND VALIDATION IMAGES
# =========================================================

def get_images():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation image directory not found: {IMAGE_DIR}"
        )

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    images = sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.suffix.lower() in extensions
    )

    if not images:
        raise FileNotFoundError(
            f"No images found inside: {IMAGE_DIR}"
        )

    return images


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# =========================================================
# WARM-UP
# =========================================================

def warmup(session, input_name, input_data):

    print("\nRunning warm-up...")

    for _ in range(WARMUP_RUNS):

        session.run(
            None,
            {
                input_name: input_data
            }
        )

    print("Warm-up completed.")


# =========================================================
# BENCHMARK
# =========================================================

def benchmark(session, input_name, images):

    print("\n" + "=" * 60)
    print("ONNX INFERENCE BENCHMARK")
    print("=" * 60)

    total_images = 0

    start_time = time.perf_counter()

    for image_path in images:

        input_data = preprocess(image_path)

        session.run(
            None,
            {
                input_name: input_data
            }
        )

        total_images += 1

    end_time = time.perf_counter()

    total_time = end_time - start_time

    average_time = total_time / total_images

    fps = total_images / total_time

    print("\nBenchmark Results")
    print("-" * 60)

    print(f"Images processed       : {total_images}")
    print(f"Total inference time   : {total_time:.4f} seconds")
    print(f"Average inference time : {average_time * 1000:.2f} ms")
    print(f"Throughput             : {fps:.2f} images/sec")


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n" + "=" * 60)
    print("WEEK 3 - DAY 2")
    print("ONNX INFERENCE BENCHMARK")
    print("=" * 60)

    session = load_model()

    images = get_images()

    print(f"\nValidation images found: {len(images)}")

    input_name = session.get_inputs()[0].name

    print(f"ONNX input name: {input_name}")

    first_image = preprocess(images[0])

    warmup(
        session,
        input_name,
        first_image
    )

    benchmark(
        session,
        input_name,
        images
    )

    print("\n" + "=" * 60)
    print("ONNX BENCHMARK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()