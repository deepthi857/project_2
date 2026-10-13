import time
from pathlib import Path

from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = Path("models/best.pt")

IMAGE_DIR = Path("data/raw/NEU-DET/validation/images")

IMAGE_SIZE = 320

WARMUP_RUNS = 5


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel not found: {MODEL_PATH}\n"
            "Place the trained best.pt file inside the models folder."
        )

    print("=" * 60)
    print("LOADING PYTORCH MODEL")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")

    return model


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
# WARM-UP
# =========================================================

def warmup(model, image):

    print("\nRunning warm-up...")

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=str(image),
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

    print("Warm-up completed.")


# =========================================================
# BENCHMARK
# =========================================================

def benchmark(model, images):

    print("\n" + "=" * 60)
    print("PYTORCH INFERENCE BENCHMARK")
    print("=" * 60)

    total_images = 0

    start_time = time.perf_counter()

    for image in images:

        model.predict(
            source=str(image),
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
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
    print("WEEK 3 - DAY 1")
    print("PYTORCH INFERENCE BENCHMARK")
    print("=" * 60)

    model = load_model()

    images = get_images()

    print(f"\nValidation images found: {len(images)}")

    warmup(model, images[0])

    benchmark(model, images)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()