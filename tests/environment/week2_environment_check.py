import torch
import ultralytics
from ultralytics import YOLO


def main():
    print("=" * 50)
    print("WEEK 2 ENVIRONMENT VERIFICATION")
    print("=" * 50)

    print(f"PyTorch Version : {torch.__version__}")
    print(f"Ultralytics    : {ultralytics.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    print(f"CUDA Version    : {torch.version.cuda}")
    print(f"GPU Count       : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("GPU Name        : No GPU available - using CPU")
        device = "cpu"

    print(f"Training Device : {device}")

    print("\nLoading YOLOv8 model...")

    model = YOLO("yolov8n.pt")

    print("YOLOv8 model loaded successfully!")

    print("\nEnvironment verification: SUCCESS")
    print("=" * 50)


if __name__ == "__main__":
    main()