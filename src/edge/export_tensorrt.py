import os

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO


def export_to_tensorrt(model_path):
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    print("Exporting to TensorRT format...")
    try:
        engine_path = model.export(format="engine")
        print(f"TensorRT engine saved at: {engine_path}")
        return engine_path
    except Exception as e:
        print(f"TensorRT export failed: {e}")
        print("This typically requires an NVIDIA GPU with TensorRT installed.")
        return None


if __name__ == "__main__":
    model_path = "yolov8n.pt"
    export_to_tensorrt(model_path)
