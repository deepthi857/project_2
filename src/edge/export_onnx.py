import os

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO


def export_to_onnx(model_path, output_dir="models"):
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    print("Exporting to ONNX format...")
    onnx_path = model.export(format="onnx")

    print(f"ONNX model saved at: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    model_path = "yolov8n.pt"
    export_to_onnx(model_path)
    print("ONNX export completed.")
