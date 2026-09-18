from ultralytics import YOLO


def train_model(data_yaml, epochs=10, img_size=640, batch_size=16):
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project="runs/train",
        name="defect_detection",
    )

    return results


if __name__ == "__main__":
    train_model(
        data_yaml="src/configs/data.yaml",
        epochs=10,
        img_size=640,
        batch_size=16,
    )

    print("Training completed.")