from ultralytics import YOLO


def main():
    print("========================================")
    print("YOLOv8 TRAINING")
    print("========================================")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data="src/configs/data.yaml",
        epochs=3,
        imgsz=256,
        batch=2,
        workers=0,
        device="cpu",
        optimizer="AdamW",
        lr0=0.001,
        amp=False,
        cache=False,
        fraction=0.25,
        project="runs/train",
        name="defect_detection",
        exist_ok=True,
        val=True,
        save=True,
        plots=True,
        seed=0,
        verbose=True,
    )

    print("\n========================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("========================================")

    print("Training output:", model.trainer.save_dir)
    print(
        "Best model:",
        model.trainer.save_dir / "weights" / "best.pt"
    )
    print(
        "Last model:",
        model.trainer.save_dir / "weights" / "last.pt"
    )


if __name__ == "__main__":
    main()