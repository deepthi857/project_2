import os

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO

print("Starting YOLO training...")

model = YOLO("yolov8n.pt")

results = model.train(
    data="src/configs/data.yaml",

    epochs=10,
    imgsz=320,
    batch=2,
    workers=0,

    device="cpu",

    optimizer="AdamW",
    lr0=0.001,

    amp=False,
    cache=False,

    project="runs/train",
    name="defect_detection_cpu",
    exist_ok=True,

    val=True,
    save=True,

    # IMPORTANT: don't generate label plots
    plots=False,

    seed=0,
    verbose=True
)

print()
print("================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")
print("Best model:", results.save_dir / "weights" / "best.pt")