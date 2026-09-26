import os
import torch

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO

print("Starting YOLO training...")

model = YOLO("yolov8n.pt")


def save_checkpoint(trainer):
    os.makedirs("models", exist_ok=True)
    torch.save(trainer.model.state_dict(), "models/checkpoint.pt")
    print(f"[Checkpoint saved after epoch {trainer.epoch + 1}] -> models/checkpoint.pt")


model.add_callback("on_fit_epoch_end", save_checkpoint)

try:
    model.train(
        data="src/configs/data.yaml",
        epochs=1,
        imgsz=320,
        batch=2,
        workers=0,
        device="cpu",
        optimizer="AdamW",
        lr0=0.001,
        amp=False,
        cache=False,
        fraction=0.05,
        project="runs/train",
        name="safe_model",
        exist_ok=True,
        val=True,
        save=True,
        plots=False,
        seed=0,
        verbose=True,
    )
except Exception as e:
    print(f"Training loop raised: {e}")

print("Script reached the end.")
print("Checking models folder:", os.listdir("models") if os.path.exists("models") else "does not exist")
