import os

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO


def run_experiment(name, lr0, batch, epochs=2, imgsz=320, fraction=0.1):
    print(f"\n=== Running experiment: {name} ===")
    print(f"lr0={lr0}, batch={batch}, epochs={epochs}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data="src/configs/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device="cpu",
        optimizer="AdamW",
        lr0=lr0,
        amp=False,
        cache=False,
        fraction=fraction,
        project="runs/train",
        name=name,
        exist_ok=True,
        val=True,
        save=True,
        plots=False,
        seed=0,
        verbose=False,
    )

    metrics = results.results_dict
    return {
        "name": name,
        "lr0": lr0,
        "batch": batch,
        "mAP50": metrics.get("metrics/mAP50(B)", None),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)", None),
        "precision": metrics.get("metrics/precision(B)", None),
        "recall": metrics.get("metrics/recall(B)", None),
    }


if __name__ == "__main__":
    experiments = [
        {"name": "tune_lr_low", "lr0": 0.0005, "batch": 2},
        {"name": "tune_lr_default", "lr0": 0.001, "batch": 2},
        {"name": "tune_batch_higher", "lr0": 0.001, "batch": 4},
    ]

    all_results = []
    for exp in experiments:
        result = run_experiment(exp["name"], exp["lr0"], exp["batch"])
        all_results.append(result)

    print("\n================================")
    print("HYPERPARAMETER TUNING SUMMARY")
    print("================================")
    for r in all_results:
        print(f"{r['name']}: lr0={r['lr0']}, batch={r['batch']} "
              f"-> mAP50={r['mAP50']:.4f}, mAP50-95={r['mAP50-95']:.4f}, "
              f"precision={r['precision']:.4f}, recall={r['recall']:.4f}")

    best = max(all_results, key=lambda r: r["mAP50"] if r["mAP50"] else 0)
    print(f"\nBest configuration: {best['name']} (mAP50={best['mAP50']:.4f})")