from ultralytics import YOLO


def run_experiment(name, lr0, batch):
    print("\n" + "=" * 60)
    print(f"STARTING EXPERIMENT: {name}")
    print(f"Learning Rate: {lr0}")
    print(f"Batch Size: {batch}")
    print("=" * 60)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data="src/configs/data.yaml",
        epochs=2,
        imgsz=320,
        batch=batch,
        workers=0,
        device="cpu",
        optimizer="AdamW",
        lr0=lr0,
        amp=False,
        cache=False,
        fraction=0.10,
        project="runs/tuning",
        name=name,
        exist_ok=True,
        val=True,
        save=True,
        plots=False,
        seed=0,
        verbose=True,
    )

    metrics = results.results_dict

    return {
        "name": name,
        "lr0": lr0,
        "batch": batch,
        "precision": metrics.get("metrics/precision(B)", 0.0),
        "recall": metrics.get("metrics/recall(B)", 0.0),
        "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
    }


def main():
    experiments = [
        {
            "name": "lr_low_batch2",
            "lr0": 0.0005,
            "batch": 2,
        },
        {
            "name": "lr_default_batch2",
            "lr0": 0.001,
            "batch": 2,
        },
        {
            "name": "lr_default_batch4",
            "lr0": 0.001,
            "batch": 4,
        },
    ]

    all_results = []

    for experiment in experiments:
        result = run_experiment(
            experiment["name"],
            experiment["lr0"],
            experiment["batch"],
        )

        all_results.append(result)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 70)

    print(
        f"{'Experiment':<22}"
        f"{'LR':<10}"
        f"{'Batch':<8}"
        f"{'mAP50':<10}"
        f"{'mAP50-95':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<10}"
    )

    for result in all_results:
        print(
            f"{result['name']:<22}"
            f"{result['lr0']:<10}"
            f"{result['batch']:<8}"
            f"{result['mAP50']:<10.4f}"
            f"{result['mAP50-95']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<10.4f}"
        )


if __name__ == "__main__":
    main()