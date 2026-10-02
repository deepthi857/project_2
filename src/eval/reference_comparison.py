from pathlib import Path

from src.eval.reference_validation import (
    CLASS_NAMES,
    load_validation_ground_truth,
    load_predictions,
    evaluate_with_torchmetrics,
)


PREDICTION_DIR = Path("data/predictions/dummy")


def print_reference_results(results):
    """Print TorchMetrics reference evaluation results."""

    print("\nTorchMetrics Reference Results")
    print("-" * 50)

    print(
        f"mAP@0.5:       "
        f"{results['map_50'].item():.6f}"
    )

    print(
        f"mAP@0.5:0.95:  "
        f"{results['map'].item():.6f}"
    )

    print("\nPer-class AP:")
    print("-" * 50)

    class_ap = results.get("map_per_class")

    if class_ap is None:
        print("Per-class AP is not available.")
        return

    for class_id, class_name in CLASS_NAMES.items():

        if class_id < len(class_ap):
            ap = class_ap[class_id].item()

            print(
                f"{class_name:<20}"
                f"{ap:.6f}"
            )


def main():
    """Run the TorchMetrics reference evaluation."""

    print("\nDay 7: Reference Evaluation")
    print("=" * 50)

    ground_truth, unmatched_labels = (
        load_validation_ground_truth()
    )

    print(
        f"Matched validation images: "
        f"{len(ground_truth)}"
    )

    print(
        f"Unmatched labels: "
        f"{len(unmatched_labels)}"
    )

    if not PREDICTION_DIR.exists():
        print(
            f"\nPrediction directory not found:"
            f"\n{PREDICTION_DIR}"
        )
        print(
            "\nReference evaluation cannot run "
            "until prediction files are available."
        )
        return

    predictions, missing_predictions = (
        load_predictions(
            PREDICTION_DIR,
            ground_truth,
        )
    )

    print(
        f"Prediction files available: "
        f"{len(ground_truth) - len(missing_predictions)}"
        f"/{len(ground_truth)}"
    )

    if missing_predictions:

        print(
            f"\nMissing prediction files: "
            f"{len(missing_predictions)}"
        )

        for prediction_file in missing_predictions[:10]:
            print(
                " -",
                prediction_file,
            )

        if len(missing_predictions) > 10:
            print(
                f" ... and "
                f"{len(missing_predictions) - 10} more"
            )

        print(
            "\nReference evaluation was not run "
            "because predictions are incomplete."
        )

        return

    results = evaluate_with_torchmetrics(
        ground_truth,
        predictions,
    )

    print_reference_results(results)


if __name__ == "__main__":
    main()