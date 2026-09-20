from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision


CLASS_NAMES = {
    0: "Crazing",
    1: "Inclusion",
    2: "Patches",
    3: "Pitted Surface",
    4: "Rolled-in Scale",
    5: "Scratches",
}

VALIDATION_LABEL_DIR = Path("data/labels/val")
VALIDATION_IMAGE_DIR = Path("data/raw/NEU-DET/validation/images")

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200


def find_validation_image(label_file):
    """Find the validation image corresponding to a YOLO label file."""

    image_name = Path(label_file).stem + ".jpg"

    for class_dir in VALIDATION_IMAGE_DIR.iterdir():
        if class_dir.is_dir():
            image_path = class_dir / image_name

            if image_path.exists():
                return image_path

    return None


def yolo_to_xyxy(
    class_id,
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height,
):
    """Convert normalized YOLO coordinates to pixel xyxy format."""

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [
        round(x1, 2),
        round(y1, 2),
        round(x2, 2),
        round(y2, 2),
    ]


def read_yolo_labels(label_file, image_width, image_height):
    """Read a YOLO ground-truth label file."""

    boxes = []
    labels = []

    with open(label_file, "r") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Invalid label format in {label_file} "
                    f"at line {line_number}: expected 5 values, "
                    f"got {len(values)}"
                )

            class_id = int(values[0])

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class ID {class_id} in {label_file} "
                    f"at line {line_number}"
                )

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            coordinates = [
                x_center,
                y_center,
                width,
                height,
            ]

            if not all(0 <= value <= 1 for value in coordinates):
                raise ValueError(
                    f"Invalid normalized coordinates in "
                    f"{label_file} at line {line_number}: "
                    f"{coordinates}"
                )

            box = yolo_to_xyxy(
                class_id,
                x_center,
                y_center,
                width,
                height,
                image_width,
                image_height,
            )

            boxes.append(box)
            labels.append(class_id)

    return boxes, labels


def read_yolo_predictions(
    prediction_file,
    image_width,
    image_height,
):
    """Read YOLO predictions.

    Prediction format:
    class_id confidence x_center y_center width height
    """

    boxes = []
    labels = []
    scores = []

    if not prediction_file.exists():
        return boxes, labels, scores

    with open(prediction_file, "r") as file:
        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 6:
                raise ValueError(
                    f"Invalid prediction format in {prediction_file} "
                    f"at line {line_number}: expected 6 values, "
                    f"got {len(values)}"
                )

            class_id = int(values[0])
            confidence = float(values[1])

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class ID {class_id} in "
                    f"{prediction_file} at line {line_number}"
                )

            if not 0 <= confidence <= 1:
                raise ValueError(
                    f"Invalid confidence {confidence} in "
                    f"{prediction_file} at line {line_number}"
                )

            x_center = float(values[2])
            y_center = float(values[3])
            width = float(values[4])
            height = float(values[5])

            coordinates = [
                x_center,
                y_center,
                width,
                height,
            ]

            if not all(0 <= value <= 1 for value in coordinates):
                raise ValueError(
                    f"Invalid normalized coordinates in "
                    f"{prediction_file} at line {line_number}: "
                    f"{coordinates}"
                )

            box = yolo_to_xyxy(
                class_id,
                x_center,
                y_center,
                width,
                height,
                image_width,
                image_height,
            )

            boxes.append(box)
            labels.append(class_id)
            scores.append(confidence)

    return boxes, labels, scores


def load_validation_ground_truth():
    """Load all matched validation images and their ground-truth boxes."""

    ground_truth = []
    unmatched_labels = []

    label_files = sorted(
        VALIDATION_LABEL_DIR.glob("*.txt")
    )

    for label_file in label_files:

        image_path = find_validation_image(label_file)

        if image_path is None:
            unmatched_labels.append(label_file)
            continue

        boxes, labels = read_yolo_labels(
            label_file,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        )

        ground_truth.append(
            {
                "image": image_path,
                "boxes": boxes,
                "labels": labels,
            }
        )

    return ground_truth, unmatched_labels


def load_predictions(prediction_dir, ground_truth):
    """Load prediction files corresponding to validation images."""

    prediction_dir = Path(prediction_dir)

    predictions = []
    missing_predictions = []

    for item in ground_truth:

        prediction_file = (
            prediction_dir / f"{item['image'].stem}.txt"
        )

        if not prediction_file.exists():
            missing_predictions.append(prediction_file)

            predictions.append(
                {
                    "image": item["image"],
                    "boxes": [],
                    "labels": [],
                    "scores": [],
                }
            )

            continue

        boxes, labels, scores = read_yolo_predictions(
            prediction_file,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        )

        predictions.append(
            {
                "image": item["image"],
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
            }
        )

    return predictions, missing_predictions


def create_torchmetrics_target(item):
    """Convert one ground-truth item to TorchMetrics format."""

    return {
        "boxes": torch.tensor(
            item["boxes"],
            dtype=torch.float32,
        ),
        "labels": torch.tensor(
            item["labels"],
            dtype=torch.int64,
        ),
    }


def create_torchmetrics_prediction(
    boxes,
    labels,
    scores,
):
    """Convert predictions to TorchMetrics format."""

    return {
        "boxes": torch.tensor(
            boxes,
            dtype=torch.float32,
        ),
        "scores": torch.tensor(
            scores,
            dtype=torch.float32,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.int64,
        ),
    }


def evaluate_with_torchmetrics(
    ground_truth,
    predictions,
):
    """Evaluate predictions using TorchMetrics."""

    if len(ground_truth) != len(predictions):
        raise ValueError(
            "Ground-truth and prediction counts do not match: "
            f"{len(ground_truth)} vs {len(predictions)}"
        )

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
    )

    for gt_item, prediction in zip(
        ground_truth,
        predictions,
    ):

        target = create_torchmetrics_target(
            gt_item
        )

        pred = create_torchmetrics_prediction(
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
        )

        metric.update(
            [pred],
            [target],
        )

    return metric.compute()


def print_validation_summary(
    ground_truth,
    unmatched_labels,
):
    """Print validation dataset statistics."""

    print("\nValidation Ground-Truth Summary")
    print("-" * 50)

    total_label_files = len(
        list(VALIDATION_LABEL_DIR.glob("*.txt"))
    )

    print(
        "Total label files:",
        total_label_files,
    )

    print(
        "Matched image-label pairs:",
        len(ground_truth),
    )

    print(
        "Unmatched label files:",
        len(unmatched_labels),
    )

    total_boxes = sum(
        len(item["boxes"])
        for item in ground_truth
    )

    print(
        "Total boxes in matched pairs:",
        total_boxes,
    )

    if unmatched_labels:

        print("\nUnmatched labels:")

        for label_file in unmatched_labels:
            print(" -", label_file)

    print("\nBoxes by class:")
    print("-" * 50)

    class_box_counts = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    for item in ground_truth:

        for class_id in item["labels"]:
            class_box_counts[class_id] += 1

    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"{class_name:<20}"
            f"{class_box_counts[class_id]}"
        )


if __name__ == "__main__":

    ground_truth, unmatched_labels = (
        load_validation_ground_truth()
    )

    print_validation_summary(
        ground_truth,
        unmatched_labels,
    )

    print(
        "\nTorchMetrics reference evaluator "
        "loaded successfully."
    )