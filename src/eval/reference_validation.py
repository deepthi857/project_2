from pathlib import Path

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
    """Convert normalized YOLO coordinates to pixel xyxy coordinates."""

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
    """Read a YOLO label file and convert boxes to pixel xyxy format."""

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


def load_validation_ground_truth():
    """Load all matched validation images and their ground-truth boxes."""

    ground_truth = []
    unmatched_labels = []

    label_files = sorted(VALIDATION_LABEL_DIR.glob("*.txt"))

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


if __name__ == "__main__":

    ground_truth, unmatched_labels = load_validation_ground_truth()

    print("\nValidation Ground-Truth Summary")
    print("-" * 50)

    print("Total label files:", len(list(VALIDATION_LABEL_DIR.glob("*.txt"))))
    print("Matched image-label pairs:", len(ground_truth))
    print("Unmatched label files:", len(unmatched_labels))

    total_boxes = sum(
        len(item["boxes"])
        for item in ground_truth
    )

    print("Total boxes in matched pairs:", total_boxes)

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