from pathlib import Path


CLASS_NAMES = {
    0: "Crazing",
    1: "Inclusion",
    2: "Patches",
    3: "Pitted Surface",
    4: "Rolled-in Scale",
    5: "Scratches",
}


def validate_prediction(
    class_id,
    confidence,
    box,
):
    """Validate one YOLOv8 prediction."""

    if class_id not in CLASS_NAMES:
        raise ValueError(
            f"Unknown class ID: {class_id}"
        )

    if not 0 <= confidence <= 1:
        raise ValueError(
            f"Invalid confidence: {confidence}"
        )

    if len(box) != 4:
        raise ValueError(
            f"Expected four box coordinates, got {len(box)}"
        )

    return True


def convert_yolov8_result(result):
    """
    Convert one Ultralytics YOLOv8 Results object into the
    prediction structure used by the reference evaluator.

    Expected output:

    {
        "boxes": [[x1, y1, x2, y2], ...],
        "labels": [class_id, ...],
        "scores": [confidence, ...]
    }

    The function expects the YOLOv8 Results object to provide:

    result.boxes.xyxy
    result.boxes.cls
    result.boxes.conf
    """

    if not hasattr(result, "boxes"):
        raise ValueError(
            "YOLOv8 result does not contain a boxes attribute."
        )

    boxes = result.boxes

    if not hasattr(boxes, "xyxy"):
        raise ValueError(
            "YOLOv8 result boxes do not contain xyxy coordinates."
        )

    if not hasattr(boxes, "cls"):
        raise ValueError(
            "YOLOv8 result boxes do not contain class IDs."
        )

    if not hasattr(boxes, "conf"):
        raise ValueError(
            "YOLOv8 result boxes do not contain confidence scores."
        )

    xyxy = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()

    if not (
        len(xyxy)
        == len(class_ids)
        == len(confidences)
    ):
        raise ValueError(
            "YOLOv8 prediction arrays have different lengths."
        )

    predictions = {
        "boxes": [],
        "labels": [],
        "scores": [],
    }

    for box, class_id, confidence in zip(
        xyxy,
        class_ids,
        confidences,
    ):
        class_id = int(class_id)
        confidence = float(confidence)

        validate_prediction(
            class_id,
            confidence,
            box,
        )

        predictions["boxes"].append(
            [round(float(value), 2) for value in box]
        )

        predictions["labels"].append(class_id)
        predictions["scores"].append(confidence)

    return predictions


def convert_yolov8_results(results):
    """
    Convert multiple YOLOv8 Results objects.

    Returns a list containing one prediction dictionary
    for each image.
    """

    return [
        convert_yolov8_result(result)
        for result in results
    ]


def prediction_to_reference_format(prediction):
    """
    Validate that a converted YOLOv8 prediction matches the
    reference evaluator structure.
    """

    required_keys = {
        "boxes",
        "labels",
        "scores",
    }

    if set(prediction.keys()) != required_keys:
        raise ValueError(
            "Prediction must contain exactly: "
            "boxes, labels, scores"
        )

    if not (
        len(prediction["boxes"])
        == len(prediction["labels"])
        == len(prediction["scores"])
    ):
        raise ValueError(
            "Prediction boxes, labels, and scores "
            "must have the same length."
        )

    for box, label, score in zip(
        prediction["boxes"],
        prediction["labels"],
        prediction["scores"],
    ):
        validate_prediction(
            int(label),
            float(score),
            box,
        )

    return prediction


if __name__ == "__main__":
    print("YOLOv8 evaluation adapter loaded successfully.")
    print("Expected output fields: boxes, labels, scores")