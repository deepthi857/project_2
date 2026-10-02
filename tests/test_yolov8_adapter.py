import unittest

from src.eval.yolov8_adapter import (
    CLASS_NAMES,
    convert_yolov8_result,
    prediction_to_reference_format,
    validate_prediction,
)


class FakeTensor:
    """Minimal tensor-like object for testing."""

    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    """Mock YOLOv8 boxes object."""

    def __init__(self, xyxy, class_ids, confidences):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(class_ids)
        self.conf = FakeTensor(confidences)


class FakeResult:
    """Mock YOLOv8 Results object."""

    def __init__(self, xyxy, class_ids, confidences):
        self.boxes = FakeBoxes(
            xyxy,
            class_ids,
            confidences,
        )


class TestYOLOv8Adapter(unittest.TestCase):

    def test_class_mapping(self):
        """Verify that all six defect classes are defined."""

        self.assertEqual(len(CLASS_NAMES), 6)

        self.assertEqual(CLASS_NAMES[0], "Crazing")
        self.assertEqual(CLASS_NAMES[1], "Inclusion")
        self.assertEqual(CLASS_NAMES[2], "Patches")
        self.assertEqual(CLASS_NAMES[3], "Pitted Surface")
        self.assertEqual(CLASS_NAMES[4], "Rolled-in Scale")
        self.assertEqual(CLASS_NAMES[5], "Scratches")

    def test_valid_prediction(self):
        """Verify valid prediction data passes validation."""

        result = validate_prediction(
            class_id=0,
            confidence=0.95,
            box=[10, 20, 50, 60],
        )

        self.assertTrue(result)

    def test_invalid_class_id(self):
        """Verify unknown class IDs are rejected."""

        with self.assertRaises(ValueError):
            validate_prediction(
                class_id=10,
                confidence=0.95,
                box=[10, 20, 50, 60],
            )

    def test_invalid_confidence(self):
        """Verify invalid confidence values are rejected."""

        with self.assertRaises(ValueError):
            validate_prediction(
                class_id=0,
                confidence=1.5,
                box=[10, 20, 50, 60],
            )

    def test_invalid_box_length(self):
        """Verify boxes must contain four coordinates."""

        with self.assertRaises(ValueError):
            validate_prediction(
                class_id=0,
                confidence=0.95,
                box=[10, 20, 50],
            )

    def test_convert_yolov8_result(self):
        """Verify a YOLOv8-like result is converted correctly."""

        result = FakeResult(
            xyxy=[
                [10.0, 20.0, 50.0, 60.0],
                [30.0, 40.0, 80.0, 90.0],
            ],
            class_ids=[0, 5],
            confidences=[0.95, 0.80],
        )

        prediction = convert_yolov8_result(result)

        self.assertEqual(
            prediction["boxes"],
            [
                [10.0, 20.0, 50.0, 60.0],
                [30.0, 40.0, 80.0, 90.0],
            ],
        )

        self.assertEqual(
            prediction["labels"],
            [0, 5],
        )

        self.assertEqual(
            prediction["scores"],
            [0.95, 0.80],
        )

    def test_prediction_to_reference_format(self):
        """Verify converted predictions match reference format."""

        prediction = {
            "boxes": [
                [10.0, 20.0, 50.0, 60.0],
            ],
            "labels": [2],
            "scores": [0.87],
        }

        validated = prediction_to_reference_format(
            prediction
        )

        self.assertEqual(validated, prediction)

    def test_mismatched_prediction_lengths(self):
        """Verify mismatched prediction arrays are rejected."""

        prediction = {
            "boxes": [
                [10.0, 20.0, 50.0, 60.0],
            ],
            "labels": [2, 3],
            "scores": [0.87],
        }

        with self.assertRaises(ValueError):
            prediction_to_reference_format(
                prediction
            )


if __name__ == "__main__":
    unittest.main()