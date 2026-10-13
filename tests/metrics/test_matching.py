from src.data.metrics import match_predictions


def test_single_correct_detection():
    gt_boxes = [
        (0, 0.50, 0.50, 0.40, 0.40)
    ]

    predictions = [
        (0, 0.95, 0.50, 0.50, 0.40, 0.40)
    ]

    tp, fp, fn = match_predictions(
        predictions,
        gt_boxes,
        iou_threshold=0.5
    )

    assert (tp, fp, fn) == (1, 0, 0)


def test_two_detections_one_ground_truth():
    gt_boxes = [
        (0, 0.50, 0.50, 0.40, 0.40)
    ]

    predictions = [
        (0, 0.95, 0.50, 0.50, 0.40, 0.40),
        (0, 0.80, 0.50, 0.50, 0.40, 0.40)
    ]

    tp, fp, fn = match_predictions(
        predictions,
        gt_boxes,
        iou_threshold=0.5
    )

    # First detection = TP
    # Second detection = duplicate = FP
    assert (tp, fp, fn) == (1, 1, 0)


def test_wrong_class_is_false_positive():
    gt_boxes = [
        (0, 0.50, 0.50, 0.40, 0.40)
    ]

    predictions = [
        (1, 0.95, 0.50, 0.50, 0.40, 0.40)
    ]

    tp, fp, fn = match_predictions(
        predictions,
        gt_boxes,
        iou_threshold=0.5
    )

    # Box overlaps perfectly, but class is wrong.
    # Therefore prediction = FP and GT = FN.
    assert (tp, fp, fn) == (0, 1, 1)


def test_detection_below_iou_threshold_is_false_positive():
    gt_boxes = [
        (0, 0.50, 0.50, 0.40, 0.40)
    ]

    predictions = [
        (0, 0.95, 0.80, 0.80, 0.20, 0.20)
    ]

    tp, fp, fn = match_predictions(
        predictions,
        gt_boxes,
        iou_threshold=0.5
    )

    # IoU is below 0.5, so prediction = FP.
    # Ground truth remains unmatched = FN.
    assert (tp, fp, fn) == (0, 1, 1)


def test_detections_are_processed_in_confidence_order():
    gt_boxes = [
        (0, 0.50, 0.50, 0.40, 0.40)
    ]

    # Lower-confidence detection is deliberately placed first.
    predictions = [
        (0, 0.60, 0.50, 0.50, 0.40, 0.40),
        (0, 0.95, 0.50, 0.50, 0.40, 0.40)
    ]

    tp, fp, fn = match_predictions(
        predictions,
        gt_boxes,
        iou_threshold=0.5
    )

    # 0.95 detection must be processed first.
    # It gets the GT.
    # 0.60 detection becomes a duplicate FP.
    assert (tp, fp, fn) == (1, 1, 0)


def test_best_ground_truth_is_selected():
    gt_boxes = [
        (0, 0.40, 0.50, 0.20, 0.20),
        (0, 0.60, 0.50, 0.40, 0.40)
    ]

    predictions = [
        (0, 0.95, 0.60, 0.50, 0.40, 0.40)
    ]

    tp, fp, fn = match_predictions(
        predictions,
        gt_boxes,
        iou_threshold=0.5
    )

    # The prediction perfectly overlaps the second GT.
    # Therefore the best matching GT must be selected.
    assert (tp, fp, fn) == (1, 0, 1)


def test_unmatched_ground_truth_becomes_false_negative():
    gt_boxes = [
        (0, 0.30, 0.30, 0.20, 0.20),
        (0, 0.70, 0.70, 0.20, 0.20)
    ]

    predictions = [
        (0, 0.95, 0.30, 0.30, 0.20, 0.20)
    ]

    tp, fp, fn = match_predictions(
        predictions,
        gt_boxes,
        iou_threshold=0.5
    )

    # First GT matched.
    # Second GT remains unmatched.
    assert (tp, fp, fn) == (1, 0, 1)