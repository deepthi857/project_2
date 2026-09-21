import pytest

from src.data.metrics import compute_iou


def test_perfect_overlap():
    box_a = (0.5, 0.5, 1.0, 1.0)
    box_b = (0.5, 0.5, 1.0, 1.0)

    assert compute_iou(box_a, box_b) == pytest.approx(1.0)


def test_no_overlap():
    box_a = (0.2, 0.2, 0.2, 0.2)
    box_b = (0.8, 0.8, 0.2, 0.2)

    assert compute_iou(box_a, box_b) == pytest.approx(0.0)


def test_known_partial_overlap():
    # Both boxes have width=0.6 and height=0.6.
    # Their overlap is 0.45 x 0.6 = 0.27.
    #
    # Area A = 0.36
    # Area B = 0.36
    # Union = 0.36 + 0.36 - 0.27 = 0.45
    # IoU = 0.27 / 0.45 = 0.6

    box_a = (0.5, 0.5, 0.6, 0.6)
    box_b = (0.65, 0.5, 0.6, 0.6)

    assert compute_iou(box_a, box_b) == pytest.approx(0.6)


def test_edge_touching_boxes():
    # Box A right edge touches Box B left edge.
    # There is no intersection area.

    box_a = (0.25, 0.5, 0.5, 0.5)
    box_b = (0.75, 0.5, 0.5, 0.5)

    assert compute_iou(box_a, box_b) == pytest.approx(0.0)


def test_zero_area_box():
    # Width is zero, so the box has zero area.

    box_a = (0.5, 0.5, 0.0, 0.5)
    box_b = (0.5, 0.5, 1.0, 1.0)

    assert compute_iou(box_a, box_b) == pytest.approx(0.0)


def test_iou_is_symmetric():
    box_a = (0.5, 0.5, 0.6, 0.6)
    box_b = (0.65, 0.5, 0.6, 0.6)

    iou_ab = compute_iou(box_a, box_b)
    iou_ba = compute_iou(box_b, box_a)

    assert iou_ab == pytest.approx(iou_ba)


def test_negative_dimensions_raise_error():
    invalid_box = (0.5, 0.5, -0.2, 0.4)
    valid_box = (0.5, 0.5, 0.6, 0.6)

    with pytest.raises(ValueError):
        compute_iou(invalid_box, valid_box)
