# Week 2 Evaluation Handoff

## Purpose

This document summarizes the evaluation-side work completed during
Week 2 and defines the next steps required for model evaluation.

---

## Completed Evaluation Work

The following components are ready:

- YOLOv8 evaluation integration plan
- YOLOv8 evaluation adapter
- YOLOv8 adapter unit tests
- Hyperparameter research
- Hyperparameter experiment matrix
- Hyperparameter evaluation protocol

---

## Evaluation Components

### YOLOv8 Evaluation Adapter

File:

`src/eval/yolov8_adapter.py`

The adapter converts YOLOv8 predictions into the evaluation structure:

- `boxes`
- `labels`
- `scores`

It also validates class IDs, confidence scores, bounding boxes, and
prediction lengths.

### Adapter Tests

File:

`tests/test_yolov8_adapter.py`

The adapter tests cover class mapping, valid and invalid predictions,
YOLOv8 result conversion, reference-format validation, and mismatched
prediction lengths.

---

## Validation Dataset

The current evaluation dataset contains:

| Property | Value |
|---|---:|
| Matched validation images | 360 |
| Ground-truth bounding boxes | 854 |
| Number of classes | 6 |
| Image dimensions | 200 × 200 |

Classes:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

---

## Required Model Outputs

Final model evaluation requires:

- Trained YOLOv8 weights
- Validation predictions
- Prediction confidence scores
- Predicted class IDs
- Predicted bounding boxes

Without these outputs, final model-performance metrics cannot be
calculated.

---

## Evaluation Metrics

Once model predictions are available, record:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Per-class AP

Training time and inference speed should also be recorded when
available.

Only measured results should be recorded.

---

## Evaluation Workflow

```text
Trained YOLOv8 model
        |
        v
Validation inference
        |
        v
YOLOv8 prediction output
        |
        v
YOLOv8 evaluation adapter
        |
        v
Reference evaluation
        |
        v
mAP / Precision / Recall / Per-class AP