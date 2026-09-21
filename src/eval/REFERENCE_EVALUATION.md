# Reference Evaluation

## Purpose

This document defines the reference evaluation procedure used to
validate the project's custom object-detection metrics.

The reference implementation uses
`torchmetrics.detection.MeanAveragePrecision`.

The reference evaluator and the custom evaluator must receive the
same prediction files and ground-truth annotations when performing
a comparison.

---

## Dataset

The validation dataset contains six metal surface defect classes:

| Class ID | Defect Class |
|---|---|
| 0 | Crazing |
| 1 | Inclusion |
| 2 | Patches |
| 3 | Pitted Surface |
| 4 | Rolled-in Scale |
| 5 | Scratches |

The validation set contains:

- 361 label files
- 360 matched image-label pairs
- 1 unmatched label file
- 854 ground-truth bounding boxes in matched pairs

The unmatched label is:

`data/labels/val/crazing_240.txt`

This label is excluded from reference evaluation because its
corresponding validation image is unavailable.

---

## Prediction Format

Predictions use the project's agreed YOLO-style format:

```text
class_id confidence x_center y_center width height