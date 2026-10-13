# Hyperparameter Evaluation Protocol

## Purpose

This document defines the evaluation procedure for comparing YOLOv8
hyperparameter experiments in the Real-Time Industrial Defect Detection
System.

The purpose is to ensure that different training configurations are
evaluated using a consistent validation procedure.

This protocol does not assume that any particular hyperparameter
configuration is optimal.

---

## Evaluation Workflow

Each completed training experiment should follow this workflow:

1. Train YOLOv8 using the selected hyperparameter configuration.
2. Save the trained model weights and experiment information.
3. Run the trained model on the validation dataset.
4. Collect the YOLOv8 predictions.
5. Convert the predictions using the project's YOLOv8 evaluation adapter.
6. Match the predictions with the corresponding validation images.
7. Evaluate the predictions using the project's evaluation framework.
8. Record the resulting metrics.
9. Compare the results with the baseline configuration.

The same evaluation procedure should be used for every experiment.

---

## Validation Dataset

The current validation dataset contains:

| Property | Value |
|---|---:|
| Matched validation images | 360 |
| Ground-truth bounding boxes | 854 |
| Number of classes | 6 |
| Image dimensions | 200 × 200 |

The six defect classes are:

| Class ID | Defect |
|---|---|
| 0 | Crazing |
| 1 | Inclusion |
| 2 | Patches |
| 3 | Pitted Surface |
| 4 | Rolled-in Scale |
| 5 | Scratches |

The same validation dataset should be used when comparing
hyperparameter configurations.

---

## Prediction Conversion

YOLOv8 prediction results contain:

- Bounding-box coordinates
- Class IDs
- Confidence scores

The project evaluation adapter converts YOLOv8 results into the
reference evaluation structure:

```text
boxes
labels
scores