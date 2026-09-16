# Object Detection Evaluation Metrics

## 1. IoU (Intersection over Union)

IoU measures the overlap between a predicted bounding box and the
ground-truth bounding box.

IoU = Intersection Area / Union Area

The value ranges from 0 to 1.

- IoU = 1.0 → Perfect overlap
- IoU = 0.0 → No overlap

IoU is used to determine whether a predicted bounding box matches
the actual defect location.

## 2. True Positive, False Positive and False Negative

### True Positive (TP)

A predicted defect is a true positive when the predicted class is
correct and the predicted bounding box sufficiently overlaps with
the ground-truth box according to the selected IoU threshold.

### False Positive (FP)

A false positive occurs when the model predicts a defect incorrectly,
such as detecting a defect where there is no ground-truth defect.

### False Negative (FN)

A false negative occurs when an actual defect exists but the model
fails to detect it.

## 3. Precision

Precision measures how many of the model's predicted defects are
actually correct.

Precision = TP / (TP + FP)

High precision means fewer false positive detections.

## 4. Recall

Recall measures how many of the actual defects were successfully
detected by the model.

Recall = TP / (TP + FN)

High recall means fewer actual defects are missed.

## 5. F1 Score

F1 score combines precision and recall using their harmonic mean.

F1 = 2 × Precision × Recall / (Precision + Recall)

It provides a single measure that balances precision and recall.

## 6. Average Precision (AP)

Average Precision summarizes the precision-recall performance for
one object class across different confidence thresholds.

AP is calculated from the Precision-Recall curve.

## 7. Mean Average Precision (mAP)

Mean Average Precision is the mean of the Average Precision values
across all defect classes.

Our project contains six defect classes:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## 8. mAP@0.5

mAP@0.5 calculates Average Precision using an IoU threshold of 0.50
and averages the AP values across the six defect classes.

## 9. mAP@0.5:0.95

mAP@0.5:0.95 evaluates the model at ten IoU thresholds:

0.50, 0.55, 0.60, 0.65, 0.70,
0.75, 0.80, 0.85, 0.90, 0.95

The AP values are averaged across these thresholds and classes.

This provides a more comprehensive evaluation of bounding-box
localization accuracy.
