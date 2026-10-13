# Week 2 Day 7 — Error Analysis

## 1. Objective

Review YOLOv8 predictions on the validation dataset, identify specific prediction errors and class confusions, and determine whether the observed errors correlate with dataset characteristics.

## 2. Model and Validation Setup

- Model: YOLOv8 trained defect-detection model
- Weights:
  `runs/detect/runs/train/defect_detection/weights/best.pt`
- Validation images: 360
- Validation image size: 200 x 200
- Inference image size: 320
- Device: CPU
- Confidence threshold: 0.25
- IoU threshold for error analysis: 0.50
- Validation classes: 6

Classes:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## 3. Validation Dataset Analysis

The validation directory contains:

- 360 validation images
- 361 ground-truth label files
- 360 label files matching validation images
- 1 additional label file: `crazing_240.txt`

The additional `crazing_240.txt` file has no corresponding validation image and was retained as a dataset observation rather than deleted.

Ground-truth class distribution:

| Class | Images | Bounding Boxes |
|---|---:|---:|
| Crazing | 61 | 165 |
| Inclusion | 71 | 159 |
| Patches | 73 | 193 |
| Pitted Surface | 60 | 87 |
| Rolled-in Scale | 60 | 132 |
| Scratches | 60 | 121 |

The matched 360-image validation set therefore contains 854 ground-truth bounding boxes. Including the additional `crazing_240.txt` label gives 857 boxes in the raw validation-label directory.

## 4. Inference Results

YOLOv8 inference was performed on all 360 validation images.

Results:

- Prediction label files generated: 71
- Validation images with predictions: 71
- Validation images without prediction files: 289
- Prediction-only files: 0

The model produced predictions for only a subset of the validation images.

## 5. Error Analysis Method

Predictions were compared with ground-truth bounding boxes using IoU.

A prediction was considered spatially matched when IoU was at least 0.50.

The analysis separated:

- True positives
- False positives
- False negatives
- Class confusions

Confidence scores were retained during inference using YOLOv8's `save_conf` option.

## 6. Overall Error Analysis

At IoU >= 0.50:

| Metric | Result |
|---|---:|
| Ground-truth boxes | 857 |
| Prediction boxes | 483 |
| True Positives | 113 |
| False Positives | 367 |
| False Negatives | 741 |
| Class Confusions | 3 |
| Precision | 0.2354 |
| Recall | 0.1323 |

The results indicate that the model currently has low precision and low recall on the validation set.

## 7. Per-Class Analysis

| Class | GT Boxes | Predictions | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| Crazing | 165 | 0 | 0 | 0 | 165 |
| Inclusion | 159 | 483 | 113 | 367 | 46 |
| Patches | 193 | 0 | 0 | 0 | 192 |
| Pitted Surface | 87 | 0 | 0 | 0 | 87 |
| Rolled-in Scale | 132 | 0 | 0 | 0 | 132 |
| Scratches | 121 | 0 | 0 | 0 | 119 |

The model shows a strong prediction bias toward the Inclusion class.

Despite Inclusion having 159 ground-truth boxes, the model generated 483 Inclusion predictions. The other five classes received zero predictions in this evaluation.

Therefore, the main observed issue is not simply that Inclusion has the largest validation representation. In fact, Patches has more ground-truth boxes than Inclusion.

## 8. Confirmed Class Confusions

Three predictions were identified as spatially overlapping a ground-truth object at IoU >= 0.50 while having the wrong class.

| Image | Ground Truth | Prediction | Confidence | IoU |
|---|---|---|---:|---:|
| patches_272 | Patches | Inclusion | 0.273 | 0.651 |
| scratches_287 | Scratches | Inclusion | 0.922 | 0.549 |
| scratches_288 | Scratches | Inclusion | 0.306 | 0.510 |

The `scratches_287` example is particularly notable because the model assigned a high confidence of 0.922 to the incorrect Inclusion class.

## 9. False Positives

The error analysis identified 367 false-positive predictions.

Many of these predictions were assigned to the Inclusion class.

Representative examples included:

- `inclusion_241`
- `inclusion_242`
- `inclusion_243`
- `inclusion_244`
- `inclusion_245`
- `inclusion_246`

The confidence values varied from approximately 0.25 to higher values.

This indicates that the model is generating a large number of Inclusion detections that do not satisfy the IoU-based matching criteria.

## 10. Missed Detections

There were 741 unmatched ground-truth boxes.

The missed detections occurred across all six ground-truth classes, while the model only produced predictions for Inclusion.

The largest missed-detection counts were observed for:

- Crazing
- Patches
- Rolled-in Scale
- Scratches
- Pitted Surface

This demonstrates that the current model has difficulty detecting five of the six defect categories.

## 11. Dataset Correlation

The validation dataset contains all six classes with substantial representation.

The class counts do not explain the prediction behavior by themselves:

- Patches has the highest number of ground-truth boxes: 193.
- Crazing has 165.
- Inclusion has 159.
- Rolled-in Scale has 132.
- Scratches has 121.
- Pitted Surface has 87.

However, the model predicts only Inclusion.

Therefore, the current results suggest a model-learning or training-distribution problem rather than a simple validation-set class-count imbalance.

The additional `crazing_240.txt` label without a corresponding validation image is also a dataset consistency issue that should be reviewed.

## 12. Representative Visual Examples

Generated visualizations are stored under:

`runs/detect/day7_error_analysis/visual_examples/`

Files:

- `patches_272_error_analysis.jpg`
- `scratches_287_error_analysis.jpg`
- `scratches_288_error_analysis.jpg`

These examples correspond to the confirmed class-confusion cases identified during automated analysis.

## 13. Key Findings

1. The model generated predictions for only 71 of 360 validation images.
2. The model produced 483 predictions, of which 113 were true positives and 367 were false positives under the implemented IoU-based analysis.
3. Recall was low at 13.23%.
4. Precision was low at 23.54%.
5. The model strongly favored the Inclusion class.
6. No predictions were produced for Crazing, Patches, Pitted Surface, Rolled-in Scale, or Scratches.
7. Three class-confusion cases were identified:
   - Patches -> Inclusion
   - Scratches -> Inclusion
8. One Scratch -> Inclusion example had a high incorrect confidence of 0.922.
9. The validation dataset contains one unmatched ground-truth label file: `crazing_240.txt`.
10. The validation class distribution alone does not explain the strong Inclusion prediction bias.

## 14. Recommendations

Based on the observed errors, the next investigation should focus on:

- Reviewing training-set class distribution.
- Checking whether all six classes are correctly represented in training.
- Reviewing class labels for consistency.
- Inspecting training examples for the five classes that received no predictions.
- Checking bounding-box sizes and object visibility.
- Reviewing training loss and per-class validation metrics.
- Investigating whether the model is under-trained or biased toward one class.
- Reviewing augmentation and training configuration.
- Re-training and comparing per-class performance after corrective changes.

## 15. Conclusion

The Week 2 Day 7 error analysis identified a strong class-specific failure pattern in the YOLOv8 model. The model currently favors Inclusion predictions and fails to produce predictions for the other five defect classes on the validation set.

The analysis also identified three concrete class-confusion examples and one dataset consistency issue. These findings provide clear targets for the next stage of model and dataset investigation.
