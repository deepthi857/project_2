# YOLOv8 Hyperparameter Experiment Matrix

## Purpose

This document defines a controlled experiment plan for evaluating
different YOLOv8 training configurations for the Real-Time Industrial
Defect Detection System.

The purpose is to compare hyperparameter configurations using the same
validation dataset and evaluation metrics.

The experiments should be performed systematically so that the effect
of individual hyperparameters can be observed.

---

## Baseline Configuration

The baseline configuration is based on the initial YOLOv8 training
setup and the starting values documented in the hyperparameter research.

| Parameter | Baseline |
|---|---:|
| Model | YOLOv8n |
| Image size (`imgsz`) | 640 |
| Batch size | 16 |
| Learning rate (`lr0`) | 0.01 |
| Final LR factor (`lrf`) | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup epochs | 3.0 |
| Epochs | 50 |

The baseline provides the reference point for the remaining experiments.

---

## Parameters Kept Constant

Unless an experiment specifically changes a parameter, the following
settings should remain unchanged:

- Model: YOLOv8n
- Validation dataset
- Learning rate
- Final learning-rate factor
- Momentum
- Weight decay
- Warmup epochs
- Image size
- Batch size
- Number of epochs

Keeping other parameters fixed makes the comparison between experiments
more meaningful.

---

## Experiment 1 — Baseline

### Configuration

| Parameter | Value |
|---|---:|
| Model | YOLOv8n |
| Image size | 640 |
| Batch size | 16 |
| Learning rate | 0.01 |
| Epochs | 50 |

### Purpose

Establish the reference performance of the initial configuration.

### Metrics

Record:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Per-class AP
- Training time
- Inference speed, if available

---

## Experiment 2 — Batch Size

### Configurations

Compare:

| Experiment | Batch size |
|---|---:|
| Batch-16 | 16 |
| Batch-8 | 8 |
| Batch-4 | 4 |

All other parameters should remain unchanged.

### Purpose

Determine how batch size affects model performance and training
requirements.

Smaller batch sizes may be considered if memory or CPU constraints
make larger batches difficult to use.

### Metrics

For each configuration record:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Per-class AP
- Training time
- Inference speed, if available

---

## Experiment 3 — Image Size

### Configurations

Compare:

| Experiment | Image size |
|---|---:|
| Image-640 | 640 |
| Image-512 | 512 |

All other parameters should remain unchanged.

### Purpose

Determine whether a smaller input resolution can reduce computational
requirements while maintaining acceptable detection performance.

### Metrics

For each configuration record:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Per-class AP
- Training time
- Inference speed, if available

---

## Experiment 4 — Learning Rate

### Configurations

Compare:

| Experiment | Learning rate (`lr0`) |
|---|---:|
| LR-005 | 0.005 |
| LR-010 | 0.01 |
| LR-020 | 0.02 |

All other parameters should remain unchanged.

### Purpose

Study the effect of the initial learning rate on training and
validation performance.

### Metrics

For each configuration record:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Per-class AP
- Training time
- Inference speed, if available

---

## Experiment Order

The experiments should be performed in the following order:

1. Baseline
2. Batch size
3. Image size
4. Learning rate

The baseline should be established before comparing the other
configurations.

Each experiment should change the intended parameter while keeping
the remaining parameters constant.

---

## Evaluation Dataset

All experiments should use the same validation dataset for comparison.

Current validation dataset information:

| Property | Value |
|---|---:|
| Matched validation images | 360 |
| Ground-truth bounding boxes | 854 |
| Number of classes | 6 |
| Image size | 200 × 200 |

The six defect classes are:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

---

## Evaluation Metrics

The following metrics should be recorded for every experiment.

### mAP@0.5

Mean Average Precision at an IoU threshold of 0.50.

### mAP@0.5:0.95

Mean Average Precision averaged across IoU thresholds from 0.50 to
0.95.

### Precision

Measures the proportion of predicted detections that are correct.

### Recall

Measures the proportion of ground-truth objects that are detected.

### Per-Class AP

Average Precision should be recorded separately for each defect class.

This helps identify classes that are more difficult for the model to
detect.

---

## Experiment Results Table

The following table can be filled after the training experiments are
performed.

| Experiment | Parameter | Value | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Training Time |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | — | — | — | — | — | — | — |
| Batch-16 | Batch size | 16 | — | — | — | — | — |
| Batch-8 | Batch size | 8 | — | — | — | — | — |
| Batch-4 | Batch size | 4 | — | — | — | — | — |
| Image-640 | Image size | 640 | — | — | — | — | — |
| Image-512 | Image size | 512 | — | — | — | — | — |
| LR-005 | Learning rate | 0.005 | — | — | — | — | — |
| LR-010 | Learning rate | 0.01 | — | — | — | — | — |
| LR-020 | Learning rate | 0.02 | — | — | — | — | — |

The table should only be populated with measured results after the
corresponding training experiments have been completed.

---

## Per-Class Evaluation

For each completed experiment, record Average Precision for all six
classes.

| Experiment | Crazing | Inclusion | Patches | Pitted Surface | Rolled-in Scale | Scratches |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | — | — | — | — | — | — |
| Batch-16 | — | — | — | — | — | — |
| Batch-8 | — | — | — | — | — | — |
| Batch-4 | — | — | — | — | — | — |
| Image-640 | — | — | — | — | — | — |
| Image-512 | — | — | — | — | — | — |
| LR-005 | — | — | — | — | — | — |
| LR-010 | — | — | — | — | — | — |
| LR-020 | — | — | — | — | — | — |

---

## Comparison Procedure

For each experiment:

1. Train YOLOv8 using the specified configuration.
2. Save the trained model and training information.
3. Run inference on the validation dataset.
4. Convert the YOLOv8 predictions using the evaluation adapter.
5. Evaluate predictions using the project's evaluation framework.
6. Record mAP@0.5.
7. Record mAP@0.5:0.95.
8. Record precision and recall.
9. Record per-class AP.
10. Record training time and inference speed when available.
11. Add the measured results to this document.

The same evaluation procedure should be used for all experiments.

---

## Fair Comparison Requirements

To make the experiment results comparable:

- Use the same validation dataset.
- Use the same class mapping.
- Use the same evaluation methodology.
- Do not change multiple experimental parameters at the same time.
- Record the exact configuration used for each training run.
- Record measured results rather than estimated values.
- Keep failed or incomplete experiments clearly marked rather than
  replacing their results.

---

## Relationship to the Evaluation Framework

The experiment results will be evaluated using the evaluation framework
developed during Week 1.

The evaluation pipeline is intended to provide:

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- Per-class AP

The YOLOv8 evaluation adapter developed during Week 2 converts
YOLOv8 prediction results into the format required by the evaluation
pipeline.

---

## Current Status

The experiment matrix has been prepared.

No experimental results are entered yet because the trained model
outputs are not currently available.

Once trained model weights and validation predictions are available,
the experiments can be executed and the measured results recorded here.

The final configuration should be selected based on measured validation
performance together with computational requirements for real-time
edge deployment.