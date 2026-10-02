# YOLOv8 Hyperparameter Research

## Purpose

This document records the initial hyperparameter research for the
Real-Time Industrial Defect Detection System.

The goal is to identify reasonable starting values for YOLOv8 training
and provide a basis for later experiments and tuning.

These values are starting points and are not considered optimal until
validated experimentally on the project dataset.

---

## Project Dataset

The project uses the NEU Metal Surface Defects Dataset with six defect
classes:

| Class ID | Defect |
|---|---|
| 0 | Crazing |
| 1 | Inclusion |
| 2 | Patches |
| 3 | Pitted Surface |
| 4 | Rolled-in Scale |
| 5 | Scratches |

The validation set currently contains 360 matched images and 854
ground-truth bounding boxes.

---

## Current Project Training Setup

The team's current training configuration uses:

| Parameter | Current value |
|---|---:|
| Model | YOLOv8n |
| Image size | 640 |
| Batch size | 16 |
| Initial epochs | 10 |

The project uses YOLOv8n as the initial model because it is a lightweight
model suitable for experimentation and potential edge deployment.

---

## Recommended Starting Hyperparameters

The following values are recommended as initial starting points based on
the official Ultralytics YOLO training configuration.

| Hyperparameter | Starting value | Purpose |
|---|---:|---|
| Learning rate (`lr0`) | 0.01 | Initial learning rate |
| Final LR factor (`lrf`) | 0.01 | Controls final learning rate |
| Batch size | 16 | Number of images processed per iteration |
| Image size (`imgsz`) | 640 | Input image resolution |
| Momentum | 0.937 | Optimizer momentum |
| Weight decay | 0.0005 | Regularization |
| Warmup epochs | 3.0 | Gradual learning-rate warmup |
| Epochs | 50–100 for a fuller run | Number of training passes |

The current project uses only 10 epochs for the initial training test.
A longer run should be considered after the training pipeline is
confirmed to work correctly.

---

## Why These Hyperparameters Matter

### 1. Learning Rate

The learning rate controls how much the model weights are updated during
training.

A learning rate that is too high can make training unstable, while a
learning rate that is too low can make training slow.

The initial value of `0.01` is the official Ultralytics starting value and
can be used as the baseline for experiments.

### 2. Batch Size

Batch size determines how many images are processed before a parameter
update.

The current project uses:

`batch = 16`

This is also the official Ultralytics default and is a reasonable
baseline when memory permits.

For CPU-constrained training, smaller values such as 8 or 4 can be
tested if memory or training speed becomes a problem.

### 3. Image Size

The current project uses:

`imgsz = 640`

640 is the standard Ultralytics starting resolution.

The original NEU images are small, so later experiments can compare
640 with smaller resolutions if inference speed becomes important for
edge deployment.

### 4. Epochs

The current initial test uses:

`epochs = 10`

Ten epochs are useful for confirming that the complete training pipeline
works.

For actual model training, a longer run such as 50–100 epochs can be
tested and evaluated using the project's mAP framework.

The final number should be selected based on validation performance and
training behavior rather than assuming that more epochs always produce
better results.

### 5. Momentum

Momentum helps the optimizer maintain a direction based on previous
updates.

The official starting value is:

`momentum = 0.937`

### 6. Weight Decay

Weight decay provides regularization and can help reduce overfitting.

The official starting value is:

`weight_decay = 0.0005`

### 7. Warmup

Warmup gradually increases the learning rate at the beginning of
training.

The official starting value is:

`warmup_epochs = 3.0`

Warmup can help stabilize the early stages of training.

---

## Proposed Experiment Plan

Hyperparameters should be changed systematically rather than changing
many parameters simultaneously.

### Experiment 1 — Baseline

Use:

```text
Model: YOLOv8n
Image size: 640
Batch size: 16
Learning rate: 0.01
Epochs: 50