# Week 2 - YOLOv8 Training Pipeline Test

## Objective

Run a first YOLOv8 training test to confirm that the dataset configuration, model, training pipeline, and validation pipeline work correctly.

## Environment

- Python: 3.14.2
- PyTorch: 2.14.0+cpu
- Ultralytics: 8.4.153
- Device: CPU
- CUDA Available: False
- GPU Count: 0

## Dataset

- Training images: 4318
- Validation images: 360
- Number of classes: 6

### Classes

1. crazing
2. inclusion
3. patches
4. pitted_surface
5. rolled-in_scale
6. scratches

## Training Configuration

- Model: YOLOv8n
- Epochs: 1
- Image size: 320 x 320
- Batch size: 4
- Device: CPU
- Workers: 0
- Validation: Enabled
- Fraction: 1.0

## Training Result

The YOLOv8 training pipeline completed successfully.

Training completed in approximately 6 minutes 51 seconds.

The model successfully completed validation after training.

## Validation Result

Overall validation results after the single training epoch:

- Precision: 0.318
- Recall: 0.431
- mAP50: 0.318
- mAP50-95: 0.143

These values are only from a one-epoch pipeline verification run and should not be interpreted as final model performance.

## Output

The training run generated:

- best.pt
- last.pt
- labels.jpg
- training/validation result files

Output directory:

runs/detect/runs/week2/dataset_check

## Dataset Observations

During dataset scanning, Ultralytics reported several duplicate labels and automatically removed the duplicate annotations from the training cache.

The dataset pipeline completed successfully despite these duplicate-label warnings.

## Conclusion

The YOLOv8 training and validation pipeline was successfully verified on the local CPU environment.

The model can:

1. Load the dataset configuration.
2. Scan the training and validation datasets.
3. Load pretrained YOLOv8 weights.
4. Train for an epoch.
5. Validate the trained model.
6. Save the trained model weights and results.

The environment is ready for further model development and evaluation.

