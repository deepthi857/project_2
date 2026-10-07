# Week 3 Day 1–3 — PyTorch to ONNX Conversion

## 1. Objective

Convert the trained YOLOv8 PyTorch model to ONNX and verify that the exported ONNX model loads correctly and produces predictions equivalent to the original PyTorch model.

## 2. Environment

- Python: 3.14.2
- PyTorch: 2.14.0+cpu
- Ultralytics: 8.4.153
- Execution device: CPU
- CUDA available: False
- ONNX Runtime: 1.30.0
- ONNX: 1.23.2
- ONNX opset: 18

## 3. Source Model

The trained YOLOv8 model used for conversion was:

runs/detect/runs/train/defect_detection/weights/best.pt

The model contains:

- 72 layers
- 3,006,818 parameters
- Approximately 2.0 GFLOPs
- Input shape: 1 x 3 x 320 x 320

## 4. ONNX Export

The model was exported using the built-in Ultralytics YOLO export functionality:

model.export(format="onnx", imgsz=320, device="cpu")

The export completed successfully.

Generated ONNX model:

runs/detect/runs/train/defect_detection/weights/best.onnx

ONNX model size:

- 11.58 MB

## 5. ONNX Loading Verification

The exported ONNX model was loaded using Ultralytics YOLO.

Verification results:

- ONNX file exists: True
- ONNX model loaded successfully
- Model task: detect

The ONNX model was therefore confirmed to be valid and loadable.

## 6. PyTorch vs ONNX Prediction Comparison

Both models were evaluated on the same validation dataset.

Validation results:

- Validation images: 360
- PyTorch predictions: 483
- ONNX predictions: 483
- Images with same classes: 360
- Images with prediction differences: 0
- Class agreement: 100.00%

These results show that the ONNX model produced the same detection classes and prediction counts as the original PyTorch model.

## 7. Bounding-Box and Confidence Comparison

A box-level comparison was also performed between the PyTorch and ONNX predictions.

Results:

- PyTorch detections: 483
- Matched ONNX detections: 483
- Minimum box IoU: 0.999996
- Average box IoU: 0.999999
- Maximum box IoU: 1.000000
- Maximum confidence difference: 0.000002
- Average confidence difference: 0.000000
- Maximum coordinate difference: 0.000002
- Average coordinate difference: 0.000000

The extremely high IoU and very small numerical differences demonstrate that the ONNX predictions are effectively equivalent to the PyTorch predictions.

## 8. Conclusion

The YOLOv8 PyTorch model was successfully converted to ONNX.

The exported ONNX model:

1. Was generated successfully.
2. Loaded successfully as a detection model.
3. Produced predictions on all 360 validation images.
4. Produced the same 483 detections as the PyTorch model.
5. Achieved 100.00% class agreement.
6. Achieved an average bounding-box IoU of 0.999999.
7. Showed negligible confidence and coordinate differences.

Therefore, the ONNX conversion was successfully validated and preserves the original model's predictions.

## 9. Generated Artifacts

- src/eval/week3_onnx_comparison.py
- src/eval/week3_onnx_box_comparison.py
- runs/detect/runs/train/defect_detection/weights/best.onnx

## 10. Next Step

The next stage is TensorRT investigation and deployment.

TensorRT requires NVIDIA GPU hardware and the appropriate TensorRT/CUDA environment. Since the current development environment is CPU-only, TensorRT execution is hardware-dependent and will be handled separately.
