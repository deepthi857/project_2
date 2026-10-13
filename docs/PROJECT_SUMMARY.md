# Project Summary - Real-Time Industrial Defect Detection System

A computer vision pipeline for manufacturing quality control. It detects and localizes six types of surface defects on metal parts with YOLOv8, exposes the detector as an API that reports defect coordinates to a PLC, and monitors itself with Prometheus and Grafana. For setup and run instructions see docs/HOW_TO_RUN.md.

## Data

- Dataset: NEU-DET (NEU Metal Surface Defects) with bounding-box annotations. Six classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.
- Annotations were converted from VOC XML to YOLO format.
- 1,440 original training images and 360 validation images.
- Albumentations (rotation, noise, brightness/contrast) generated 2,878 augmented image/label pairs, giving 4,318 training images. All 2,878 augmented labels kept at least one box.
- Integrity check: one known mismatch (crazing_240 has no label in the training split).
- Training boxes per class: inclusion 2556, patches 2064, crazing 1572, rolled-in_scale 1488, scratches 1281, pitted_surface 1035.

## Model and results

YOLOv8n, 50 epochs, 640 px, batch 16, trained on a Tesla T4 (Google Colab) in about 0.89 hours.

| Metric (validation set) | Value |
|---|---|
| Precision | 0.646 |
| Recall | 0.694 |
| mAP50 | 0.727 |
| mAP50-95 | 0.401 |
| Inference speed (T4) | about 2.5 ms per image |

| Class | mAP50 | mAP50-95 |
|---|---|---|
| patches | 0.917 | 0.586 |
| inclusion | 0.842 | 0.480 |
| pitted_surface | 0.831 | 0.466 |
| scratches | 0.801 | 0.454 |
| rolled-in_scale | 0.545 | 0.244 |
| crazing | 0.427 | 0.177 |

Crazing is the weakest class: its recall is only 0.295, so most crazing defects are missed. Rolled-in scale is the next weakest. The next steps would be more crazing-specific augmentation and longer training.

Note: metrics are measured on the validation split. There is no separate held-out test split.

## System status

| Component | Status |
|---|---|
| Data conversion, augmentation, integrity checks | Verified |
| Custom evaluation (IoU, precision, recall) | Verified |
| YOLOv8 training | Verified (Colab T4) |
| ONNX export | Verified with yolov8n; re-export from the trained best.pt |
| TensorRT export | Attempted; not possible without an NVIDIA GPU |
| ONNX Runtime frame-by-frame inference with FPS | Implemented (verify before review) |
| FastAPI /detect and /health | Verified |
| Input validation (bad file, empty file, oversize) | Implemented (verify with src/api/test_api_inputs.py) |
| PLC broadcast (UDP JSON) | Implemented (verify with src/api/mock_plc.py) |
| Prometheus metrics, including camera_up | Verified for /metrics; camera_up implemented |
| Grafana datasource and Docker Compose | Written, not yet run |
| Docker image | Written, not yet built |

## Known limitations

- The development laptop's CPU lacks AVX instructions, so local training crashed with Illegal Instruction (exit code -1073741795). Training was moved to Google Colab.
- Trained weights are not stored in git. Without models/best.pt the API falls back to the generic yolov8n.pt, which does not detect these defects.
- TensorRT needs an NVIDIA GPU, which was not available.
- The PLC message is UDP JSON. A real line would replace send_to_plc() with its own protocol (for example Modbus TCP or OPC UA).
- Docker files have not been built or run on the development machine.
