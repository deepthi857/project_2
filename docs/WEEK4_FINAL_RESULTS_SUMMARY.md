# Week 4 — Final Project Results Summary



## 1. Project Overview



**Project:** Real-Time Industrial Defect Detection System (Computer Vision)



**Objective:** Develop a computer-vision pipeline for detecting and evaluating metal surface defects using the NEU Metal Surface Defects Database.



The project targets six defect classes:



1. Crazing

2. Inclusion

3. Patches

4. Pitted Surface

5. Rolled-in Scale

6. Scratches



## 2. Dataset Summary



The documented validation dataset contains:



| Item | Result |

|---|---:|

| Training images | 4,318 |

| Validation images | 360 |

| Defect classes | 6 |

| Matched validation image-label pairs | 360 |

| Bounding boxes in matched validation pairs | 854 |

| Unmatched validation label files | 1 |



The unmatched label file is `crazing\_240.txt`. It contains three boxes but has no corresponding validation image.



All validation images are 200 × 200 pixels.



## 3. Model and Training Configuration



The documented baseline model is YOLOv8n.



| Configuration | Value |

|---|---|

| Model | YOLOv8n |

| Training epochs | 1 |

| Image size | 320 |

| Batch size | 4 |

| Device | CPU |

| Data-loader workers | 0 |

| Validation | Enabled |



The training run was intended to verify the training and evaluation pipeline. One epoch is insufficient to establish final model performance.



## 4. Baseline Evaluation Results



The documented validation results are:



| Metric | Value |

|---|---:|

| Precision | 0.3182 |

| Recall | 0.4281 |

| mAP@0.5 | 0.3180 |

| mAP@0.5:0.95 | 0.1428 |



An independent validation of the documented `best.pt` model reported:



| Metric | Value |

|---|---:|

| Precision | 0.3182 |

| Recall | 0.4312 |

| mAP@0.5 | 0.3183 |

| mAP@0.5:0.95 | 0.1429 |



The small differences between these runs may result from differences in evaluation execution or configuration.



### Interpretation



- **Precision:** Measures the proportion of predicted detections that are correct.

- **Recall:** Measures the proportion of ground-truth objects that the model detects.

- **mAP@0.5:** Measures mean average precision at an IoU threshold of 0.50.

- **mAP@0.5:0.95:** Evaluates average precision across IoU thresholds from 0.50 to 0.95.



The baseline results indicate that further training and evaluation are needed before considering the model suitable for production deployment.



## 5. Evaluation Framework



The project includes evaluation utilities for:



- Validation dataset class-distribution analysis.

- YOLO-format annotation handling.

- Bounding-box coordinate conversion.

- Intersection over Union (IoU).

- Precision and recall calculations.

- Mean Average Precision (mAP) evaluation.

- Comparison between predicted detections and ground-truth annotations.



The validation dataset contains 854 boxes across matched image-label pairs.



The evaluation workflow also identifies unmatched label files to prevent dataset inconsistencies from being overlooked.



## 6. ONNX Conversion Validation



The documented PyTorch-versus-ONNX comparison used 360 validation images.



| Metric | Result |

|---|---:|

| Validation images | 360 |

| PyTorch predictions | 483 |

| ONNX predictions | 483 |

| Class agreement | 100% |

| Minimum matched-box IoU | 0.999996 |

| Average matched-box IoU | 0.999999 |

| Maximum matched-box IoU | 1.000000 |

| Maximum confidence difference | 0.000002 |



These results indicate that the documented ONNX export closely preserved the original model's predictions.



The comparison validates prediction equivalence; it does not establish that the model has high defect-detection accuracy.



## 7. OpenCV Inference Validation



The documented OpenCV inference test reported:



| Measurement | Result |

|---|---:|

| Frames processed | 30 |

| Total detections | 145 |



The video-capture pipeline supports reading frames and running YOLO inference. The installed OpenCV environment did not support its HighGUI display functionality, so the test saved output frames instead.



These results demonstrate a functional inference test but do not establish a verified end-to-end FPS benchmark.



## 8. FastAPI Validation



The `/detect` endpoint was tested using images from all six NEU-DET validation class folders.



The tests verified:



- Image upload and decoding.

- JSON response structure.

- Detection count.

- Confidence extraction.

- Bounding-box coordinate extraction.

- Consistency between direct YOLO inference and API inference.

- Handling of invalid image input.



All six class-folder requests returned HTTP 200. An invalid text file returned the message `Could not decode image`.



### Model limitation



The current API loads the generic `yolov8n.pt` model rather than the trained NEU-DET model.



Therefore, these API tests validate input handling and inference-output consistency, not the accuracy of the six NEU-DET defect classes.



The invalid-image response currently uses HTTP 200, which could be improved to an appropriate 4xx client-error status.



## 9. Prometheus Latency Validation



The custom histogram `model\_inference\_latency\_seconds` was tested against the API's reported inference timings.



Seven inference requests were recorded.



| Metric | Result |

|---|---:|

| Requests recorded | 7 |

| Cumulative inference latency | 3.341906 seconds |

| Average across all seven requests | Approximately 0.4774 seconds |



The Prometheus count and cumulative latency agreed with the API-reported values within expected rounding differences.



The first request was substantially slower than subsequent requests. The seven-request average includes this initial observation and should not be treated as a definitive steady-state benchmark.



## 10. Performance and Deployment Limitations



The following limitations remain:



1. The documented baseline used only one training epoch.

2. CPU training and inference are constrained by available hardware.

3. A verified, sustained end-to-end model FPS benchmark has not been established by the results summarized here.

4. The API currently loads generic YOLOv8n weights rather than the trained NEU-DET model.

5. API output validation does not establish defect-classification accuracy.

6. TensorRT acceleration was not validated in the documented CPU-only environment.

7. Additional testing is required before production deployment.



## 11. Recommended Future Work



- Train the defect-detection model for more epochs when sufficient compute is available.

- Re-evaluate the trained model using the established validation framework.

- Ensure the API loads the trained NEU-DET weights.

- Re-test class mapping and API predictions using the trained model.

- Measure sustained FPS, latency percentiles, and throughput.

- Improve invalid-input responses and API error handling.

- Validate the Docker deployment and document reproducible setup instructions.

- Investigate class-level performance and remaining detection errors.



## 12. Conclusion



The project has documented a YOLOv8 baseline evaluation, developed evaluation utilities, validated the documented ONNX conversion, tested OpenCV inference, verified FastAPI response consistency, and cross-checked Prometheus inference-latency measurements.



The results demonstrate progress toward an integrated defect-detection pipeline. However, additional training, trained-model API integration, and reproducible performance benchmarking are needed before claiming production-ready defect-detection performance.



All reported metrics in this summary are based on documented project results. Missing or unmeasured performance values have not been fabricated.

