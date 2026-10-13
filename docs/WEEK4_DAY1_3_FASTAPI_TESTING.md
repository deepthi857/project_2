# Week 4, Days 1–3: FastAPI Defect Detection Testing

## Objective
Test the FastAPI `/detect` endpoint using sample images representing all six NEU-DET defect classes. Verify response consistency and record local environment issues.

## Environment
- OS: Windows
- Python: 3.14.2
- PyTorch: 2.14.0+cpu
- Ultralytics: 8.4.153
- API: FastAPI with Uvicorn
- Model: runs/detect/runs/train/defect_detection/weights/best.pt
- Inference device: CPU
- API URL: http://127.0.0.1:8000
- Image upload interface: FastAPI Swagger UI at /docs
- Confidence threshold: 0.3

## Endpoint Checks

### Health endpoint
- Endpoint: GET /health
- Result: Passed
- Status: healthy
- Model loaded: true
- The endpoint reported the trained model path and class names.

### Detection endpoint
Endpoint: POST /detect

| Sample image | Expected class | HTTP status | Detections | Inference time (s) |
|---|---|---:|---:|---:|
| crazing_1.jpg | crazing | 200 | 0 | 7.6363 |
| crazing_10.jpg | crazing | 200 | 0 | 0.0669 |
| inclusion_1.jpg | inclusion | 200 | 6 | 0.0733 |
| patches_1.jpg | patches | 200 | 0 | 0.1424 |
| pitted_surface_1.jpg | pitted_surface | 200 | 0 | 0.0465 |
| rolled-in_scale_1.jpg | rolled-in_scale | 200 | 0 | 0.0625 |
| scratches_1.jpg | scratches | 200 | 0 | 0.0825 |

All seven requests returned HTTP 200 with the expected JSON response fields: detections, count, and inference_time_seconds.

## Repeatability Test
The inclusion_1.jpg image was submitted a second time.

- HTTP status: 200
- Detection count: 6
- Repeat inference time: 0.0797 seconds
- Predicted class for all six returned boxes: inclusion
- Confidence scores and bounding boxes matched the first response.

Result: Repeatable output was observed for this sample image.

## Environment-Specific Issues
1. FastAPI and Uvicorn were initially missing from the virtual environment.
2. The python-multipart package was required for FastAPI file uploads.
3. PyTorch initially raised a Windows Application Control DLL error. A later standalone import succeeded and reported PyTorch 2.14.0+cpu.
4. Python 3.14.2 was used for testing.
5. The Swagger UI Reset button did not clear the selected file as expected during manual testing.
6. Earlier OpenCV video testing in this project encountered a build without GUI/HighGUI support. This did not prevent the FastAPI image-upload tests.

## Findings and Limitations
- The API started successfully and the health endpoint reported the model as loaded.
- All tested image uploads received HTTP 200 responses.
- Only the inclusion sample produced detections in these tests; the other tested samples returned zero detections.
- Successful HTTP responses confirm request handling, not prediction accuracy.
- The repeated inclusion test produced identical detections, confidence scores, and bounding boxes.
- The first inference took considerably longer than subsequent requests. Warm-up or initialization may contribute, but this was not isolated.
- The small sample set is not sufficient to establish model accuracy across the dataset.

## Conclusion
The FastAPI service was started and its health and detection endpoints were tested locally. Response structure and repeatability were verified for the tested inclusion image. Further model evaluation is needed to investigate zero-detection results for the other classes and to assess detection accuracy across a larger validation set.
