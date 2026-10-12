# Week 4 Days 6-7 — Docker Deployment and Handover

## 1. Objective

Build and run the Real-Time Industrial Defect Detection System API in Docker and verify model inference, health checks, and Prometheus monitoring.

## 2. Environment

- Operating system: Windows with Docker Desktop and WSL 2
- Docker engine: 29.8.2
- Base image: python:3.14-slim
- Application: FastAPI with Uvicorn
- Model: YOLOv8n trained for six metal surface defect classes
- Container port: 8000

## 3. Docker Configuration

A Dockerfile was created in the project root. It installs the Python requirements, copies the API source code and trained model, and starts Uvicorn on 0.0.0.0:8000.

Image name: defect-detection-api:latest

## 4. Verification Results

### Image Build
Result: PASS

The Docker image built successfully with all 13 build steps completed.

### Container Startup
Result: PASS

The container started successfully and Uvicorn reported that the application startup was complete.

### Health Endpoint
Endpoint: GET /health

Result: PASS

- Status: healthy
- Model loaded: true
- Model path: /app/runs/detect/runs/train/defect_detection/weights/best.pt

### Defect Detection
Endpoint: POST /detect

Test image: inclusion_1.jpg

Result: PASS

- HTTP status: 200
- Detections returned: 6
- Detected class: inclusion
- Inference time: approximately 4.9142 seconds

### Prometheus Monitoring
Endpoint: GET /metrics

Result: PASS

The metrics endpoint exposed the custom inference latency histogram, defect detection counter, and HTTP request metrics.

Observed values after one detection request:

- model_inference_latency_seconds_count: 1
- model_inference_latency_seconds_sum: approximately 4.9142
- defects_detected_total{defect_class="inclusion"}: 6
- Successful health and detection requests were recorded.

## 5. Run Instructions

Build the image:

    docker build -t defect-detection-api .

Start the container:

    docker run -d --name defect-api-test -p 8000:8000 defect-detection-api

Check health:

    http://localhost:8000/health

View metrics:

    http://localhost:8000/metrics

View logs:

    docker logs defect-api-test

## 6. Notes

- The model ran successfully inside the container using CPU inference.
- The first image build took approximately 20 minutes because of dependency installation and image export.
- Ultralytics displayed a configuration-directory warning and used /tmp/Ultralytics instead. This warning did not prevent startup or inference.
- The detection result verifies functionality on the supplied test image; it does not establish overall model accuracy.
- The Docker image has been tested locally. Deployment to another machine has not yet been verified.

## 7. Conclusion

The Docker image built successfully, the FastAPI container started, the trained YOLOv8 model loaded, the detection endpoint returned successful results, and Prometheus metrics reflected the inference request.
