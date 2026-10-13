# Week 4, Days 4–5: Prometheus Monitoring Verification

## Objective
Verify that the FastAPI `/metrics` endpoint exposes Prometheus metrics and that metrics update when requests are sent to the API.

## Environment
- Operating System: Windows
- Python: 3.14.2
- API: FastAPI with Uvicorn
- Monitoring: prometheus-fastapi-instrumentator and prometheus-client
- API URL: http://127.0.0.1:8000
- Metrics endpoint: GET /metrics

## Tests Performed

### 1. Health Check
- Endpoint: GET /health
- Result: Passed
- Status: healthy
- Trained model loaded successfully.

### 2. Metrics Endpoint
- Endpoint: GET /metrics
- Result: Passed
- Prometheus-formatted output was returned.
- Metrics included model_inference_latency_seconds, defects_detected_total, and http_requests_total.

### 3. First Detection Request
- Endpoint: POST /detect
- Sample: inclusion_1.jpg
- HTTP status: 200
- Detections returned: 6
- Inference time from response: approximately 5.5537 seconds
- Recorded inference count: 1
- Recorded inference time sum: approximately 5.5537 seconds
- Inclusion defect counter: 6
- Successful /detect request counter: 1

### 4. Second Detection Request
The same sample image was submitted again.

- HTTP status: 200
- Recorded inference count: 2
- Recorded inference time sum: approximately 5.6134 seconds
- Inclusion defect counter: 12
- Successful /detect request counter: 2

## Results
- The /metrics endpoint returned Prometheus-formatted metric output.
- The inference latency histogram recorded both inference requests.
- The defect counter increased from 6 to 12 for the inclusion class.
- The HTTP request counter increased from 1 to 2 for successful POST requests to /detect.
- The metrics changed as requests were processed, confirming that monitoring instrumentation is active.

## Conclusion
The FastAPI Prometheus monitoring endpoint was verified locally. The inference latency, defect detection, and HTTP request metrics updated correctly across two successful detection requests.
