\# Week 4 Day 4–5 — Prometheus Latency Cross-Check



\## Objective



Verify that the custom Prometheus inference-latency histogram records the same model inference timings reported by the FastAPI `/detect` endpoint.



\## Environment



\- Python 3.14.5

\- FastAPI

\- Uvicorn

\- Ultralytics YOLO

\- Prometheus Client

\- Prometheus FastAPI Instrumentator

\- CPU inference



\## Metric Under Test



Metric name:



`model\_inference\_latency\_seconds`



Metric type: Histogram



Description: Time spent running model inference on a single image.



The API records the inference duration using `inference\_latency.observe(inference\_time)`.



\## Test Procedure



1\. Started the FastAPI server.

2\. Queried `/metrics` before making inference requests.

3\. Submitted an image to `/detect`.

4\. Recorded the API-reported inference latency.

5\. Queried `/metrics` again.

6\. Repeated the inference requests and compared the cumulative latency measurements.



Test image:



`data/raw/NEU-DET/validation/images/patches/patches\_277.jpg`



\## Results



\### Initial Requests



| Request | API-reported latency |

|---|---:|

| 1 | 2.7708 seconds |

| 2 | 0.1221 seconds |



Prometheus recorded a count of 2 and a cumulative latency sum of approximately 2.892914 seconds.



\### Additional Requests



| Request | API-reported latency |

|---|---:|

| 3 | 0.1098 seconds |

| 4 | 0.0811 seconds |

| 5 | 0.1000 seconds |

| 6 | 0.0863 seconds |

| 7 | 0.0718 seconds |



The five additional requests completed successfully with HTTP 200.



\### Final Prometheus Measurements



| Metric | Observed value |

|---|---:|

| `model\_inference\_latency\_seconds\_count` | 7 |

| `model\_inference\_latency\_seconds\_sum` | 3.341906 seconds |

| Average recorded latency | Approximately 0.4774 seconds |



The Prometheus histogram also showed:



\- 1 observation at or below 0.075 seconds.

\- 4 observations at or below 0.1 seconds.

\- 6 observations at or below 0.25 seconds.

\- 7 observations at or below 5 seconds.



\## Cross-Check



The sum of the seven API-reported, rounded latency values is approximately 3.3419 seconds.



Prometheus recorded a cumulative latency of approximately 3.341906 seconds across seven requests.



The cumulative values agree within the precision expected from rounding the API response.



\## Observations



\- The `/metrics` endpoint exposed the custom inference-latency histogram.

\- The histogram count increased as inference requests were submitted.

\- The histogram sum increased to reflect the recorded inference durations.

\- The first request was substantially slower than subsequent requests.

\- The measured timings include the effects of the current environment and should not be treated as a definitive production benchmark.

\- The API currently uses the generic `yolov8n.pt` model, not the trained NEU-DET model. These measurements validate instrumentation behavior, not defect-classification accuracy.



\## Conclusion



The Prometheus inference-latency histogram was successfully cross-checked against the FastAPI `/detect` endpoint.



Seven inference requests were recorded, and the cumulative Prometheus latency agreed with the API-reported timings within expected rounding differences.



This validates the custom latency instrumentation for the tested requests. Further repeated testing would be needed to establish reliable steady-state latency and throughput benchmarks.

