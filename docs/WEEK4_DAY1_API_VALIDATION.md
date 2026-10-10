\# Week 4 Day 1 - FastAPI Detection Output Validation



\## Objective



Validate the FastAPI `/detect` endpoint and verify that:



\- Valid images are accepted correctly.

\- Detection results are returned in the expected JSON format.

\- Detection count is reported correctly.

\- Confidence values are extracted correctly.

\- Bounding-box coordinates are extracted correctly.

\- API inference results are consistent with direct YOLO inference.

\- Invalid image files are handled without crashing the API.



\---



\## API Endpoint



\### Endpoint



```text

POST /detect

