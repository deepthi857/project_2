import os
import time

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

app = FastAPI(title="Defect Detection API")

Instrumentator().instrument(app).expose(app)

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

inference_latency = Histogram(
    "model_inference_latency_seconds",
    "Time spent running model inference on a single image"
)

detections_total = Counter(
    "defects_detected_total",
    "Total number of defects detected",
    ["defect_class"]
)


@app.get("/")
def root():
    return {"status": "running", "service": "Defect Detection API"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/detect")
async def detect_defects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not decode image"}

    start_time = time.time()
    results = model.predict(frame, conf=0.3, verbose=False)
    inference_time = time.time() - start_time
    inference_latency.observe(inference_time)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"

            detections_total.labels(defect_class=cls_name).inc()

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                }
            })

    return {
        "detections": detections,
        "count": len(detections),
        "inference_time_seconds": round(inference_time, 4)
    }
