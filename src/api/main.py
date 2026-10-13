<<<<<<< HEAD
﻿import json
=======

>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f
import os
import socket
import time
from pathlib import Path

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

<<<<<<< HEAD
=======
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO


app = FastAPI(title="Defect Detection API")
Instrumentator().instrument(app).expose(app)


# Load the trained six-class defect detection model
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

<<<<<<< HEAD
MODEL_PATH = "models/best.pt" if os.path.exists("models/best.pt") else "yolov8n.pt"
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.3"))

PLC_ENABLED = os.getenv("PLC_ENABLED", "1") == "1"
PLC_HOST = os.getenv("PLC_HOST", "127.0.0.1")
PLC_PORT = int(os.getenv("PLC_PORT", "9000"))
CAMERA_TIMEOUT_SECONDS = float(os.getenv("CAMERA_TIMEOUT_SECONDS", "10"))

model = YOLO(MODEL_PATH)
START_TIME = time.time()
plc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
last_camera_heartbeat = {"time": None}
=======
# Resolve the model path relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "train"
    / "defect_detection"
    / "weights"
    / "best.pt"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f

inference_latency = Histogram(
    "model_inference_latency_seconds",
    "Time spent running model inference on a single image",
)
detections_total = Counter(
    "defects_detected_total",
    "Total number of defects detected",
    ["defect_class"],
)
images_processed_total = Counter(
    "images_processed_total",
    "Images successfully processed by /detect",
)
defective_images_total = Counter(
    "defective_images_total",
    "Images with at least one detected defect",
)
invalid_requests_total = Counter(
    "invalid_requests_total",
    "Rejected /detect requests",
    ["reason"],
)
plc_messages_total = Counter(
    "plc_messages_sent_total",
    "Messages sent to the PLC",
)
plc_send_errors_total = Counter(
    "plc_send_errors_total",
    "Failed attempts to send a message to the PLC",
)
camera_frames_total = Counter(
    "camera_frames_total",
    "Frames reported by the camera heartbeat",
)
uptime_seconds = Gauge("service_uptime_seconds", "Seconds since the API started")
uptime_seconds.set_function(lambda: time.time() - START_TIME)


def camera_is_up():
    last = last_camera_heartbeat["time"]
    if last is None:
        return 0
    return 1 if time.time() - last <= CAMERA_TIMEOUT_SECONDS else 0


camera_up = Gauge("camera_up", "1 if the camera sent a frame recently, else 0")
camera_up.set_function(camera_is_up)


def send_to_plc(payload):
    # UDP + JSON stands in for the factory protocol (Modbus TCP, OPC UA, ...).
    # Swap only this function when the real PLC protocol is known.
    if not PLC_ENABLED:
        return
    try:
        plc_socket.sendto(json.dumps(payload).encode("utf-8"), (PLC_HOST, PLC_PORT))
        plc_messages_total.inc()
    except OSError:
        plc_send_errors_total.inc()


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Defect Detection API",
    }


@app.get("/health")
def health_check():
<<<<<<< HEAD
    return {"status": "healthy", "model_loaded": model is not None, "model_path": MODEL_PATH}


@app.post("/camera/heartbeat")
def camera_heartbeat():
    last_camera_heartbeat["time"] = time.time()
    camera_frames_total.inc()
    return {"status": "ok"}


@app.get("/camera/status")
def camera_status():
    last = last_camera_heartbeat["time"]
    return {
        "camera_up": bool(camera_is_up()),
        "seconds_since_last_frame": None if last is None else round(time.time() - last, 1),
        "timeout_seconds": CAMERA_TIMEOUT_SECONDS,
=======
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "classes": model.names,
>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f
    }


@app.post("/detect")
async def detect_defects(file: UploadFile = File(...)):
    contents = await file.read()
<<<<<<< HEAD
=======

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f

    if len(contents) == 0:
        invalid_requests_total.labels(reason="empty_file").inc()
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(contents) > MAX_UPLOAD_BYTES:
        invalid_requests_total.labels(reason="too_large").inc()
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB")

    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
<<<<<<< HEAD
        invalid_requests_total.labels(reason="not_an_image").inc()
        raise HTTPException(status_code=400, detail="File is not a readable image")

    height, width = frame.shape[:2]

    start_time = time.time()
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
=======
        raise HTTPException(
            status_code=400,
            detail="Could not decode image",
        )

    start_time = time.time()

    results = model.predict(
        frame,
        conf=0.3,
        verbose=False,
    )

>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f
    inference_time = time.time() - start_time
    inference_latency.observe(inference_time)

    detections = []
<<<<<<< HEAD
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
=======
>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            cls_name = (
                CLASSES[cls_id]
                if cls_id < len(CLASSES)
                else f"class_{cls_id}"
            )

            detections_total.labels(
                defect_class=cls_name
            ).inc()

            detections.append({
                "class": cls_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                },
            })

    action = "REJECT" if detections else "PASS"

    send_to_plc({
        "timestamp": round(time.time(), 3),
        "action": action,
        "count": len(detections),
        "image_width": width,
        "image_height": height,
        "detections": detections,
    })

    images_processed_total.inc()
    if detections:
        defective_images_total.inc()

    return {
        "action": action,
        "detections": detections,
        "count": len(detections),
<<<<<<< HEAD
        "inference_time_seconds": round(inference_time, 4),
=======
        "inference_time_seconds": round(
            inference_time, 4
        ),
>>>>>>> dbc17f642f472f8e3551e2e9f2a697ab8a96ff5f
    }
