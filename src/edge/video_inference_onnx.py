import os

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import time

import cv2
from ultralytics import YOLO

MODEL_CANDIDATES = ["models/best.onnx", "yolov8n.onnx"]
SOURCE = "data/raw/NEU-DET/validation/images/crazing"
CONF_THRESHOLD = 0.3
MAX_FRAMES = 100
SHOW_WINDOW = False
SAVE_DIR = "runs/inference_frames"
SAVE_EVERY = 10
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def pick_model(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No ONNX model found. Looked for: {candidates}")


def frame_generator(source):
    if isinstance(source, int) or os.path.isfile(str(source)):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
    elif os.path.isdir(str(source)):
        for name in sorted(os.listdir(source)):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                frame = cv2.imread(os.path.join(source, name))
                if frame is not None:
                    yield frame
    else:
        raise FileNotFoundError(f"Source not found: {source}")


def draw_detections(frame, result):
    names = result.names
    count = 0
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        count += 1
    return count


def run(model_path, source):
    print(f"Loading model: {model_path}")
    model = YOLO(model_path, task="detect")
    os.makedirs(SAVE_DIR, exist_ok=True)

    frames = 0
    detections = 0
    infer_seconds = 0.0
    start = time.perf_counter()

    for frame in frame_generator(source):
        t0 = time.perf_counter()
        result = model.predict(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)[0]
        infer_seconds += time.perf_counter() - t0

        frames += 1
        detections += draw_detections(frame, result)

        fps = frames / (time.perf_counter() - start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if frames % SAVE_EVERY == 0:
            cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{frames:04d}.jpg"), frame)

        if SHOW_WINDOW:
            cv2.imshow("Defect Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frames >= MAX_FRAMES:
            break

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    total = time.perf_counter() - start
    print("\n=== Summary ===")
    print("Frames processed:", frames)
    if frames:
        print(f"Average inference time: {infer_seconds / frames * 1000:.1f} ms per frame")
        print(f"End-to-end throughput: {frames / total:.1f} FPS")
    print("Total detections:", detections)
    print("Sample annotated frames saved in:", SAVE_DIR)


if __name__ == "__main__":
    run(pick_model(MODEL_CANDIDATES), SOURCE)
