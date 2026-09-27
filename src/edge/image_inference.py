import os

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

import cv2
from ultralytics import YOLO


CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def run_image_inference(model_path, image_path, output_path="inference_output.jpg", conf_threshold=0.3):
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    print(f"Running inference on {image_path}")
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Could not read image: {image_path}")
        return

    results = model.predict(frame, conf=conf_threshold, verbose=False)

    detection_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detection_count += 1

    cv2.imwrite(output_path, frame)
    print(f"Detections found: {detection_count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    model_path = "yolov8n.pt"
    image_path = "data/raw/NEU-DET/train/images/crazing/crazing_1.jpg"
    run_image_inference(model_path, image_path)
