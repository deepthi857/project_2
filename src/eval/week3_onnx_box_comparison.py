from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics import YOLO
from src.data.metrics import compute_iou

PT_MODEL = r"runs\detect\runs\train\defect_detection\weights\best.pt"
ONNX_MODEL = r"runs\detect\runs\train\defect_detection\weights\best.onnx"
SOURCE = r"data\images\val"

pt_model = YOLO(PT_MODEL)
onnx_model = YOLO(ONNX_MODEL)

print("=" * 70)
print("WEEK 3 - BOX LEVEL PYTORCH vs ONNX COMPARISON")
print("=" * 70)

print()
print("Running PyTorch inference...")

pt_results = pt_model.predict(
    source=SOURCE,
    imgsz=320,
    conf=0.25,
    device="cpu",
    verbose=False
)

print(f"PyTorch images processed: {len(pt_results)}")

print()
print("Running ONNX inference...")

onnx_results = onnx_model.predict(
    source=SOURCE,
    imgsz=320,
    conf=0.25,
    device="cpu",
    verbose=False
)

print(f"ONNX images processed: {len(onnx_results)}")

total_detections = 0
matched_detections = 0

iou_values = []
confidence_differences = []
coordinate_differences = []

for pt_result, onnx_result in zip(pt_results, onnx_results):

    pt_boxes = pt_result.boxes
    onnx_boxes = onnx_result.boxes

    pt_data = []

    for i in range(len(pt_boxes)):

        cls = int(pt_boxes.cls[i].item())
        conf = float(pt_boxes.conf[i].item())

        xywh = pt_boxes.xywhn[i].cpu().numpy()

        pt_data.append(
            (
                cls,
                conf,
                tuple(float(x) for x in xywh)
            )
        )

    onnx_data = []

    for i in range(len(onnx_boxes)):

        cls = int(onnx_boxes.cls[i].item())
        conf = float(onnx_boxes.conf[i].item())

        xywh = onnx_boxes.xywhn[i].cpu().numpy()

        onnx_data.append(
            (
                cls,
                conf,
                tuple(float(x) for x in xywh)
            )
        )

    used_onnx = set()

    for pt_cls, pt_conf, pt_box in pt_data:

        total_detections += 1

        best_iou = 0.0
        best_index = -1

        for j, (onnx_cls, onnx_conf, onnx_box) in enumerate(onnx_data):

            if j in used_onnx:
                continue

            if pt_cls != onnx_cls:
                continue

            iou = compute_iou(pt_box, onnx_box)

            if iou > best_iou:

                best_iou = iou
                best_index = j

        if best_index >= 0:

            used_onnx.add(best_index)
            matched_detections += 1

            onnx_cls, onnx_conf, onnx_box = onnx_data[best_index]

            iou_values.append(best_iou)

            confidence_differences.append(
                abs(pt_conf - onnx_conf)
            )

            coordinate_differences.append(
                max(
                    abs(pt_box[k] - onnx_box[k])
                    for k in range(4)
                )
            )

print()
print("=" * 70)
print("BOX LEVEL RESULTS")
print("=" * 70)

print(f"PyTorch detections       : {total_detections}")
print(f"Matched ONNX detections  : {matched_detections}")

if iou_values:

    average_iou = sum(iou_values) / len(iou_values)
    average_conf_diff = (
        sum(confidence_differences)
        / len(confidence_differences)
    )
    average_coord_diff = (
        sum(coordinate_differences)
        / len(coordinate_differences)
    )

    print(f"Minimum box IoU          : {min(iou_values):.6f}")
    print(f"Average box IoU          : {average_iou:.6f}")
    print(f"Maximum box IoU          : {max(iou_values):.6f}")

    print(
        f"Maximum confidence diff : "
        f"{max(confidence_differences):.6f}"
    )

    print(
        f"Average confidence diff : "
        f"{average_conf_diff:.6f}"
    )

    print(
        f"Maximum coordinate diff : "
        f"{max(coordinate_differences):.6f}"
    )

    print(
        f"Average coordinate diff : "
        f"{average_coord_diff:.6f}"
    )

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if iou_values:

    average_iou = sum(iou_values) / len(iou_values)

    if average_iou >= 0.99:

        print(
            "ONNX predictions are numerically very close "
            "to the PyTorch predictions."
        )

    elif average_iou >= 0.95:

        print(
            "ONNX predictions are highly similar "
            "to the PyTorch predictions."
        )

    else:

        print(
            "ONNX predictions show measurable numerical "
            "differences from PyTorch."
        )

print("=" * 70)
