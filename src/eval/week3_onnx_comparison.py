from pathlib import Path
from ultralytics import YOLO

PT_MODEL = r"runs\detect\runs\train\defect_detection\weights\best.pt"
ONNX_MODEL = r"runs\detect\runs\train\defect_detection\weights\best.onnx"
SOURCE = r"data\images\val"

OUT_DIR = Path(r"runs\detect\week3_onnx_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("WEEK 3 - PYTORCH vs ONNX PREDICTION COMPARISON")
print("=" * 70)

print()
print("Loading PyTorch model...")
pt_model = YOLO(PT_MODEL)

print("Loading ONNX model...")
onnx_model = YOLO(ONNX_MODEL)

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

# ------------------------------------------------------------
# Compare predictions
# ------------------------------------------------------------

total_pt_boxes = 0
total_onnx_boxes = 0
same_detection_count = 0
different_detection_count = 0

image_comparisons = []

for pt_result, onnx_result in zip(pt_results, onnx_results):

    pt_boxes = pt_result.boxes
    onnx_boxes = onnx_result.boxes

    pt_count = len(pt_boxes)
    onnx_count = len(onnx_boxes)

    total_pt_boxes += pt_count
    total_onnx_boxes += onnx_count

    pt_classes = (
        pt_boxes.cls.cpu().numpy().astype(int).tolist()
        if pt_count > 0
        else []
    )

    onnx_classes = (
        onnx_boxes.cls.cpu().numpy().astype(int).tolist()
        if onnx_count > 0
        else []
    )

    pt_classes_sorted = sorted(pt_classes)
    onnx_classes_sorted = sorted(onnx_classes)

    if pt_classes_sorted == onnx_classes_sorted:
        same_detection_count += 1
        status = "SAME"
    else:
        different_detection_count += 1
        status = "DIFFERENT"

    image_name = Path(pt_result.path).name

    image_comparisons.append(
        (
            image_name,
            pt_count,
            onnx_count,
            status,
            pt_classes_sorted,
            onnx_classes_sorted,
        )
    )

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 70)
print("COMPARISON RESULTS")
print("=" * 70)

print(f"Validation images        : {len(pt_results)}")
print(f"PyTorch predictions      : {total_pt_boxes}")
print(f"ONNX predictions         : {total_onnx_boxes}")
print(f"Images with same classes : {same_detection_count}")
print(f"Images with differences  : {different_detection_count}")

agreement = (
    same_detection_count / len(pt_results) * 100
    if pt_results
    else 0
)

print(f"Class agreement          : {agreement:.2f}%")

print()
print("DIFFERENT PREDICTION EXAMPLES")
print("-" * 70)

shown = 0

for item in image_comparisons:

    if item[3] == "DIFFERENT":

        print(
            f"{item[0]} | "
            f"PyTorch={item[1]} boxes "
            f"{item[4]} | "
            f"ONNX={item[2]} boxes "
            f"{item[5]}"
        )

        shown += 1

        if shown >= 20:
            break

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

report_path = OUT_DIR / "pytorch_vs_onnx_comparison.txt"

with report_path.open("w", encoding="utf-8") as f:

    f.write("WEEK 3 - PYTORCH vs ONNX PREDICTION COMPARISON\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Validation images        : {len(pt_results)}\n")
    f.write(f"PyTorch predictions      : {total_pt_boxes}\n")
    f.write(f"ONNX predictions         : {total_onnx_boxes}\n")
    f.write(f"Images with same classes : {same_detection_count}\n")
    f.write(f"Images with differences  : {different_detection_count}\n")
    f.write(f"Class agreement          : {agreement:.2f}%\n\n")

    f.write("DIFFERENT PREDICTIONS\n")
    f.write("-" * 70 + "\n")

    for item in image_comparisons:

        if item[3] == "DIFFERENT":

            f.write(
                f"{item[0]} | "
                f"PyTorch={item[1]} boxes {item[4]} | "
                f"ONNX={item[2]} boxes {item[5]}\n"
            )

print()
print(f"Comparison report saved to: {report_path}")
print("=" * 70)
