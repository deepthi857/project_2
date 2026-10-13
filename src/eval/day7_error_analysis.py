from pathlib import Path
from collections import Counter, defaultdict
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.metrics import compute_iou

GT_DIR = Path("data/labels/val")
PRED_DIR = Path("runs/detect/runs/day7_error_analysis/validation_conf/labels")

CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

IOU_THRESHOLD = 0.5


def load_gt(path):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text().splitlines():
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls, x, y, w, h = parts

        boxes.append({
            "cls": int(float(cls)),
            "box": (
                float(x),
                float(y),
                float(w),
                float(h)
            )
        })

    return boxes


def load_predictions(path):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text().splitlines():
        parts = line.strip().split()

        if len(parts) != 6:
            continue

        cls, x, y, w, h, conf = parts

        boxes.append({
            "cls": int(float(cls)),
            "conf": float(conf),
            "box": (
                float(x),
                float(y),
                float(w),
                float(h)
            )
        })

    return boxes


def main():

    print("=" * 70)
    print("WEEK 2 DAY 7 - ERROR ANALYSIS")
    print("=" * 70)

    gt_files = {
        p.stem: p
        for p in GT_DIR.glob("*.txt")
    }

    pred_files = {
        p.stem: p
        for p in PRED_DIR.glob("*.txt")
    }

    image_stems = sorted(gt_files.keys())

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_confusions = 0

    gt_class_count = Counter()
    pred_class_count = Counter()
    tp_class_count = Counter()
    fp_class_count = Counter()
    fn_class_count = Counter()

    confusion_matrix = Counter()
    confusion_examples = []

    missed_examples = []
    false_positive_examples = []

    for stem in image_stems:

        gt = load_gt(gt_files[stem])
        pred = load_predictions(pred_files[stem]) if stem in pred_files else []

        total_gt += len(gt)
        total_pred += len(pred)

        for item in gt:
            gt_class_count[item["cls"]] += 1

        for item in pred:
            pred_class_count[item["cls"]] += 1

        matched_gt = set()
        matched_pred = set()

        # ------------------------------------------------------------
        # FIRST: identify class confusions
        # ------------------------------------------------------------

        for pi, p in enumerate(pred):

            best_iou = 0.0
            best_gi = -1

            for gi, g in enumerate(gt):

                if gi in matched_gt:
                    continue

                iou = compute_iou(
                    p["box"],
                    g["box"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_gi >= 0 and best_iou >= IOU_THRESHOLD:

                g = gt[best_gi]

                if p["cls"] == g["cls"]:

                    total_tp += 1
                    tp_class_count[g["cls"]] += 1

                else:

                    total_confusions += 1

                    confusion_matrix[
                        (g["cls"], p["cls"])
                    ] += 1

                    confusion_examples.append({
                        "image": stem,
                        "ground_truth": CLASS_NAMES.get(
                            g["cls"],
                            str(g["cls"])
                        ),
                        "prediction": CLASS_NAMES.get(
                            p["cls"],
                            str(p["cls"])
                        ),
                        "confidence": p["conf"],
                        "iou": best_iou,
                    })

                matched_gt.add(best_gi)
                matched_pred.add(pi)

        # ------------------------------------------------------------
        # SECOND: unmatched predictions = false positives
        # ------------------------------------------------------------

        for pi, p in enumerate(pred):

            if pi not in matched_pred:

                total_fp += 1
                fp_class_count[p["cls"]] += 1

                false_positive_examples.append({
                    "image": stem,
                    "prediction": CLASS_NAMES.get(
                        p["cls"],
                        str(p["cls"])
                    ),
                    "confidence": p["conf"],
                })

        # ------------------------------------------------------------
        # THIRD: unmatched GT = missed detections
        # ------------------------------------------------------------

        for gi, g in enumerate(gt):

            if gi not in matched_gt:

                total_fn += 1
                fn_class_count[g["cls"]] += 1

                missed_examples.append({
                    "image": stem,
                    "ground_truth": CLASS_NAMES.get(
                        g["cls"],
                        str(g["cls"])
                    ),
                })

    # ------------------------------------------------------------
    # OVERALL RESULTS
    # ------------------------------------------------------------

    print()
    print("OVERALL RESULTS")
    print("-" * 70)

    print(f"Ground-truth boxes : {total_gt}")
    print(f"Prediction boxes   : {total_pred}")
    print(f"True positives     : {total_tp}")
    print(f"False positives    : {total_fp}")
    print(f"False negatives    : {total_fn}")
    print(f"Class confusions   : {total_confusions}")

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0
    )

    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")

    # ------------------------------------------------------------
    # CLASS DISTRIBUTION
    # ------------------------------------------------------------

    print()
    print("PER-CLASS RESULTS")
    print("-" * 70)

    for cls in range(6):

        name = CLASS_NAMES[cls]

        gt_count = gt_class_count[cls]
        pred_count = pred_class_count[cls]
        tp_count = tp_class_count[cls]
        fp_count = fp_class_count[cls]
        fn_count = fn_class_count[cls]

        print(
            f"{name:18} "
            f"GT={gt_count:3} "
            f"Pred={pred_count:3} "
            f"TP={tp_count:3} "
            f"FP={fp_count:3} "
            f"FN={fn_count:3}"
        )

    # ------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------

    print()
    print("CLASS CONFUSIONS")
    print("-" * 70)

    if confusion_matrix:

        for (gt_cls, pred_cls), count in confusion_matrix.most_common():

            print(
                f"{CLASS_NAMES.get(gt_cls, str(gt_cls))}"
                f" -> "
                f"{CLASS_NAMES.get(pred_cls, str(pred_cls))}"
                f" : {count}"
            )

    else:

        print("No class confusions found at IoU >= 0.5.")

    # ------------------------------------------------------------
    # REPRESENTATIVE CONFUSION EXAMPLES
    # ------------------------------------------------------------

    print()
    print("REPRESENTATIVE CONFUSION EXAMPLES")
    print("-" * 70)

    for item in confusion_examples[:20]:

        print(
            f"{item['image']}: "
            f"GT={item['ground_truth']} | "
            f"Pred={item['prediction']} | "
            f"Conf={item['confidence']:.3f} | "
            f"IoU={item['iou']:.3f}"
        )

    # ------------------------------------------------------------
    # REPRESENTATIVE MISSED DETECTIONS
    # ------------------------------------------------------------

    print()
    print("REPRESENTATIVE MISSED DETECTIONS")
    print("-" * 70)

    for item in missed_examples[:30]:

        print(
            f"{item['image']}: "
            f"GT={item['ground_truth']}"
        )

    # ------------------------------------------------------------
    # REPRESENTATIVE FALSE POSITIVES
    # ------------------------------------------------------------

    print()
    print("REPRESENTATIVE FALSE POSITIVES")
    print("-" * 70)

    for item in false_positive_examples[:30]:

        print(
            f"{item['image']}: "
            f"Pred={item['prediction']} | "
            f"Conf={item['confidence']:.3f}"
        )

    # ------------------------------------------------------------
    # SAVE MACHINE-READABLE REPORT
    # ------------------------------------------------------------

    report_dir = Path("runs/detect/day7_error_analysis")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "error_analysis_summary.txt"

    with report_path.open("w", encoding="utf-8") as f:

        f.write("WEEK 2 DAY 7 - ERROR ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Ground-truth boxes : {total_gt}\n")
        f.write(f"Prediction boxes   : {total_pred}\n")
        f.write(f"True positives     : {total_tp}\n")
        f.write(f"False positives    : {total_fp}\n")
        f.write(f"False negatives    : {total_fn}\n")
        f.write(f"Class confusions   : {total_confusions}\n")
        f.write(f"Precision          : {precision:.4f}\n")
        f.write(f"Recall             : {recall:.4f}\n\n")

        f.write("PER-CLASS RESULTS\n")
        f.write("-" * 70 + "\n")

        for cls in range(6):

            f.write(
                f"{CLASS_NAMES[cls]:18} "
                f"GT={gt_class_count[cls]:3} "
                f"Pred={pred_class_count[cls]:3} "
                f"TP={tp_class_count[cls]:3} "
                f"FP={fp_class_count[cls]:3} "
                f"FN={fn_class_count[cls]:3}\n"
            )

        f.write("\nCLASS CONFUSIONS\n")
        f.write("-" * 70 + "\n")

        for (gt_cls, pred_cls), count in confusion_matrix.most_common():

            f.write(
                f"{CLASS_NAMES[gt_cls]} -> "
                f"{CLASS_NAMES[pred_cls]} : {count}\n"
            )

    print()
    print("=" * 70)
    print(f"Report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
