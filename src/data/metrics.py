import os

def yolo_to_corners(x, y, w, h):
    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2
    return xmin, ymin, xmax, ymax

def compute_iou(box1, box2):
    x1min, y1min, x1max, y1max = yolo_to_corners(*box1)
    x2min, y2min, x2max, y2max = yolo_to_corners(*box2)

    inter_xmin = max(x1min, x2min)
    inter_ymin = max(y1min, y2min)
    inter_xmax = min(x1max, x2max)
    inter_ymax = min(y1max, y2max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1max - x1min) * (y1max - y1min)
    area2 = (x2max - x2min) * (y2max - y2min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def load_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = parts
            boxes.append((int(cls_id), float(x), float(y), float(w), float(h)))
    return boxes

def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    matched_gt = set()
    true_positives = 0
    false_positives = 0

    for pred in pred_boxes:
        pred_cls, px, py, pw, ph = pred
        best_iou = 0
        best_match = -1

        for i, gt in enumerate(gt_boxes):
            gt_cls, gx, gy, gw, gh = gt
            if gt_cls != pred_cls or i in matched_gt:
                continue
            iou = compute_iou((px, py, pw, ph), (gx, gy, gw, gh))
            if iou > best_iou:
                best_iou = iou
                best_match = i

        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_match)
        else:
            false_positives += 1

    false_negatives = len(gt_boxes) - len(matched_gt)
    return true_positives, false_positives, false_negatives

def compute_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def evaluate_dataset(labels_dir, iou_threshold=0.5):
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        gt_boxes = load_labels(label_path)

        tp, fp, fn = match_predictions(gt_boxes, gt_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision, recall = compute_precision_recall(total_tp, total_fp, total_fn)

    print(f"Evaluated {len(label_files)} label files")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return precision, recall


if __name__ == "__main__":
    print("Testing on train set (predictions = ground truth, should score perfectly):")
    evaluate_dataset("data/labels/train", iou_threshold=0.5)
