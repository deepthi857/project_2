import cv2
import os


def draw_boxes(image_path, label_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    h, w = image.shape[:2]
    if not os.path.exists(label_path):
        print(f"No label file found: {label_path}")
        return
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, bw, bh = parts
            x, y, bw, bh = float(x), float(y), float(bw), float(bh)
            xmin = int((x - bw / 2) * w)
            ymin = int((y - bh / 2) * h)
            xmax = int((x + bw / 2) * w)
            ymax = int((y + bh / 2) * h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, cls_id, (xmin, max(ymin - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    draw_boxes(
        "data/images/train_augmented/crazing_1_aug0.jpg",
        "data/labels/train_augmented/crazing_1_aug0.txt",
        "verify_output.jpg",
    )