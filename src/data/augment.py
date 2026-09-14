import os
import cv2
import albumentations as A


def build_transform():
    transform = A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )
    return transform


def load_yolo_labels(label_path):
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return bboxes, class_labels


def save_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for cls_id, (x, y, w, h) in zip(class_labels, bboxes):
            f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def augment_one_image(image_path, label_path, output_image_path, output_label_path, transform):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    bboxes, class_labels = load_yolo_labels(label_path)
    if not bboxes:
        print(f"No valid bounding boxes found in {label_path}.")
        return False
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    cv2.imwrite(output_image_path, augmented["image"])
    save_yolo_labels(output_label_path, augmented["bboxes"], augmented["class_labels"])
    return True


def augment_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir, num_augmentations=2):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    transform = build_transform()
    class_folders = [f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))]
    total_created = 0
    for class_folder in class_folders:
        class_images_dir = os.path.join(images_dir, class_folder)
        image_files = [f for f in os.listdir(class_images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"Processing {len(image_files)} images in {class_folder}")
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(class_images_dir, image_file)
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            for i in range(num_augmentations):
                output_image_path = os.path.join(output_images_dir, f"{base_name}_aug{i}.jpg")
                output_label_path = os.path.join(output_labels_dir, f"{base_name}_aug{i}.txt")
                success = augment_one_image(image_path, label_path, output_image_path, output_label_path, transform)
                if success:
                    total_created += 1
    print(f"-> Created {total_created} augmented image/label pairs total")


if __name__ == "__main__":
    augment_dataset(
        images_dir="data/raw/NEU-DET/train/images",
        labels_dir="data/labels/train",
        output_images_dir="data/images/train_augmented",
        output_labels_dir="data/labels/train_augmented",
        num_augmentations=2,
    )
    print("Augmentation completed.")
