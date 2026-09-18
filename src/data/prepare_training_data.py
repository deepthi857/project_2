import os
import shutil


def copy_files(src_dir, dst_dir, extensions):
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(extensions):
            shutil.copy(
                os.path.join(src_dir, filename),
                os.path.join(dst_dir, filename)
            )
            count += 1
    return count


def flatten_class_images(raw_images_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    total = 0
    class_folders = [f for f in os.listdir(raw_images_dir) if os.path.isdir(os.path.join(raw_images_dir, f))]
    for class_folder in class_folders:
        class_path = os.path.join(raw_images_dir, class_folder)
        count = copy_files(class_path, dst_dir, (".jpg", ".jpeg", ".png"))
        total += count
        print(f"  Copied {count} images from {class_folder}")
    return total


if __name__ == "__main__":
    print("Preparing training images...")
    original_count = flatten_class_images("data/raw/NEU-DET/train/images", "data/images/train")
    augmented_count = copy_files("data/images/train_augmented", "data/images/train", (".jpg", ".jpeg", ".png"))
    print(f"Total original images: {original_count}")
    print(f"Total augmented images: {augmented_count}")

    print("\nPreparing validation images...")
    val_count = flatten_class_images("data/raw/NEU-DET/validation/images", "data/images/val")
    print(f"Total validation images: {val_count}")

    print("\nPreparing training labels...")
    os.makedirs("data/labels/train_combined", exist_ok=True)
    train_label_count = copy_files("data/labels/train", "data/labels/train_combined", (".txt",))
    aug_label_count = copy_files("data/labels/train_augmented", "data/labels/train_combined", (".txt",))
    print(f"Combined labels: {train_label_count + aug_label_count}")

    print("\nDone preparing training data.")