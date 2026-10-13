"""
Day 5 - Albumentations Final Validation

Validates multiple image augmentations while ensuring
bounding boxes remain correctly transformed.
"""

import cv2
import numpy as np
import albumentations as A


print("=" * 65)
print("DAY 5 - ALBUMENTATIONS FINAL VALIDATION")
print("=" * 65)


# -------------------------------------------------
# 1. Create Sample Image
# -------------------------------------------------

print("\n[1] Creating sample image")

image = np.zeros((400, 600, 3), dtype=np.uint8)

# Object bounding box
original_bbox = [150, 100, 350, 300]

# Draw object
cv2.rectangle(
    image,
    (150, 100),
    (350, 300),
    (255, 255, 255),
    -1
)

cv2.imwrite("final_original.jpg", image)

print("Original bounding box:", original_bbox)


# -------------------------------------------------
# 2. Define Augmentation Pipeline
# -------------------------------------------------

print("\n[2] Creating augmentation pipeline")

transform = A.Compose(
    [
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=1.0
        ),
        A.Resize(height=300, width=450)
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"]
    )
)


# -------------------------------------------------
# 3. Apply Transformations
# -------------------------------------------------

print("\n[3] Applying transformations")

result = transform(
    image=image,
    bboxes=[original_bbox],
    class_labels=["object"]
)

augmented_image = result["image"]
transformed_bbox = result["bboxes"][0]


# -------------------------------------------------
# 4. Save Augmented Image
# -------------------------------------------------

x1, y1, x2, y2 = map(int, transformed_bbox)

cv2.rectangle(
    augmented_image,
    (x1, y1),
    (x2, y2),
    (0, 0, 255),
    3
)

cv2.imwrite(
    "final_augmented.jpg",
    augmented_image
)

print("Augmented image saved as final_augmented.jpg")


# -------------------------------------------------
# 5. Validate Bounding Box
# -------------------------------------------------

print("\n[4] Bounding Box Validation")

print("Transformed bounding box:", transformed_bbox)

image_height, image_width = augmented_image.shape[:2]

bbox_valid = (
    0 <= x1 < x2 <= image_width
    and
    0 <= y1 < y2 <= image_height
)

print("Output image size:", image_width, "x", image_height)

if bbox_valid:
    print("Bounding box remains valid: SUCCESS")
else:
    print("Bounding box remains valid: FAILED")


# -------------------------------------------------
# 6. Validate Image Transformation
# -------------------------------------------------

print("\n[5] Image Validation")

if augmented_image.shape[:2] == (300, 450):
    print("Resize transformation: SUCCESS")
else:
    print("Resize transformation: FAILED")


if not np.array_equal(image, augmented_image):
    print("Image was transformed: SUCCESS")
else:
    print("Image was transformed: FAILED")


# -------------------------------------------------
# 7. Final Result
# -------------------------------------------------

print("\n" + "=" * 65)

if bbox_valid and augmented_image.shape[:2] == (300, 450):
    print("DAY 5 FINAL ALBUMENTATIONS VALIDATION PASSED")
else:
    print("DAY 5 FINAL ALBUMENTATIONS VALIDATION FAILED")

print("=" * 65)