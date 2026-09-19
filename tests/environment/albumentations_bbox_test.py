"""
Day 4 - Albumentations Bounding Box Verification

Tests whether Albumentations correctly transforms
bounding boxes together with the image.
"""

import cv2
import numpy as np
import albumentations as A


print("=" * 60)
print("DAY 4 - BOUNDING BOX AUGMENTATION TEST")
print("=" * 60)


# -------------------------------------------------
# 1. Create Sample Image
# -------------------------------------------------

print("\n[1] Creating sample image")

image = np.zeros((400, 600, 3), dtype=np.uint8)

# Original object bounding box
x_min = 150
y_min = 100
x_max = 350
y_max = 300

# Draw the object
cv2.rectangle(
    image,
    (x_min, y_min),
    (x_max, y_max),
    (255, 255, 255),
    -1
)

# Draw original bounding box in green
cv2.rectangle(
    image,
    (x_min, y_min),
    (x_max, y_max),
    (0, 255, 0),
    3
)

cv2.imwrite("bbox_original.jpg", image)

print("Original image saved as bbox_original.jpg")


# -------------------------------------------------
# 2. Define Bounding Box
# -------------------------------------------------

original_bbox = [
    x_min,
    y_min,
    x_max,
    y_max
]

print("Original bounding box:", original_bbox)


# -------------------------------------------------
# 3. Define Albumentations Transform
# -------------------------------------------------

print("\n[2] Creating Albumentations transform")

transform = A.Compose(
    [
        A.HorizontalFlip(p=1.0)
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"]
    )
)


# -------------------------------------------------
# 4. Apply Transformation
# -------------------------------------------------

print("\n[3] Applying transformation")

result = transform(
    image=image,
    bboxes=[original_bbox],
    class_labels=["object"]
)

augmented_image = result["image"]
transformed_bbox = result["bboxes"][0]

print("Transformed bounding box:", transformed_bbox)


# -------------------------------------------------
# 5. Draw Transformed Bounding Box
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
    "bbox_augmented.jpg",
    augmented_image
)

print("Augmented image saved as bbox_augmented.jpg")


# -------------------------------------------------
# 6. Verify Bounding Box Transformation
# -------------------------------------------------

print("\n[4] Bounding Box Verification")

expected_bbox = [
    image.shape[1] - x_max,
    y_min,
    image.shape[1] - x_min,
    y_max
]

print("Expected bounding box:", expected_bbox)
print("Actual bounding box:  ", list(transformed_bbox))


# Allow tiny floating-point differences
bbox_correct = np.allclose(
    transformed_bbox,
    expected_bbox,
    atol=1.0
)


if bbox_correct:
    print("Bounding box transformation: SUCCESS")
else:
    print("Bounding box transformation: FAILED")


# -------------------------------------------------
# 7. Final Result
# -------------------------------------------------

print("\n" + "=" * 60)

if bbox_correct:
    print("DAY 4 BOUNDING BOX TEST PASSED")
else:
    print("DAY 4 BOUNDING BOX TEST FAILED")

print("=" * 60)