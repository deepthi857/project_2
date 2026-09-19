"""
Day 3 - Albumentations Environment Verification

Verifies that Albumentations works correctly
alongside the existing PyTorch and YOLOv8 environment.
"""

import albumentations as A
import cv2
import numpy as np
import torch
import ultralytics


print("=" * 60)
print("DAY 3 - ALBUMENTATIONS ENVIRONMENT VERIFICATION")
print("=" * 60)


# -------------------------------------------------
# 1. Library Versions
# -------------------------------------------------

print("\n[1] Library Versions")

print("Albumentations:", A.__version__)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("OpenCV:", cv2.__version__)
print("CUDA Available:", torch.cuda.is_available())


# -------------------------------------------------
# 2. Create Sample Image
# -------------------------------------------------

print("\n[2] Creating Sample Image")

image = np.zeros((400, 600, 3), dtype=np.uint8)

# Create a simple white rectangle as an object
cv2.rectangle(
    image,
    (150, 100),
    (350, 300),
    (255, 255, 255),
    -1
)

cv2.imwrite("albumentations_original.jpg", image)

print("Original image saved as albumentations_original.jpg")


# -------------------------------------------------
# 3. Define Albumentations Transform
# -------------------------------------------------

print("\n[3] Applying Albumentations Transform")

transform = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(p=1.0)
])


# -------------------------------------------------
# 4. Apply Transform
# -------------------------------------------------

transformed = transform(image=image)

augmented_image = transformed["image"]

cv2.imwrite(
    "albumentations_augmented.jpg",
    augmented_image
)

print("Augmented image saved as albumentations_augmented.jpg")


# -------------------------------------------------
# 5. Verification
# -------------------------------------------------

print("\n[4] Verification")

if image.shape == augmented_image.shape:
    print("Image dimensions preserved: SUCCESS")
else:
    print("Image dimensions preserved: FAILED")


if not np.array_equal(image, augmented_image):
    print("Image transformation applied: SUCCESS")
else:
    print("Image transformation applied: FAILED")


print("\n" + "=" * 60)
print("DAY 3 ALBUMENTATIONS VERIFICATION COMPLETED")
print("=" * 60)