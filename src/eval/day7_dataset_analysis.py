from pathlib import Path
from collections import Counter

LABEL_DIR = Path("data/labels/val")
IMAGE_DIR = Path("data/images/val")

NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

image_stems = {
    p.stem
    for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
}

box_count = Counter()
image_count = Counter()

total_label_files = 0
matched_label_files = 0
extra_labels = []

for label_file in sorted(LABEL_DIR.glob("*.txt")):

    total_label_files += 1

    if label_file.stem in image_stems:
        matched_label_files += 1
    else:
        extra_labels.append(label_file.stem)

    classes_in_image = set()

    for line in label_file.read_text().splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls = int(float(parts[0]))

        box_count[cls] += 1
        classes_in_image.add(cls)

    for cls in classes_in_image:
        image_count[cls] += 1


print("=" * 70)
print("WEEK 2 DAY 7 - VALIDATION DATASET ANALYSIS")
print("=" * 70)

print()
print("DATASET SUMMARY")
print("-" * 70)

print(f"Validation images      : {len(image_stems)}")
print(f"GT label files         : {total_label_files}")
print(f"Matched label files    : {matched_label_files}")
print(f"Extra label files      : {len(extra_labels)}")

if extra_labels:
    print(f"Extra labels           : {extra_labels}")

print()
print("CLASS DISTRIBUTION")
print("-" * 70)

for cls in range(6):

    print(
        f"{NAMES[cls]:18} "
        f"Images={image_count[cls]:3} "
        f"Boxes={box_count[cls]:3}"
    )

print()
print(f"Total GT boxes         : {sum(box_count.values())}")

print()
print("OBSERVATION")
print("-" * 70)

print(
    "Compare these class counts with the Day 7 model predictions."
)
print(
    "The model currently predicts only the inclusion class."
)

print("=" * 70)
