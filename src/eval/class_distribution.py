from pathlib import Path

# Path to validation labels
LABEL_DIR = Path("data/labels/val")

# Class ID to defect name
CLASS_NAMES = {
    0: "Crazing",
    1: "Inclusion",
    2: "Patches",
    3: "Pitted Surface",
    4: "Rolled-in Scale",
    5: "Scratches",
}

# Store counts
image_counts = {class_id: 0 for class_id in CLASS_NAMES}
box_counts = {class_id: 0 for class_id in CLASS_NAMES}

# Process every validation label file
for label_file in LABEL_DIR.glob("*.txt"):

    # Keep track of classes present in this image
    classes_in_image = set()

    with open(label_file, "r") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            values = line.split()

            # First value is the class ID
            class_id = int(values[0])

            # Count bounding box
            box_counts[class_id] += 1

            # Record that this class appears in this image
            classes_in_image.add(class_id)

    # Count image once for every class present
    for class_id in classes_in_image:
        image_counts[class_id] += 1


# Display results
print("\nValidation Dataset Class Distribution")
print("-" * 55)
print(f"{'Class':<20}{'Images':<15}{'Bounding Boxes':<20}")
print("-" * 55)

for class_id, class_name in CLASS_NAMES.items():
    print(
        f"{class_name:<20}"
        f"{image_counts[class_id]:<15}"
        f"{box_counts[class_id]:<20}"
    )

print("-" * 55)
total_images = len(list(LABEL_DIR.glob("*.txt")))
total_boxes = sum(box_counts.values())

print(f"\nTotal validation images: {total_images}")
print(f"Total bounding boxes: {total_boxes}")