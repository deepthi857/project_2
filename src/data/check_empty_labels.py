import os

labels_dir = "data/labels/train_augmented"
empty_count = 0
total = 0

for filename in os.listdir(labels_dir):
    total += 1
    path = os.path.join(labels_dir, filename)
    if os.path.getsize(path) == 0:
        empty_count += 1

print(f"Total augmented labels: {total}")
print(f"Empty labels (no boxes survived): {empty_count}")