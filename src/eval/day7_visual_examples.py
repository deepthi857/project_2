from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IMAGE_DIR = Path("data/images/val")
GT_DIR = Path("data/labels/val")
PRED_DIR = Path("runs/detect/runs/day7_error_analysis/validation_conf/labels")
OUT_DIR = Path("runs/detect/day7_error_analysis/visual_examples")

OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

EXAMPLES = [
    "patches_272",
    "scratches_287",
    "scratches_288",
]


def read_labels(path, prediction=False):

    items = []

    if not path.exists():
        return items

    for line in path.read_text().splitlines():

        parts = line.strip().split()

        if prediction and len(parts) == 6:

            cls, x, y, w, h, conf = parts

            items.append(
                (
                    int(float(cls)),
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                    float(conf),
                )
            )

        elif not prediction and len(parts) == 5:

            cls, x, y, w, h = parts

            items.append(
                (
                    int(float(cls)),
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                    None,
                )
            )

    return items


def draw_box(draw, box, image_width, image_height, text):

    cls, x, y, w, h, conf = box

    xmin = int((x - w / 2) * image_width)
    ymin = int((y - h / 2) * image_height)
    xmax = int((x + w / 2) * image_width)
    ymax = int((y + h / 2) * image_height)

    draw.rectangle(
        [xmin, ymin, xmax, ymax],
        outline="red" if conf is not None else "green",
        width=3,
    )

    draw.text(
        (xmin, max(0, ymin - 15)),
        text,
        fill="red" if conf is not None else "green",
    )


for stem in EXAMPLES:

    image_path = None

    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:

        candidate = IMAGE_DIR / f"{stem}{ext}"

        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"Image not found: {stem}")
        continue

    image = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(image)

    gt_boxes = read_labels(
        GT_DIR / f"{stem}.txt",
        prediction=False
    )

    pred_boxes = read_labels(
        PRED_DIR / f"{stem}.txt",
        prediction=True
    )

    for box in gt_boxes:

        cls = box[0]

        draw_box(
            draw,
            box,
            image.width,
            image.height,
            f"GT: {CLASS_NAMES.get(cls, str(cls))}",
        )

    for box in pred_boxes:

        cls = box[0]
        conf = box[5]

        draw_box(
            draw,
            box,
            image.width,
            image.height,
            f"Pred: {CLASS_NAMES.get(cls, str(cls))} {conf:.2f}",
        )

    output = OUT_DIR / f"{stem}_error_analysis.jpg"

    image.save(output)

    print(f"Saved: {output}")


print()
print("Visual error examples generated successfully.")
