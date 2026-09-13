import os
import xml.etree.ElementTree as ET

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def convert_one_file(xml_path, output_txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_w = int(root.find('size').find('width').text)
    img_h = int(root.find('size').find('height').text)

    with open(output_txt_path, 'w') as f:
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in CLASSES:
                continue
            cls_id = CLASSES.index(cls_name)
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            x_center, y_center, width, height = convert_bbox(img_w, img_h, xmin, ymin, xmax, ymax)
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


def convert_split(xml_dir, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    converted = 0
    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(xml_dir, filename)
            output_txt_path = os.path.join(output_dir, filename.replace(".xml", ".txt"))
            convert_one_file(xml_path, output_txt_path)
            converted += 1

    print(f"  -> Converted {converted} files into {output_dir}")


if __name__ == "__main__":
    BASE = "data/raw/NEU-DET"

    splits = [
        (os.path.join(BASE, "train", "annotations"), "data/labels/train"),
        (os.path.join(BASE, "validation", "annotations"), "data/labels/val"),
    ]

    for xml_dir, output_dir in splits:
        if not os.path.isdir(xml_dir):
            print(f"Directory not found: {xml_dir}")
            continue
        convert_split(xml_dir, output_dir)

    print("Conversion completed.")