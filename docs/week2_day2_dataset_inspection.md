\# Week 2 - Day 2: Dataset Inspection and YOLO Configuration



\## Objective



Inspect the YOLO dataset structure, verify the training and validation data, check image-label pairing, and validate the YOLO dataset configuration before training.



\## Dataset Structure



The dataset contains:



\- data/images/train

\- data/images/train\_augmented

\- data/images/val

\- data/labels/train

\- data/labels/train\_augmented

\- data/labels/val

\- data/raw/NEU-DET



\## Dataset Counts



\### Training Dataset



\- Training images: 4318

\- Training labels: 4317



\### Validation Dataset



\- Validation images: 360

\- Validation labels: 361



\## Image-Label Consistency



A filename-based image-label pairing check identified:



\- Training image without matching training label: `crazing\_240.jpg`

\- Validation label without matching validation image: `crazing\_240.txt`



Further inspection showed that the same basename exists in different raw NEU-DET splits:



\- `data/raw/NEU-DET/train/images/crazing/crazing\_240.jpg`

\- `data/raw/NEU-DET/validation/annotations/crazing\_240.xml`



No dataset files were deleted or moved.



\## YOLO Classes



The dataset contains six defect classes:



0\. crazing

1\. inclusion

2\. patches

3\. pitted\_surface

4\. rolled-in\_scale

5\. scratches



\## YOLO Dataset Configuration



Configuration file:



`src/configs/data.yaml`



The configuration contains the training path, validation path, and six class names.



\## Configuration Validation



The Ultralytics dataset checker successfully resolved:



Training:



`C:\\Users\\shash\\Desktop\\Zaalima\_git\\data\\images\\train`



Validation:



`C:\\Users\\shash\\Desktop\\Zaalima\_git\\data\\images\\val`



All six class names were recognized correctly.



\## Validation Result



The dataset configuration check completed successfully.



The dataset was ready for the YOLOv8 training pipeline test performed on Day 3.



\## Conclusion



Week 2 Day 2 dataset inspection and YOLO configuration validation were completed successfully.

