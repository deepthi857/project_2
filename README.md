 # Steel Surface Defect Annotation Conversion

 ## Introduction

 Steel products may contain different types of surface defects that affect their quality and performance. Manual inspection is time-consuming and may not always be accurate. This project prepares the NEU-DET steel surface defect dataset for automatic defect detection using deep learning.

 ## Objective

 The objective of this project is to convert the XML annotation files of the NEU-DET dataset into YOLO-compatible text files. The converted annotations can be used to train an object detection model.

 ## Methodology

 The original annotations are stored in XML format. Each XML file contains the image dimensions, defect class, and bounding-box coordinates. The conversion process reads these XML files, identifies the defect class, extracts the bounding-box coordinates, converts them into normalized YOLO format, and saves the result as a text file.

 YOLO annotations use the following format:

 ```text
 class_id x_center y_center width height
 ```

 The coordinate values are normalized between 0 and 1 according to the image width and height. The annotations are converted separately for the training and validation datasets.

 ## Defect Classes

 The project includes six steel surface defect classes:

 - Crazing
 - Inclusion
 - Patches
 - Pitted surface
 - Rolled-in scale
 - Scratches

 ## Expected Result

 Each XML annotation file is converted into a corresponding YOLO `.txt` file. These files are stored in the label folders and can be used with YOLO-based object detection frameworks.

 ## Conclusion

 This project completes an important preprocessing step for steel surface defect detection. Converting the annotations into YOLO format makes the NEU-DET dataset compatible with modern deep learning object detection models. The prepared dataset can be used in the next stage to train and evaluate a model for automatic steel defect detection.
