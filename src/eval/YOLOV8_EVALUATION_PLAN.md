\# YOLOv8 Evaluation Integration Plan



\## Purpose



This document defines how the trained YOLOv8 defect detection model will

be connected to the reference evaluation framework prepared during Week 1.



The purpose is to evaluate the trained model using the same validation

dataset and metrics already defined in the evaluation framework.



\## Current Training Setup



The current project training configuration uses:



\- Model: YOLOv8n

\- Epochs: 10

\- Image size: 640

\- Batch size: 16



The base model is `yolov8n.pt`.



\## Validation Dataset



The validation dataset currently contains:



\- 360 matched validation images

\- 6 defect classes

\- 854 ground-truth bounding boxes



Class mapping:



| Class ID | Defect |

|---|---|

| 0 | Crazing |

| 1 | Inclusion |

| 2 | Patches |

| 3 | Pitted Surface |

| 4 | Rolled-in Scale |

| 5 | Scratches |



One label file, `crazing\_240.txt`, does not have a matching validation

image and is excluded from the matched validation evaluation.



\## Evaluation Workflow



Once the trained YOLOv8 model is available, the evaluation pipeline will

follow these steps:



1\. Load the trained YOLOv8 model.

2\. Run inference on all matched validation images.

3\. Collect predicted bounding boxes, confidence scores, and class IDs.

4\. Inspect the actual YOLOv8 prediction output format.

5\. Convert the predictions into the format required by the reference

&#x20;  evaluator.

6\. Match predictions with validation ground-truth bounding boxes using IoU.

7\. Calculate precision and recall.

8\. Calculate per-class Average Precision.

9\. Calculate mAP@0.5.

10\. Calculate mAP@0.5:0.95.

11\. Compare the results with the TorchMetrics reference evaluator.



\## Evaluation Metrics



The evaluation will report:



\- Precision

\- Recall

\- Per-class AP

\- mAP@0.5

\- mAP@0.5:0.95



For mAP@0.5:0.95, AP will be evaluated at IoU thresholds from 0.50 to

0.95 in increments of 0.05.



\## Prediction Format



The existing reference evaluation framework uses the following prediction

representation:



`class\_id confidence x\_center y\_center width height`



The actual YOLOv8 prediction output will be inspected when the trained

model is available.



The prediction adapter will only be finalized after confirming the

actual output format. This avoids assuming a file format before the

trained model has generated predictions.



\## Reference Evaluation



The existing reference evaluator uses TorchMetrics

`MeanAveragePrecision`.



The reference evaluation will provide an independent implementation for

checking the custom evaluation results.



The intended comparison is:



YOLOv8 predictions

→ custom evaluation

→ TorchMetrics reference evaluation

→ compare mAP and per-class AP



\## Current Status



The Week 1 reference evaluation framework is complete.



The Week 2 training configuration is available in the project.



However, a trained `best.pt` or `last.pt` model is not currently

available in the local workspace or in the tracked repository.



Therefore, actual trained-model evaluation is pending the availability

of the trained model.



\## Next Evaluation Task



After the trained model becomes available:



1\. Locate the trained weights.

2\. Run validation/inference using the trained model.

3\. Inspect the generated prediction output.

4\. Implement the YOLOv8 prediction adapter.

5\. Run the reference evaluator.

6\. Record mAP@0.5, mAP@0.5:0.95, precision, recall, and per-class AP.

7\. Compare the reference results with the team's custom evaluation.



\## Acceptance Criteria



The integration will be considered complete when:



\- All matched validation images can be evaluated.

\- YOLOv8 predictions are correctly converted to the evaluator format.

\- mAP@0.5 is calculated.

\- mAP@0.5:0.95 is calculated.

\- Per-class AP is reported.

\- Precision and recall are reported.

\- Results can be independently checked using TorchMetrics.

