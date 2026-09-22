\# Greedy Detection Matching



\## Overview



Detection matching assigns predicted bounding boxes to ground-truth

bounding boxes so that True Positives (TP), False Positives (FP), and

False Negatives (FN) can be calculated.



The matching logic uses:



\- Detection confidence

\- Object class

\- Intersection over Union (IoU)

\- IoU threshold

\- Ground-truth matching status



\## Prediction Format



Predictions use:



(class\_id, confidence, x\_center, y\_center, width, height)



Ground-truth boxes use:



(class\_id, x\_center, y\_center, width, height)



\## Matching Process



Predictions are sorted by confidence in descending order.



For every prediction:



1\. Find unmatched ground-truth boxes.

2\. Ignore ground-truth boxes with a different class.

3\. Calculate IoU with each eligible ground-truth box.

4\. Select the ground-truth box with the highest IoU.

5\. If the best IoU is greater than or equal to the threshold, count

&#x20;  the prediction as a True Positive.

6\. Mark the matched ground-truth box as used.

7\. Otherwise, count the prediction as a False Positive.



After all predictions are processed:



False Negatives = Total ground-truth boxes - Matched ground-truth boxes



\## Edge-Case Decisions



\### 1. Two detections for one ground-truth box



Only one detection can match a ground-truth box.



The higher-confidence detection is processed first and becomes the

True Positive when the IoU threshold is satisfied.



A second detection for the same ground-truth object becomes a

False Positive.



\### 2. Wrong class



A detection with a high IoU but an incorrect class cannot match the

ground-truth box.



Decision:



\- Detection = False Positive

\- Ground truth = False Negative if it remains unmatched



\### 3. IoU below threshold



If the best available IoU is below the configured threshold, the

prediction is counted as a False Positive.



The ground-truth box remains unmatched and therefore contributes to

False Negatives.



\### 4. Confidence ordering



Predictions are processed from highest confidence to lowest confidence.



This ensures that when multiple detections compete for the same

ground-truth box, the higher-confidence detection gets the first

opportunity to match.



\### 5. Multiple ground-truth boxes



A prediction is matched to the unmatched ground-truth box having the

highest IoU, provided that the class is correct and the IoU reaches

the configured threshold.



\### 6. Equal confidence



Python's stable sorting preserves the original input order when two

predictions have the same confidence.



\## Validation



Seven hand-built unit tests were created.



The tests verify:



\- Correct detection

\- Duplicate detections

\- Wrong class

\- IoU threshold

\- Confidence ordering

\- Best ground-truth selection

\- Unmatched ground-truth handling



Test result:



7 passed

