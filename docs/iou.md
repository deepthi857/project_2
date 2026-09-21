\# IoU Calculation and Edge-Case Decisions



\## Overview



Intersection over Union (IoU) measures the overlap between two bounding boxes.



The formula is:



IoU = Intersection Area / Union Area



The IoU value ranges from 0.0 to 1.0:



\- 0.0 = no overlap

\- 1.0 = perfect overlap



\## Bounding Box Format



The project uses YOLO bounding-box format:



(x\_center, y\_center, width, height)



The `compute\_iou()` function converts YOLO coordinates to corner coordinates before calculating the intersection and union areas.



\## Unit Tests



The following hand-built test cases were created:



| Test Case | Expected Result |

|---|---:|

| Perfect overlap | 1.0 |

| No overlap | 0.0 |

| Known partial overlap | 0.6 |

| Edge-touching boxes | 0.0 |

| Zero-area box | 0.0 |

| Symmetry | IoU(A,B) = IoU(B,A) |



All six tests pass.



\## Edge-Case Decisions



\### 1. Edge-touching boxes



If two boxes only touch at an edge or corner, their intersection area is zero.



Decision:



IoU = 0.0



\### 2. Zero-area boxes



A box with zero width or zero height does not contain an area.



Decision:



IoU = 0.0



\### 3. Invalid bounding boxes



A bounding box with negative width or height is invalid input.



Decision:



Raise ValueError



Invalid input should not silently produce a metric value.



\### 4. IoU symmetry



IoU should be independent of the order of the two boxes.



Therefore:



IoU(A, B) = IoU(B, A)



\## Validation Result



The Day 6 IoU unit tests verify the mathematical behavior of the existing IoU implementation using hand-calculated expected answers.



The implementation is ready for the matching logic work planned for Day 7 after invalid-input validation is added.

