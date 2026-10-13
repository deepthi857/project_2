\# Week 2 - Day 4: YOLOv8 Training Results Validation



\## Objective



Validate the YOLOv8 model produced during the Week 2 training pipeline test and document the resulting performance metrics and generated outputs.



\## Training Configuration



\- Model: YOLOv8n

\- Epochs: 1

\- Image size: 320 x 320

\- Batch size: 4

\- Device: CPU

\- Workers: 0

\- Validation: Enabled

\- Ultralytics: 8.4.153

\- PyTorch: 2.14.0+cpu



\## Dataset



\- Training images: 4318

\- Validation images: 360

\- Validation instances: 854

\- Number of classes: 6



\## Model Outputs



The training process successfully generated:



\- `best.pt`

\- `last.pt`



Model output directory:



`runs/detect/runs/week2/dataset\_check/weights/`



Both model files were generated successfully.



\## Training Results



The one-epoch training run produced the following overall validation metrics:



| Metric | Value |

|---|---:|

| Precision | 0.3182 |

| Recall | 0.4281 |

| mAP@50 | 0.3180 |

| mAP@50-95 | 0.1428 |



\## Independent Validation



The generated `best.pt` model was loaded separately and evaluated using the validation dataset.



Independent validation results:



| Metric | Value |

|---|---:|

| Precision | 0.3182 |

| Recall | 0.4312 |

| mAP@50 | 0.3183 |

| mAP@50-95 | 0.1429 |



The independently calculated metrics are consistent with the metrics recorded during training validation.



\## Class-wise Validation Results



| Class | Precision | Recall | mAP@50 | mAP@50-95 |

|---|---:|---:|---:|---:|

| crazing | 0.188 | 0.623 | 0.244 | 0.0646 |

| inclusion | 0.467 | 0.457 | 0.457 | 0.199 |

| patches | 0.466 | 0.813 | 0.669 | 0.356 |

| pitted\_surface | 0.352 | 0.448 | 0.332 | 0.168 |

| rolled-in\_scale | 0.354 | 0.212 | 0.185 | 0.0595 |

| scratches | 0.0829 | 0.0331 | 0.023 | 0.0102 |



\## Generated Evaluation Outputs



The training run generated:



\- `results.csv`

\- `results.png`

\- `labels.jpg`

\- `confusion\_matrix.png`

\- `confusion\_matrix\_normalized.png`

\- Precision curve

\- Recall curve

\- F1 curve

\- Precision-Recall curve

\- Validation label images

\- Validation prediction images



Training output directory:



`runs/detect/runs/week2/dataset\_check/`



\## Validation Output



An independent validation run using `best.pt` was also completed successfully.



Validation results were saved to:



`runs/detect/val/`



\## Performance Observation



The one-epoch run demonstrates that the complete YOLOv8 pipeline is functioning correctly.



The model successfully completed:



1\. Dataset loading

2\. Model initialization

3\. Training

4\. Validation

5\. Metric calculation

6\. Model weight saving

7\. Prediction generation



The metrics should not be treated as final model performance because the model was trained for only one epoch as a pipeline verification test.



\## CPU Training Observation



The environment does not have CUDA GPU support available.



Training was therefore performed using the CPU.



The one-epoch training run took approximately 6 minutes and 51 seconds.



A full production training run would require significantly more training time on the current CPU-only environment.



\## Conclusion



Week 2 Day 4 validation was completed successfully.



The trained YOLOv8n model was independently loaded and evaluated, and the resulting metrics were consistent with the original training validation results.



The complete training and evaluation pipeline is operational and ready for further model development when required.

