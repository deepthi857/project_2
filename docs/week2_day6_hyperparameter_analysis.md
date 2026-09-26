\# Week 2 Day 6 - Hyperparameter Tuning Analysis



\## Objective



Compare different YOLOv8 hyperparameter configurations on a small dataset subset and identify the configuration with the highest measured mAP50.



\## Experimental Setup



All experiments used the same:



\- Model: YOLOv8n

\- Dataset fraction: 0.10

\- Training images: 432

\- Validation images: 360

\- Epochs: 2

\- Image size: 320

\- Device: CPU

\- Optimizer: AdamW

\- Workers: 0

\- Seed: 0



Only the learning rate and batch size were varied.



\## Results



| Experiment | Learning Rate | Batch Size | mAP50 | mAP50-95 | Precision | Recall |

|---|---:|---:|---:|---:|---:|---:|

| lr\_low\_batch2 | 0.0005 | 2 | 0.0087 | 0.0019 | 0.5127 | 0.0350 |

| lr\_default\_batch2 | 0.001 | 2 | 0.0107 | 0.0021 | 0.1832 | 0.0329 |

| lr\_default\_batch4 | 0.001 | 4 | 0.0067 | 0.0017 | 0.5113 | 0.0278 |



\## Comparison



\### Learning Rate



The learning rate was compared using batch size 2:



\- Learning rate 0.0005 produced mAP50 = 0.0087.

\- Learning rate 0.001 produced mAP50 = 0.0107.



The 0.001 learning rate produced the higher measured mAP50 in this experiment.



\### Batch Size



The learning rate 0.001 was used to compare batch sizes:



\- Batch size 2 produced mAP50 = 0.0107.

\- Batch size 4 produced mAP50 = 0.0067.



The batch size 2 configuration produced the higher measured mAP50 in this experiment.



\## Best Tested Configuration



Among the three configurations tested, the configuration with:



\- Learning rate: 0.001

\- Batch size: 2

\- mAP50: 0.0107

\- mAP50-95: 0.0021



gave the highest measured mAP50.



Therefore, it was selected as the best tested configuration for this small-subset experiment.



\## Interpretation



The results show that changing the learning rate and batch size affected the measured detection performance on the small subset.



The 0.001 learning rate with batch size 2 produced the highest mAP50 among the tested configurations.



However, these experiments used only 10% of the training dataset and 2 epochs. Therefore, the result should be treated as an experimental comparison rather than a conclusion about the globally optimal YOLOv8 configuration.



\## CPU Constraint



The experiments were intentionally kept small because training was performed on CPU.



Using a small dataset fraction and only two epochs reduced computational time while allowing multiple configurations to be compared.



\## Conclusion



Three YOLOv8 hyperparameter configurations were successfully tested.



The configuration with learning rate 0.001 and batch size 2 achieved the highest measured mAP50 of 0.0107 among the tested configurations.



The experiments successfully demonstrate the hyperparameter tuning and comparison workflow under the available CPU constraint.

