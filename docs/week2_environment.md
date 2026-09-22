\# Week 2 Day 1 - YOLOv8 Environment and GPU Verification



\## Objective



The objective of Day 1 was to verify that the YOLOv8 and PyTorch

environment created during Week 1 is still working correctly and to

check whether a GPU is available for model training.



\## Environment Verification



The environment was tested using the existing Python virtual

environment.



| Component | Result |

|---|---|

| PyTorch | 2.14.0+cpu |

| Ultralytics | 8.4.153 |

| CUDA Available | False |

| CUDA Version | None |

| GPU Count | 0 |

| Training Device | CPU |

| YOLOv8 Model Loading | Successful |



\## GPU Availability



CUDA is not available in the current environment.



The PyTorch installation is the CPU build:



`2.14.0+cpu`



Therefore, the training tests will use the CPU.



\## YOLOv8 Verification



The YOLOv8 model was successfully loaded using Ultralytics.



The verification script completed successfully without errors.



\## Verification Script



The following script was created:



`tests/environment/week2\_environment\_check.py`



It checks:



\- PyTorch version

\- Ultralytics version

\- CUDA availability

\- CUDA version

\- GPU count

\- GPU name

\- Training device

\- YOLOv8 model loading



\## Result



Environment verification: SUCCESS



The YOLOv8/PyTorch environment is ready for the next stage:

running a small training test on a tiny dataset subset before

attempting a full training run.

