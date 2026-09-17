import torch
import ultralytics
import numpy
import pandas
import matplotlib
import sklearn


print("====================================")
print("ML ENVIRONMENT VERIFICATION")
print("====================================")

print("PyTorch       :", torch.__version__)
print("Ultralytics   :", ultralytics.__version__)
print("NumPy         :", numpy.__version__)
print("Pandas        :", pandas.__version__)
print("Matplotlib    :", matplotlib.__version__)
print("Scikit-learn  :", sklearn.__version__)

print("CUDA Available:", torch.cuda.is_available())

print("\nAll required ML libraries imported successfully.")

print("====================================")