"""
Day 2 - Machine Learning Library Demonstration

Demonstrates the basic functionality of:
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- Ultralytics YOLO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.linear_model import LinearRegression
from ultralytics import YOLO


print("=" * 50)
print("DAY 2 - ML LIBRARY DEMONSTRATION")
print("=" * 50)


# -------------------------------------------------
# 1. NumPy
# -------------------------------------------------

print("\n[1] NumPy")

numbers = np.array([10, 20, 30, 40, 50])

print("Array:", numbers)
print("Mean:", numbers.mean())
print("Maximum:", numbers.max())


# -------------------------------------------------
# 2. Pandas
# -------------------------------------------------

print("\n[2] Pandas")

data = {
    "Hours": [1, 2, 3, 4, 5],
    "Scores": [20, 35, 50, 65, 80]
}

df = pd.DataFrame(data)

print(df)
print("Average Score:", df["Scores"].mean())


# -------------------------------------------------
# 3. Matplotlib
# -------------------------------------------------

print("\n[3] Matplotlib")

plt.plot(df["Hours"], df["Scores"], marker="o")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")

plt.savefig("ml_library_demo_plot.png")
plt.close()

print("Plot saved as ml_library_demo_plot.png")


# -------------------------------------------------
# 4. Scikit-learn
# -------------------------------------------------

print("\n[4] Scikit-learn")

X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict(pd.DataFrame({"Hours": [6]}))

print("Predicted score for 6 hours:", prediction[0])


# -------------------------------------------------
# 5. PyTorch
# -------------------------------------------------

print("\n[5] PyTorch")

tensor = torch.tensor([10.0, 20.0, 30.0])

print("Tensor:", tensor)
print("Tensor mean:", tensor.mean())
print("CUDA available:", torch.cuda.is_available())


# -------------------------------------------------
# 6. Ultralytics YOLO
# -------------------------------------------------

print("\n[6] Ultralytics YOLO")

yolo_model = YOLO("yolov8n.pt")

print("YOLOv8 model loaded successfully!")


print("\n" + "=" * 50)
print("DAY 2 ML LIBRARY DEMONSTRATION COMPLETED")
print("=" * 50)