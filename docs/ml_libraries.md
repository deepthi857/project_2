# Day 2 - Machine Learning Libraries

## Objective

Understand and demonstrate the main Python libraries used in the Zaalima Machine Learning environment.

## 1. NumPy

NumPy is a Python library used for numerical computing and array operations.

### Demonstrated

- Creating NumPy arrays
- Calculating the mean
- Finding maximum values

Example:

```python
numbers = np.array([10, 20, 30, 40, 50])
numbers.mean()
numbers.max()
```

## 2. Pandas

Pandas is a Python library used for loading, manipulating, and analyzing structured data.

### Demonstrated

- Creating a DataFrame
- Accessing DataFrame columns
- Calculating the average of a column

Example:

```python
df = pd.DataFrame(data)
df["Scores"].mean()
```

## 3. Matplotlib

Matplotlib is a Python library used for data visualization.

### Demonstrated

- Creating a line plot
- Adding labels
- Adding a title
- Saving the plot as an image

The Day 2 demonstration generates a plot showing the relationship between study hours and scores.

## 4. Scikit-learn

Scikit-learn is a machine learning library that provides algorithms and tools for building and evaluating machine learning models.

### Demonstrated

- Linear Regression
- Model training
- Making predictions

The demonstration trains a Linear Regression model using study hours and scores and predicts the expected score for 6 hours of study.

## 5. PyTorch

PyTorch is a deep learning framework used for building and training neural networks.

### Demonstrated

- Creating tensors
- Performing tensor operations
- Checking CUDA availability

The current system uses CPU because CUDA is not available on the system.

## 6. Ultralytics YOLOv8

Ultralytics provides the YOLO implementation used for computer vision tasks.

### Demonstrated

- Loading a pretrained YOLOv8 model
- Verifying that the model loads successfully
- Running object detection

YOLOv8 can be used for:

- Object Detection
- Image Classification
- Instance Segmentation

## ML Library Workflow

```text
Raw Data
   |
   v
NumPy / Pandas
   |
   v
Data Analysis
   |
   v
Matplotlib
   |
   v
Scikit-learn / PyTorch
   |
   v
Machine Learning / Deep Learning
   |
   v
Ultralytics YOLO
   |
   v
Prediction / Detection
```

## Day 2 Demonstration Results

The following libraries were successfully tested in the local environment:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch
- Ultralytics YOLOv8

The demonstration successfully performed:

- NumPy array operations
- Pandas DataFrame operations
- Matplotlib visualization
- Linear Regression using Scikit-learn
- PyTorch tensor operations
- YOLOv8 model loading

## Result

The Day 2 machine learning library demonstration was completed successfully.

The demonstration script is available at:

`tests/environment/ml_library_demo.py`