import torch
from ultralytics import YOLO


print("====================================")
print("Zaalima ML Environment Test")
print("====================================")

# PyTorch test
print("\nPyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 test
print("\nLoading YOLOv8 model...")

model = YOLO("yolov8n.pt")

print("YOLOv8 model loaded successfully!")

# Run detection
print("\nRunning YOLOv8 detection...")

results = model("https://ultralytics.com/images/bus.jpg")

print("\nDetection completed!")

for result in results:
    print("\nDetected objects:")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(
            f"- {class_name}: "
            f"{confidence:.2f}"
        )

print("\n====================================")
print("Environment test completed successfully")
print("====================================")