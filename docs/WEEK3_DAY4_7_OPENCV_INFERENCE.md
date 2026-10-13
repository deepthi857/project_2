# Week 3 Day 4–7 — OpenCV Video Capture and YOLO Inference

## 1. Objective

Implement the core OpenCV video capture loop for reading frames from a webcam or video file and integrate the trained YOLOv8 model to perform inference on the captured frames.

## 2. Environment

- Python: 3.14.2
- OpenCV: 5.0.0
- PyTorch: 2.14.0+cpu
- Ultralytics: 8.4.153
- Execution device: CPU
- CUDA available: False

## 3. Implementation

The video processing script was created at:

src/eval/opencv_video_capture.py

The implementation supports:

- Webcam input using camera index 0
- Video-file input by providing a video path
- Frame-by-frame video capture using OpenCV
- YOLOv8 inference on each captured frame
- Detection counting
- Saving an annotated inference frame
- Proper release of the video capture resource

## 4. Video Capture Loop

OpenCV VideoCapture was used to open the webcam or video source.

For every frame, the script performs:

1. Read the frame using capture.read().
2. Check whether the frame was captured successfully.
3. Run YOLO inference on the frame.
4. Count the detected objects.
5. Save the first annotated frame as evidence.
6. Continue until the configured frame limit is reached or the video ends.
7. Release the video capture resource.

The core capture operation is:

capture = cv2.VideoCapture(source)

Frames are read using:

ret, frame = capture.read()

The loop checks the ret value to handle failed reads or the end of a video file.

## 5. YOLOv8 Inference

The trained YOLOv8 model was loaded from:

runs/detect/runs/train/defect_detection/weights/best.pt

Inference was performed using:

model.predict(
    source=frame,
    imgsz=320,
    conf=0.25,
    device="cpu",
    verbose=False
)

The model processes each captured frame independently.

## 6. Webcam Capture Test

The webcam capture loop was tested successfully.

Test configuration:

- Input source: Default webcam
- Maximum frames: 30
- Frames captured: 30

The first captured frame was successfully saved as:

runs/day4_7_opencv/sample_frame.jpg

The captured sample frame was confirmed to exist with a file size of approximately 60 KB.

## 7. YOLO Inference Test

The complete webcam + YOLO inference pipeline was tested successfully.

Test results:

- Frames processed: 30
- Total detections: 145
- YOLO model loaded successfully
- Annotated frame generated successfully
- Video capture resources released successfully

The first annotated inference frame was saved as:

runs/day4_7_opencv/sample_inference_frame.jpg

The annotated output was confirmed to exist with a file size of 76,569 bytes.

## 8. OpenCV GUI Compatibility

The installed OpenCV 5.0.0 environment does not provide HighGUI support for functions such as cv2.imshow() and cv2.destroyAllWindows().

The initial test confirmed that webcam frame capture and YOLO inference worked correctly, while only the GUI display functions failed.

The implementation was therefore adapted to process and save frames without depending on OpenCV GUI functionality.

This allows the capture and inference pipeline to run successfully in the current CPU environment.

## 9. Result

The complete video processing pipeline was successfully implemented and tested:

Webcam
    ↓
OpenCV VideoCapture
    ↓
Frame Reading
    ↓
YOLOv8 Inference
    ↓
Detection Counting
    ↓
Annotated Frame Output
    ↓
Resource Release

The system successfully processed 30 webcam frames and produced 145 total detections.

## 10. Generated Artifacts

- src/eval/opencv_video_capture.py
- runs/day4_7_opencv/sample_frame.jpg
- runs/day4_7_opencv/sample_inference_frame.jpg

## 11. Conclusion

The OpenCV video capture and YOLOv8 inference pipeline was successfully implemented.

The system can accept webcam or video-file input, read frames sequentially, run YOLOv8 inference on each frame, count detections, and save annotated results.

The implementation is ready for further integration into a real-time defect detection application.

## 12. Next Step

The next stage can extend this pipeline for real-time deployment, including optimized inference, configurable video sources, frame-rate measurement, and integration with the ONNX/TensorRT deployment workflow.
