"""
Week 3 Day 4-7
OpenCV Video Capture + YOLOv8 Inference

Reads frames from a webcam or video file and runs the trained
YOLOv8 model on each captured frame.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = r"runs\detect\runs\train\defect_detection\weights\best.pt"


def open_video(source=0):
    """Open a webcam or video file."""
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    return capture


def run_inference(source=0, max_frames=30):
    """
    Capture frames and run YOLO inference.

    Args:
        source: 0 for the default webcam, or a video file path.
        max_frames: Maximum number of frames to process during testing.
    """
    model = YOLO(MODEL_PATH)
    capture = open_video(source)

    output_dir = Path("runs/day4_7_opencv")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    detection_count = 0

    print("Video capture started.")
    print("YOLO model loaded successfully.")

    while frame_count < max_frames:
        ret, frame = capture.read()

        if not ret:
            print("End of video or failed to read frame.")
            break

        frame_count += 1

        # Run YOLO inference on the current frame.
        results = model.predict(
            source=frame,
            imgsz=320,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        result = results[0]

        if result.boxes is not None:
            frame_detections = len(result.boxes)
            detection_count += frame_detections
        else:
            frame_detections = 0

        # Save the first annotated frame as evidence.
        if frame_count == 1:
            annotated_frame = result.plot()
            output_path = output_dir / "sample_inference_frame.jpg"
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"Annotated frame saved: {output_path}")

        print(
            f"Frame {frame_count}: "
            f"{frame_detections} detections"
        )

    capture.release()
    # GUI cleanup is skipped because this OpenCV build has no HighGUI support.

    print()
    print("INFERENCE SUMMARY")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {detection_count}")
    print("Video capture resources released.")


if __name__ == "__main__":
    # Use 0 for the default webcam.
    # Replace 0 with a video file path to process a video.
    run_inference(0)
