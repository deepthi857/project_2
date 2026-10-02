import time

import cv2

from fps_logger import FPSLogger


# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_INDEX = 0
WINDOW_NAME = "OpenCV FPS Benchmark"


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("WEEK 3 - DAY 5")
    print("OPEN CV FPS BENCHMARK")
    print("=" * 60)

    print(f"Opening camera: {CAMERA_INDEX}")

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera with index {CAMERA_INDEX}."
        )

    fps_logger = FPSLogger()

    fps_logger.start()

    print("Camera opened successfully.")
    print("Press 'q' to stop the benchmark.")

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read frame from camera.")
            break

        # -------------------------------------------------
        # Frame processing placeholder
        # -------------------------------------------------
        #
        # YOLO inference will be integrated here later.
        #
        # Example:
        #
        # results = model.predict(
        #     frame,
        #     verbose=False
        # )
        #
        # -------------------------------------------------

        fps_logger.update()

        current_fps = fps_logger.get_fps()

        cv2.putText(
            frame,
            f"FPS: {current_fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    result = fps_logger.summary()

    print("\n" + "=" * 60)
    print("FINAL FPS RESULTS")
    print("=" * 60)

    print(f"Frames processed       : {result['frames']}")
    print(
        f"Elapsed time           : "
        f"{result['elapsed_seconds']:.4f} seconds"
    )
    print(f"Average FPS            : {result['fps']:.2f}")
    print(
        f"Average frame time     : "
        f"{result['average_frame_time_ms']:.2f} ms"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()