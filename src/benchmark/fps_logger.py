import time


class FPSLogger:
    """
    Measures real-time inference performance.

    FPS = number of processed frames / elapsed time
    """

    def __init__(self):
        self.start_time = None
        self.frame_count = 0

    def start(self):
        """Start the FPS timer."""
        self.start_time = time.perf_counter()
        self.frame_count = 0

    def update(self):
        """Record one processed frame."""
        if self.start_time is None:
            raise RuntimeError(
                "FPSLogger has not been started. Call start() first."
            )

        self.frame_count += 1

    def elapsed_time(self):
        """Return elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError(
                "FPSLogger has not been started. Call start() first."
            )

        return time.perf_counter() - self.start_time

    def get_fps(self):
        """Return the current frames-per-second value."""
        elapsed = self.elapsed_time()

        if elapsed <= 0:
            return 0.0

        return self.frame_count / elapsed

    def get_average_frame_time_ms(self):
        """Return average processing time per frame in milliseconds."""
        fps = self.get_fps()

        if fps <= 0:
            return 0.0

        return 1000.0 / fps

    def summary(self):
        """Return the current benchmark statistics."""
        elapsed = self.elapsed_time()

        if elapsed <= 0:
            fps = 0.0
        else:
            fps = self.frame_count / elapsed

        average_ms = (
            1000.0 / fps
            if fps > 0
            else 0.0
        )

        return {
            "frames": self.frame_count,
            "elapsed_seconds": elapsed,
            "fps": fps,
            "average_frame_time_ms": average_ms,
        }


if __name__ == "__main__":

    logger = FPSLogger()

    logger.start()

    # Demonstration only.
    # In the real application, update() will be called
    # after each processed video frame.

    for _ in range(10):
        time.sleep(0.01)
        logger.update()

    result = logger.summary()

    print("=" * 50)
    print("FPS LOGGER TEST")
    print("=" * 50)

    print(f"Frames processed       : {result['frames']}")
    print(f"Elapsed time           : {result['elapsed_seconds']:.4f} seconds")
    print(f"FPS                    : {result['fps']:.2f}")
    print(
        f"Average frame time     : "
        f"{result['average_frame_time_ms']:.2f} ms"
    )