from pathlib import Path
import csv


OUTPUT_FILE = Path("benchmark_results.csv")


def calculate_fps(total_images, total_time):
    """
    Calculate throughput in frames/images per second.
    """
    if total_time <= 0:
        return 0.0

    return total_images / total_time


def calculate_average_time_ms(total_images, total_time):
    """
    Calculate average processing time per image/frame in milliseconds.
    """
    if total_images <= 0:
        return 0.0

    return (total_time / total_images) * 1000.0


def calculate_speedup(reference_time, comparison_time):
    """
    Calculate how many times faster the comparison is
    relative to the reference.

    Example:
        reference = 100 ms
        comparison = 50 ms
        speedup = 2.0x
    """
    if comparison_time <= 0:
        return 0.0

    return reference_time / comparison_time


def create_record(
    framework,
    images_processed,
    total_time,
    status="measured",
):
    """
    Create one benchmark result record.
    """

    average_time_ms = calculate_average_time_ms(
        images_processed,
        total_time,
    )

    fps = calculate_fps(
        images_processed,
        total_time,
    )

    return {
        "framework": framework,
        "images_processed": images_processed,
        "total_time_seconds": round(total_time, 4),
        "average_time_ms": round(average_time_ms, 2),
        "fps": round(fps, 2),
        "status": status,
    }


def save_results(records, output_file=OUTPUT_FILE):
    """
    Save benchmark records to a CSV file.
    """

    fieldnames = [
        "framework",
        "images_processed",
        "total_time_seconds",
        "average_time_ms",
        "fps",
        "status",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nBenchmark results saved to: {output_file}")


def print_report(records):
    """
    Print a readable benchmark comparison.
    """

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON REPORT")
    print("=" * 70)

    print(
        f"{'Framework':<15}"
        f"{'Images':<12}"
        f"{'Avg ms':<15}"
        f"{'FPS':<12}"
        f"{'Status':<15}"
    )

    print("-" * 70)

    for record in records:
        print(
            f"{record['framework']:<15}"
            f"{record['images_processed']:<12}"
            f"{record['average_time_ms']:<15.2f}"
            f"{record['fps']:<12.2f}"
            f"{record['status']:<15}"
        )

    print("=" * 70)


def main():

    print("=" * 70)
    print("WEEK 3 - DAY 6")
    print("BENCHMARK RESULT REPORTER")
    print("=" * 70)

    # ---------------------------------------------------------
    # IMPORTANT:
    # These are example measurements used only to test the
    # reporting system. They are NOT model benchmark results.
    # ---------------------------------------------------------

    test_records = [
        create_record(
            framework="PyTorch",
            images_processed=100,
            total_time=10.0,
            status="example",
        ),
        create_record(
            framework="ONNX",
            images_processed=100,
            total_time=7.5,
            status="example",
        ),
        create_record(
            framework="OpenCV",
            images_processed=100,
            total_time=2.0,
            status="example",
        ),
    ]

    print_report(test_records)

    save_results(test_records)


if __name__ == "__main__":
    main()