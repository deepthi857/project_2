import os
import importlib

REQUIRED_FOLDERS = [
    "data/raw/NEU-DET/train/annotations",
    "data/raw/NEU-DET/validation/annotations",
    "src/data",
    "configs",
]

REQUIRED_PACKAGES = [
    "torch",
    "ultralytics",
    "cv2",
    "albumentations",
]


def check_folders():
    print("Checking folder structure...")
    all_ok = True
    for folder in REQUIRED_FOLDERS:
        if not os.path.isdir(folder):
            print(f"  -> Missing folder: {folder}")
            all_ok = False
    if all_ok:
        print("  -> All required folders are present.")
    return all_ok


def check_packages():
    print("Checking required packages...")
    all_ok = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"  -> Missing package: {package}")
            all_ok = False
    if all_ok:
        print("  -> All required packages are installed.")
    return all_ok


if __name__ == "__main__":
    folders_ok = check_folders()
    packages_ok = check_packages()

    print("\nSummary:")
    if folders_ok and packages_ok:
        print("  -> All required folders and packages are present.")
    else:
        print("  -> Some required folders or packages are missing. Please address the issues above.")