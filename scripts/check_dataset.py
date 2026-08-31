#!/usr/bin/env python3
"""
Checks:
1. Every image has a matching label.
2. Every label has a matching image.
3. YOLO annotation rows contain five values.
4. Class IDs are within the expected range.
5. Bounding-box values are numeric and normalized.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NUM_CLASSES = 4


def find_image_for_label(label_path, image_folder):
    for extension in IMAGE_EXTENSIONS:
        candidate = image_folder / f"{label_path.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def check_split(split_name):
    image_folder = DATASET_ROOT / "images" / split_name
    label_folder = DATASET_ROOT / "labels" / split_name

    image_files = [
        path for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    label_files = [
        path for path in label_folder.glob("*.txt")
        if path.is_file()
    ]

    errors = 0
    class_counts = [0] * NUM_CLASSES

    for image_path in image_files:
        label_path = label_folder / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"[error] Missing label: {image_path.name}")
            errors += 1
            continue

        with label_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                values = line.split()

                if len(values) != 5:
                    print(
                        f"[error] {label_path.name}, "
                        f"line {line_number}: expected 5 values"
                    )
                    errors += 1
                    continue

                try:
                    class_id = int(values[0])
                    box_values = [float(value) for value in values[1:]]
                except ValueError:
                    print(
                        f"[error] {label_path.name}, "
                        f"line {line_number}: invalid numeric values"
                    )
                    errors += 1
                    continue

                if not 0 <= class_id < NUM_CLASSES:
                    print(
                        f"[error] {label_path.name}, "
                        f"line {line_number}: invalid class ID {class_id}"
                    )
                    errors += 1
                    continue

                if not all(0.0 <= value <= 1.0 for value in box_values):
                    print(
                        f"[error] {label_path.name}, "
                        f"line {line_number}: box values must be within [0, 1]"
                    )
                    errors += 1
                    continue

                if box_values[2] <= 0 or box_values[3] <= 0:
                    print(
                        f"[error] {label_path.name}, "
                        f"line {line_number}: width and height must be positive"
                    )
                    errors += 1
                    continue

                class_counts[class_id] += 1

    for label_path in label_files:
        if find_image_for_label(label_path, image_folder) is None:
            print(f"[error] Orphan label: {label_path.name}")
            errors += 1

    print(f"\n[{split_name}]")
    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")
    print(
        "Objects: "
        f"phone={class_counts[0]}, "
        f"book={class_counts[1]}, "
        f"laptop={class_counts[2]}, "
        f"pen={class_counts[3]}"
    )
    print(f"Errors: {errors}")

    return errors


def main():
    total_errors = 0

    print("========== DATASET CHECK ==========")

    for split_name in ("train", "val", "test"):
        total_errors += check_split(split_name)

    print("\n========== RESULT ==========")

    if total_errors == 0:
        print("Dataset check passed successfully.")
    else:
        print(f"Dataset check failed with {total_errors} error(s).")


if __name__ == "__main__":
    main()
