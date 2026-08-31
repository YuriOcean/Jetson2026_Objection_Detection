#!/usr/bin/env python3
"""
    70% training
    20% validation
    10% testing
"""

from pathlib import Path
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "dataset_raw"
DATASET_ROOT = PROJECT_ROOT / "dataset"

CLASSES = ["phone", "book", "laptop", "pen"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

RANDOM_SEED = 2026
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20


def prepare_output_folders():
    """Create clean output folders for every dataset split."""
    for split_name in ("train", "val", "test"):
        for data_type in ("images", "labels"):
            target = DATASET_ROOT / data_type / split_name

            if target.exists():
                shutil.rmtree(target)

            target.mkdir(parents=True, exist_ok=True)


def find_valid_pairs(class_name):
    """Find images that have a matching YOLO label file."""
    class_folder = RAW_ROOT / class_name

    if not class_folder.exists():
        raise FileNotFoundError(f"Class folder not found: {class_folder}")

    pairs = []

    for image_path in sorted(class_folder.iterdir()):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = image_path.with_suffix(".txt")

        if not label_path.exists():
            print(f"[warning] Missing label: {image_path.name}")
            continue

        pairs.append((image_path, label_path))

    return pairs


def copy_pair(image_path, label_path, split_name):
    """Copy an image and its matching label."""
    shutil.copy2(
        image_path,
        DATASET_ROOT / "images" / split_name / image_path.name
    )

    shutil.copy2(
        label_path,
        DATASET_ROOT / "labels" / split_name / label_path.name
    )


def split_one_class(class_name):
    """Shuffle and split one class independently."""
    pairs = find_valid_pairs(class_name)

    if not pairs:
        raise RuntimeError(f"No valid pairs found for class: {class_name}")

    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    print(f"\n[class] {class_name}: total={total}")

    for split_name, split_pairs in splits.items():
        for image_path, label_path in split_pairs:
            copy_pair(image_path, label_path, split_name)

        print(f"  {split_name}: {len(split_pairs)}")


def main():
    random.seed(RANDOM_SEED)
    prepare_output_folders()

    print("========== DATASET SPLIT ==========")

    for class_name in CLASSES:
        split_one_class(class_name)

    print("\n========== COMPLETED ==========")


if __name__ == "__main__":
    main()
