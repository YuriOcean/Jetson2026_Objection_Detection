#!/usr/bin/env python3
"""
Classes:
    0 - phone
    1 - book
    2 - laptop
    3 - pen
"""

from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"


def main():
    print("=" * 50)
    print("JETSON DESKTOP OBJECT DETECTION TRAINING")
    print("=" * 50)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset yaml : {DATA_CONFIG}")
    print(f"Output root  : {RUNS_DIR}")

    # YOLO11 nano is lightweight and suitable for Jetson Orin NX.
    # Pretrained weights are used as the starting point.
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATA_CONFIG),

        epochs=100,
        imgsz=640,
        batch=8,

        device=0,
        workers=2,

        project=str(RUNS_DIR),
        name="desktop_objects",

        pretrained=True,
        patience=30,

        optimizer="auto",
        seed=2026,

        save=True,
        plots=True,
        verbose=True,
    )

    print("=" * 50)
    print("TRAINING FINISHED")
    print("=" * 50)

    print(f"Results saved under: {RUNS_DIR / 'desktop_objects'}")

    return results


if __name__ == "__main__":
    main()
