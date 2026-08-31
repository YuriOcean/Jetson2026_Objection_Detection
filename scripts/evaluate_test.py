from pathlib import Path
from ultralytics import YOLO



PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "desktop_objects" / "weights" / "best.pt"
DATASET_YAML = PROJECT_ROOT / "configs" / "dataset.yaml"


def main():
    print("=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    print(f"Model: {MODEL_PATH}")
    print(f"Dataset config: {DATASET_YAML}")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")

    model = YOLO(str(MODEL_PATH))

    # Evaluate only the independent test split.
    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        project=str(PROJECT_ROOT / "results"),
        name="test_evaluation",
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

