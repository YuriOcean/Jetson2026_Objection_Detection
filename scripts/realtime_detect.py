from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
        PROJECT_ROOT
        / "runs"
        / "desktop_objects"
        / "weights"
        / "best.pt"
)

VIDEO_DEVICE = 0

OUTPUT_DIR = PROJECT_ROOT / "results" / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection settings
CONFIDENCE_THRESHOLD = 0.45
IMAGE_SIZE = 1280

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30


CUSTOM_COLORS = [
    (255, 190, 120),   
    (120, 220, 200),   
    (150, 180, 255),   
    (255, 180, 200),   
]


USE_GRADIENT_BORDER = True
GRADIENT_WIDTH = 3  


def apply_gradient_border(image, bbox, color, thickness=3):

    x1, y1, x2, y2 = bbox

    for i in range(thickness):
        alpha = 1.0 - (i / thickness) * 0.5  
        current_color = tuple(int(c * alpha) for c in color)

        offset = i
        cv2.rectangle(
            image,
            (x1 + offset, y1 + offset),
            (x2 - offset, y2 - offset),
            current_color,
            1
        )

    inner_color = tuple(min(255, c + 40) for c in color)
    cv2.rectangle(
        image,
        (x1 + thickness, y1 + thickness),
        (x2 - thickness, y2 - thickness),
        inner_color,
        1
    )


def set_custom_colors(model, colors):

    num_classes = len(model.names)

    if len(colors) < num_classes:
        colors = colors * (num_classes // len(colors) + 1)

    model.model.names = model.names 

    from ultralytics.utils.plotting import colors as ultralytics_colors

    palette = []
    for i in range(num_classes):
        if i < len(colors):
            color_rgb = (colors[i][2], colors[i][1], colors[i][0])
            palette.append(color_rgb)
        else:
            palette.append((255, 0, 0))

    model.model.names = model.names

    return model


# Main program


def main():
    print("=" * 60)
    print("JETSON REAL-TIME DESKTOP OBJECT DETECTION")
    print("=" * 60)

    print(f"Model path : {MODEL_PATH}")
    print(f"Camera     : /dev/video{VIDEO_DEVICE}")
    print(f"Confidence : {CONFIDENCE_THRESHOLD}")
    print("Classes    : phone, book, laptop, pen")
    print("Color scheme: Morandi (custom)")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model was not found:\n{MODEL_PATH}"
        )

    print("\nLoading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    set_custom_colors(model, CUSTOM_COLORS)

    class_names = list(model.names.values())
    print(f"Classes: {class_names}")
    print(f"Colors (BGR): {CUSTOM_COLORS[:len(class_names)]}")

    print("Model loaded successfully.")


    print("\nOpening USB camera...")

    camera = cv2.VideoCapture(VIDEO_DEVICE, cv2.CAP_V4L2)

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not camera.isOpened():
        raise RuntimeError(
            f"Cannot open /dev/video{VIDEO_DEVICE}"
        )

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(
        f"Camera opened: "
        f"{actual_width}x{actual_height}, "
        f"{actual_fps:.1f} FPS"
    )

    # Create output video
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    video_path = OUTPUT_DIR / f"detection_{timestamp}.avi"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video_writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        20.0,
        (actual_width, actual_height)
    )

    print(f"Recording result video to:")
    print(video_path)

    print("\nPress 'q' to stop detection.")
    print("Press 's' to save the current frame.\n")

    # FPS variables

    previous_time = time.perf_counter()
    fps_history = []

    frame_index = 0

    # Real-time detection loop
    while True:

        success, frame = camera.read()

        if not success:
            print("Warning: failed to read camera frame.")
            continue

        frame_index += 1


        results = model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=0,
            verbose=False
        )

        result = results[0]

        annotated_frame = frame.copy()

        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:

            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for i, (bbox, cls_id, conf) in enumerate(zip(xyxy, cls_ids, confs)):
                x1, y1, x2, y2 = bbox
                class_name = result.names[cls_id]


                color_idx = cls_id % len(CUSTOM_COLORS)
                color = CUSTOM_COLORS[color_idx]

                if USE_GRADIENT_BORDER:

                    apply_gradient_border(annotated_frame, (x1, y1, x2, y2), color, GRADIENT_WIDTH)
                else:
   
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)


                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.08, annotated_frame, 0.92, 0, annotated_frame)

  
                label = f"{class_name} {conf:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                label_y1 = max(y1 - label_height - 10, 0)
                label_y2 = y1

                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y1 - 5),
                    (x1 + label_width + 10, label_y2 + 5),
                    color,
                    -1
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )


        current_time = time.perf_counter()

        elapsed = current_time - previous_time

        if elapsed > 0:
            current_fps = 1.0 / elapsed
            fps_history.append(current_fps)

        previous_time = current_time

        if len(fps_history) > 30:
            fps_history.pop(0)

        average_fps = (
            sum(fps_history) / len(fps_history)
            if fps_history else 0.0
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {average_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (120, 200, 140),  
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_index}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )

        legend_y = actual_height - 20
        for idx, (class_name, color) in enumerate(zip(result.names.values(), CUSTOM_COLORS)):
            if idx >= 4:
                break
            x_pos = 20 + idx * 150
            # 小色块
            cv2.rectangle(
                annotated_frame,
                (x_pos, legend_y - 15),
                (x_pos + 20, legend_y + 5),
                color,
                -1
            )
            cv2.putText(
                annotated_frame,
                class_name,
                (x_pos + 25, legend_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )


        video_writer.write(annotated_frame)


        cv2.imshow(
            "Jetson Desktop Object Detection",
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nDetection stopped by user.")
            break

        elif key == ord("s"):

            image_path = (
                    OUTPUT_DIR
                    / f"snapshot_{timestamp}_{frame_index}.jpg"
            )

            cv2.imwrite(
                str(image_path),
                annotated_frame
            )

            print(f"Snapshot saved: {image_path}")

    camera.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("REAL-TIME DETECTION FINISHED")
    print("=" * 60)

    print(f"Result video: {video_path}")


if __name__ == "__main__":
    main()
