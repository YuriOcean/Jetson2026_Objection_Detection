from pathlib import Path
import csv
import json
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ultralytics import YOLO


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path("/home/adam/Team9/yer/jetson2026")

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "desktop_objects"
    / "weights"
    / "best.pt"
)

VIDEO_DEVICE = 0

CONFIDENCE_THRESHOLD = 0.45
IMAGE_SIZE = 1280

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30


# ============================================================
# Result directories
# ============================================================

RESULT_ROOT = PROJECT_ROOT / "results"

VIDEO_DIR = RESULT_ROOT / "videos"
SNAPSHOT_DIR = RESULT_ROOT / "snapshots"
ERROR_DIR = RESULT_ROOT / "error_cases"
TEST_DIR = RESULT_ROOT / "test_results"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Custom detection colors
# OpenCV uses BGR format
# phone / book / laptop / pen
# ============================================================

CUSTOM_COLORS = [
    (255, 190, 120),
    (120, 220, 200),
    (150, 180, 255),
    (255, 180, 200),
]

USE_GRADIENT_BORDER = True
GRADIENT_WIDTH = 3


# ============================================================
# Drawing utilities
# ============================================================

def apply_gradient_border(image, bbox, color, thickness=3):
    """
    Draw a layered gradient-style border around one detection box.
    """

    x1, y1, x2, y2 = bbox

    for i in range(thickness):

        alpha = 1.0 - (i / thickness) * 0.5

        current_color = tuple(
            int(channel * alpha)
            for channel in color
        )

        cv2.rectangle(
            image,
            (x1 + i, y1 + i),
            (x2 - i, y2 - i),
            current_color,
            1,
        )

    inner_color = tuple(
        min(255, channel + 40)
        for channel in color
    )

    cv2.rectangle(
        image,
        (x1 + thickness, y1 + thickness),
        (x2 - thickness, y2 - thickness),
        inner_color,
        1,
    )


# ============================================================
# ROS2 detector node
# ============================================================

class DesktopDetectorNode(Node):

    def __init__(self):

        super().__init__("desktop_detector")

        # ----------------------------------------------------
        # ROS2 publisher
        # ----------------------------------------------------

        self.publisher_ = self.create_publisher(
            String,
            "/desktop_detections",
            10,
        )

        self.get_logger().info(
            "Desktop detector node is starting..."
        )

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        self.get_logger().info(
            f"Loading YOLO model: {MODEL_PATH}"
        )

        if not MODEL_PATH.exists():

            self.get_logger().error(
                f"Model file not found: {MODEL_PATH}"
            )

            raise FileNotFoundError(str(MODEL_PATH))

        self.model = YOLO(str(MODEL_PATH))

        self.class_names = self.model.names

        self.get_logger().info(
            f"YOLO model loaded successfully: "
            f"{list(self.class_names.values())}"
        )

        # ----------------------------------------------------
        # Open camera
        # ----------------------------------------------------

        self.camera = cv2.VideoCapture(
            VIDEO_DEVICE,
            cv2.CAP_V4L2,
        )

        self.camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )

        self.camera.set(
            cv2.CAP_PROP_FPS,
            CAMERA_FPS,
        )

        if not self.camera.isOpened():

            self.get_logger().error(
                f"Cannot open camera /dev/video{VIDEO_DEVICE}"
            )

            raise RuntimeError(
                f"Cannot open camera /dev/video{VIDEO_DEVICE}"
            )

        self.actual_width = int(
            self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.actual_height = int(
            self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.actual_fps = self.camera.get(
            cv2.CAP_PROP_FPS
        )

        self.get_logger().info(
            f"Camera opened: "
            f"{self.actual_width}x{self.actual_height}, "
            f"{self.actual_fps:.1f} FPS"
        )

        # ----------------------------------------------------
        # Create timestamp for this experiment
        # ----------------------------------------------------

        self.timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        # ----------------------------------------------------
        # Create result video
        # ----------------------------------------------------

        self.video_path = (
            VIDEO_DIR
            / f"detection_{self.timestamp}.avi"
        )

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            20.0,
            (
                self.actual_width,
                self.actual_height,
            ),
        )

        # ----------------------------------------------------
        # Create detection CSV log
        # ----------------------------------------------------

        self.csv_path = (
            TEST_DIR
            / f"detection_log_{self.timestamp}.csv"
        )

        self.csv_file = open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.csv_writer = csv.writer(
            self.csv_file
        )

        self.csv_writer.writerow([
            "frame",
            "timestamp",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ])

        # ----------------------------------------------------
        # FPS state
        # ----------------------------------------------------

        self.previous_time = time.perf_counter()

        self.fps_history = []

        self.frame_index = 0

        # ----------------------------------------------------
        # Error-case state
        # ----------------------------------------------------

        self.last_error_save_time = 0.0

        # ----------------------------------------------------
        # ROS2 timer
        # ----------------------------------------------------

        self.timer = self.create_timer(
            0.001,
            self.process_frame,
        )

        self.get_logger().info(
            "Node started successfully."
        )

        self.get_logger().info(
            "ROS2 topic: /desktop_detections"
        )

        self.get_logger().info(
            f"Result video: {self.video_path}"
        )

        self.get_logger().info(
            f"Detection log: {self.csv_path}"
        )

        print()
        print("=" * 60)
        print("JETSON ROS2 DESKTOP OBJECT DETECTION")
        print("=" * 60)
        print(f"Model      : {MODEL_PATH}")
        print(f"Camera     : /dev/video{VIDEO_DEVICE}")
        print(f"Confidence : {CONFIDENCE_THRESHOLD}")
        print("Classes    : phone, book, laptop, pen")
        print("ROS2 Topic : /desktop_detections")
        print()
        print("Press 's' to save a snapshot.")
        print("Press 'q' to stop.")
        print("=" * 60)
        print()

    # ========================================================
    # Main frame processing
    # ========================================================

    def process_frame(self):

        success, frame = self.camera.read()

        if not success:

            self.get_logger().warning(
                "Failed to read camera frame."
            )

            return

        self.frame_index += 1

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = self.model(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=0,
            verbose=False,
        )

        result = results[0]

        annotated_frame = frame.copy()

        boxes = result.boxes

        detection_list = []

        # ----------------------------------------------------
        # Draw and collect detections
        # ----------------------------------------------------

        if boxes is not None and len(boxes) > 0:

            xyxy = (
                boxes.xyxy.cpu()
                .numpy()
                .astype(int)
            )

            cls_ids = (
                boxes.cls.cpu()
                .numpy()
                .astype(int)
            )

            confs = (
                boxes.conf.cpu()
                .numpy()
            )

            for bbox, cls_id, conf in zip(
                xyxy,
                cls_ids,
                confs,
            ):

                x1, y1, x2, y2 = bbox

                class_name = result.names[cls_id]

                color_idx = (
                    cls_id
                    % len(CUSTOM_COLORS)
                )

                color = CUSTOM_COLORS[color_idx]

                # ------------------------------------------------
                # Gradient border
                # ------------------------------------------------

                if USE_GRADIENT_BORDER:

                    apply_gradient_border(
                        annotated_frame,
                        (x1, y1, x2, y2),
                        color,
                        GRADIENT_WIDTH,
                    )

                else:

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                # ------------------------------------------------
                # Transparent box fill
                # ------------------------------------------------

                overlay = annotated_frame.copy()

                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    color,
                    -1,
                )

                cv2.addWeighted(
                    overlay,
                    0.08,
                    annotated_frame,
                    0.92,
                    0,
                    annotated_frame,
                )

                # ------------------------------------------------
                # Label
                # ------------------------------------------------

                label = (
                    f"{class_name} {conf:.2f}"
                )

                (
                    label_width,
                    label_height,
                ), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2,
                )

                label_y1 = max(
                    y1 - label_height - 10,
                    0,
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y1 - 5),
                    (
                        x1 + label_width + 10,
                        y1 + 5,
                    ),
                    color,
                    -1,
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # ------------------------------------------------
                # Save one detection record
                # ------------------------------------------------

                detection = {
                    "class": class_name,
                    "confidence": round(
                        float(conf),
                        4,
                    ),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                }

                detection_list.append(
                    detection
                )

                self.csv_writer.writerow([
                    self.frame_index,
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    class_name,
                    f"{conf:.4f}",
                    x1,
                    y1,
                    x2,
                    y2,
                ])

        # ----------------------------------------------------
        # Flush CSV
        # ----------------------------------------------------

        self.csv_file.flush()

        # ----------------------------------------------------
        # Calculate FPS
        # ----------------------------------------------------

        current_time = time.perf_counter()

        elapsed = (
            current_time
            - self.previous_time
        )

        if elapsed > 0:

            current_fps = 1.0 / elapsed

            self.fps_history.append(
                current_fps
            )

        self.previous_time = current_time

        if len(self.fps_history) > 30:

            self.fps_history.pop(0)

        average_fps = (
            sum(self.fps_history)
            / len(self.fps_history)
            if self.fps_history
            else 0.0
        )

        # ----------------------------------------------------
        # Add FPS
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"FPS: {average_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (120, 200, 140),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Add frame number
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"Frame: {self.frame_index}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Add ROS2 status
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            "ROS2: /desktop_detections",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Class legend
        # ----------------------------------------------------

        legend_y = (
            self.actual_height - 20
        )

        for idx, (
            class_name,
            color,
        ) in enumerate(
            zip(
                result.names.values(),
                CUSTOM_COLORS,
            )
        ):

            if idx >= 4:
                break

            x_pos = 20 + idx * 150

            cv2.rectangle(
                annotated_frame,
                (
                    x_pos,
                    legend_y - 15,
                ),
                (
                    x_pos + 20,
                    legend_y + 5,
                ),
                color,
                -1,
            )

            cv2.putText(
                annotated_frame,
                class_name,
                (
                    x_pos + 25,
                    legend_y + 5,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Publish ROS2 result
        # ----------------------------------------------------

        message_data = {
            "frame": self.frame_index,
            "timestamp": time.time(),
            "fps": round(
                average_fps,
                2,
            ),
            "count": len(detection_list),
            "detections": detection_list,
        }

        ros_message = String()

        ros_message.data = json.dumps(
            message_data,
            ensure_ascii=False,
        )

        self.publisher_.publish(
            ros_message
        )

        # ----------------------------------------------------
        # Automatic error-case candidate saving
        #
        # Save low-confidence detections as possible
        # difficult/error examples for later inspection.
        # ----------------------------------------------------

        low_confidence = any(
            item["confidence"] < 0.60
            for item in detection_list
        )

        now = time.time()

        if (
            low_confidence
            and now - self.last_error_save_time > 3.0
        ):

            error_path = (
                ERROR_DIR
                / (
                    f"low_conf_"
                    f"{self.timestamp}_"
                    f"frame_{self.frame_index}.jpg"
                )
            )

            cv2.imwrite(
                str(error_path),
                annotated_frame,
            )

            self.last_error_save_time = now

            self.get_logger().info(
                f"Possible difficult case saved: "
                f"{error_path.name}"
            )

        # ----------------------------------------------------
        # Save video
        # ----------------------------------------------------

        self.video_writer.write(
            annotated_frame
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "Jetson ROS2 Desktop Object Detection",
            annotated_frame,
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            snapshot_path = (
                SNAPSHOT_DIR
                / (
                    f"snapshot_"
                    f"{self.timestamp}_"
                    f"frame_{self.frame_index}.jpg"
                )
            )

            cv2.imwrite(
                str(snapshot_path),
                annotated_frame,
            )

            self.get_logger().info(
                f"Snapshot saved: {snapshot_path}"
            )

        elif key == ord("q"):

            self.get_logger().info(
                "Detection stopped by user."
            )

            rclpy.shutdown()

    # ========================================================
    # Cleanup
    # ========================================================

    def destroy_node(self):

        self.get_logger().info(
            "Releasing camera and result files..."
        )

        if hasattr(self, "camera"):

            self.camera.release()

        if hasattr(self, "video_writer"):

            self.video_writer.release()

        if hasattr(self, "csv_file"):

            self.csv_file.close()

        cv2.destroyAllWindows()

        super().destroy_node()


# ============================================================
# ROS2 entry point
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = None

    try:

        node = DesktopDetectorNode()

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        if node is not None:

            node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":
    main()
