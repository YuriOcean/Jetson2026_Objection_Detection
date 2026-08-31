# Jetson Desktop Object Detection

A YOLO-based desktop object detection project deployed on NVIDIA Jetson ONX-DEV with ROS 2.

The system detects four types of desktop objects: phone📱, book📖, laptop💻, and pen🖊️.

## Links
Item	Link
📁 Original Dataset	and Processed Dataset	(to be added)
🎥 Demo Video	(to be added)
📄 Experiment Report	(to be added)

## Project Structure

📂`configs/` Configuration files

📂`dataset/` Training, validation, and test images and YOLO labels

📂`results/` Detection results and test results

📂`ros2_ws/` ROS 2 workspace and `desktop_detector` package

📂`runs/` YOLO training results and model weights

📂`scripts/` Dataset processing, training, camera testing, and detection scripts

📄 `classes.txt` Detection class names

📄 `README.md` Project documentation

## Main Scripts

⚙️`scripts/split_dataset.py` Split the dataset

⚙️`scripts/check_dataset.py` Check the dataset and labels

⚙️`scripts/fix_label_classes.py` Fix label class IDs

⚙️`scripts/train_model.py` Train the YOLO model

⚙️`scripts/realtime_detect.py` Run real-time detection

## Running

Check the dataset

```bash
python3 scripts/check_dataset.py
```

Train the model

```bash
python3 scripts/train_model.py
```

Run real-time detection

```bash
python3 scripts/realtime_detect.py
```

Run ROS 2 detection

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 run desktop_detector detector_node
```

Check ROS 2 topics

```bash
ros2 topic list
```
