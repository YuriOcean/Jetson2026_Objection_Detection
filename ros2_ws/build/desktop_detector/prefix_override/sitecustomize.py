import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/adam/Team9/yer/jetson2026/ros2_ws/install/desktop_detector'
