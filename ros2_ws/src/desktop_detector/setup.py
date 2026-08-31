from setuptools import find_packages, setup

package_name = "desktop_detector"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="adam",
    maintainer_email="adam@localhost",
    description="Jetson desktop object detection with YOLO and ROS2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "detector_node = desktop_detector.detector_node:main",
        ],
    },
)
