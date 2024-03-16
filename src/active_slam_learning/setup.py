from setuptools import find_packages, setup
import os
from glob import glob

package_name = "active_slam_learning"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Theo Moore-Calters",
    maintainer_email="Theo.MooreCalters@gmail.com",
    description="This Package implements the learning to our robots, specifically reinforcement learning. It contains the environment logic, reward system, and RL algorithms that we can use to train our mobile agents",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "learning_ddpg = active_slam_learning.learning.learning_ddpg:main",
            "learning_environment = active_slam_learning.learning_environment.learning_environment:main",
            "gazebo_environment = active_slam_learning.gazebo_environment.gazebo_environment:main",
        ],
    },
)
