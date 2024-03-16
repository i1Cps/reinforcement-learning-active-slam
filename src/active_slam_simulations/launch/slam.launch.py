import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from ament_index_python import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

# This package launches the slam_toolbox created and managed by Steve Macenski
# You can create a config file and change the toolbox's settings if you want


def generate_launch_description():
    # Try to launch turtlebot3 gazebo world simulation
    slam = "slam_toolbox"
    start_slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(slam), "launch", "online_async_launch.py"
            )
        ),
        launch_arguments=[{"use_sim_time", "true"}],
    )
    return LaunchDescription([start_slam])
