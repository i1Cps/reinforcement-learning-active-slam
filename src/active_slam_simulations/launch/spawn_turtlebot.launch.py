import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the urdf file
    model_folder = "turtlebot3_burger"

    urdf_path = os.path.join(
        get_package_share_directory("active_slam_simulations"),
        "models",
        model_folder,
        "model.sdf",
    )

    # Launch configuration variables specific to simulation
    x_pose = LaunchConfiguration("x_pose", default="0.0")
    y_pose = LaunchConfiguration("y_pose", default="0.0")

    # Declare the launch arguments
    declare_x_position_cmd = DeclareLaunchArgument(
        "x_pose", default_value="0.0", description="Specify namespace of the robot"
    )

    declare_y_position_cmd = DeclareLaunchArgument(
        "y_pose", default_value="0.0", description="Specify namespace of the robot"
    )

    start_gazebo_ros_spawner_cmd = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity",
            model_folder,
            "-file",
            urdf_path,
            "-x",
            x_pose,
            "-y",
            y_pose,
            "-z",
            "0.01",
        ],
        output="screen",
    )

    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_x_position_cmd)
    ld.add_action(declare_y_position_cmd)

    # Add any conditioned actions
    ld.add_action(start_gazebo_ros_spawner_cmd)

    return ld
