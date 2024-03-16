import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Package is responsible for launching the Robot State Publisher


# TODO: Read in a URDF file path from a launch configuration, currently hardcoded tb3
def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    urdf_file_name = "turtlebot3_burger.urdf"

    print("urdf_file_name : {}".format(urdf_file_name))

    urdf_path = os.path.join(
        get_package_share_directory("active_slam_simulations"),
        "robot_descriptions",
        urdf_file_name,
    )

    # We have to parse urdf files for robot state publisher to read
    with open(urdf_path, "r") as infp:
        robot_desc = infp.read()

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {"use_sim_time": use_sim_time, "robot_description": robot_desc}
                ],
            ),
        ]
    )
