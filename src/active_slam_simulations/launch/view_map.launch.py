from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from ament_index_python import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [
                        get_package_share_directory("active_slam_simulations"),
                        "config",
                        "map.rviz",
                    ]
                ),
                description="Full path to the Rviz configuration file",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config")],
            ),
        ]
    )
