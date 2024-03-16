import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


# Launch file for launching the first world training simulation environment
def generate_launch_description():
    package_dir = get_package_share_directory("active_slam_simulations")
    launch_file_dir = os.path.join(package_dir, "launch")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    x_pose = LaunchConfiguration("x_pose", default="-2.0")
    y_pose = LaunchConfiguration("y_pose", default="-0.5")

    # World file path
    world = os.path.join(
        package_dir,
        "worlds",
        "first_world.world",
    )

    # TODO: Update to gazebo ignite
    # Handles gazebo_ros
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
    )

    # Launches the Robot State Publisher, this reads in URDF form robot descriptions
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # Change this to custom launch file for custom robots ------------ <<<<<<<<----------------------------
    # Launches a node that spawns turtle bot into gazebo world, this uses direct robot models from.... model
    spawn_robot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, "spawn_turtlebot.launch.py")
        ),
        launch_arguments={"x_pose": x_pose, "y_pose": y_pose}.items(),
    )

    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_robot_cmd)

    return ld
