import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


# Launch file for launching the training simulation
def generate_launch_description():
    ld = LaunchDescription()

    package_dir = get_package_share_directory("active_slam_simulations")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    # World file path
    world = os.path.join(
        package_dir,
        "worlds",
        "main_directions.world",
    )

    # Model file path
    model_folder = "turtlebot3_burger"
    model_path = os.path.join(
        get_package_share_directory("active_slam_simulations"),
        "models",
        model_folder,
        "model.sdf",
    )

    # URDF file path (redundant F)
    urdf_file_name = "turtlebot3_burger.urdf"
    urdf_path = os.path.join(
        get_package_share_directory("active_slam_simulations"),
        "robot_descriptions",
        urdf_file_name,
    )

    # We have to parse urdf files for robot state publisher to read
    with open(urdf_path, "r") as infp:
        robot_desc = infp.read()

    # TODO: Update to gazebo harmonic and ros2 jazzy

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

    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    pose = [0, 7]

    # Create state publisher node
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_desc}],
    )

    # Create spawner Node
    robot_spawner = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity",
            "robot",
            "-file",
            model_path,
            "-x",
            str(pose[0]),
            "-y",
            str(pose[1]),
            "-z",
            "0.01",
        ],
        output="screen",
    )
    ld.add_action(robot_state_publisher)
    ld.add_action(robot_spawner)

    return ld
