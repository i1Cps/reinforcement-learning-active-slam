import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
from active_slam_interfaces.srv import ResetGazeboEnv
import math
from geometry_msgs.msg import Pose, Point, Quaternion
from slam_toolbox.srv import Reset
from active_slam_learning.common.settings import ROBOT_NAME


class GazeboBridge(Node):
    """
    Node that handles direct communication with Gazebo services and the training environment.
    This node controls the spawning and movement of goal models and manages the robot's state in the environment.
    """

    def __init__(self):
        super().__init__("gazebo_bridge")

        self.initialise_publishers()
        self.initialise_clients()
        self.initialise_services()

        self.setup_constants()
        self.setup_variables()

    def initialise_publishers(self) -> None:
        self.goal_position_publisher = self.create_publisher(Pose, "/goal_position", 10)

    def initialise_clients(self) -> None:
        self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.pause_gazebo_client = self.create_client(Empty, "/pause_physics")
        self.unpause_gazebo_client = self.create_client(Empty, "/unpause_physics")
        self.set_entity_state_client = self.create_client(
            SetEntityState, "/set_entity_state"
        )
        self.slam_reset_client = self.create_client(Reset, "/slam_toolbox/reset")

    def initialise_services(self) -> None:
        self.gazebo_bridge_success_service = self.create_service(
            Empty,
            "/gazebo_bridge_success",
            self.gazebo_bridge_success_callback,
        )
        self.gazebo_bridge_reset_service = self.create_service(
            ResetGazeboEnv,
            "/gazebo_bridge_reset",
            self.gazebo_bridge_reset_callback,
        )
        self.gazebo_bridge_pause_service = self.create_service(
            Empty, "/gazebo_bridge_pause", self.gazebo_bridge_pause_callback
        )
        self.gazebo_bridge_unpause_service = self.create_service(
            Empty,
            "/gazebo_bridge_unpause",
            self.gazebo_bridge_unpause_callback,
        )
        self.gazebo_bridge_init_service = self.create_service(
            Empty, "/gazebo_bridge_init", self.init_callback
        )

    def setup_constants(self) -> None:
        self.RIGHT = Quaternion(z=0.0, w=1.0)  # No rotation
        self.DOWN = Quaternion(
            z=math.sin(math.pi / 4), w=math.cos(math.pi / 4)
        )  # 90 degrees rotation around Z
        self.LEFT = Quaternion(
            z=math.sin(math.pi / 2), w=math.cos(math.pi / 2)
        )  # 180 degrees rotation around Z
        self.UP = Quaternion(
            z=math.sin(3 * math.pi / 4), w=math.cos(3 * math.pi / 4)
        )  # 270 degrees rotation around Z
        self.UP = Quaternion(z=math.sin(math.pi / 4), w=math.cos(math.pi / 4))
        self.setup_model_poses()
        self.GOAL_ENTITY_NAME = "goal_pad"
        self.GOAL_ENTITY = os.path.join(
            get_package_share_directory("active_slam_simulations"),
            "models",
            "goal_pad",
            "model.sdf",
        )
        self.ROBOT_NAME = ROBOT_NAME

    def setup_model_poses(self) -> None:
        # Possible spawn locations for the robot
        self.ROBOT_POSITIONS = [
            Point(x=-2.5, y=3.5, z=0.001),
            Point(x=3.0, y=4.0, z=0.001),
            Point(x=0.0, y=2.0, z=0.001),
            Point(x=-4.3, y=0.0, z=0.001),
            Point(x=4.0, y=0.0, z=0.001),
            Point(x=0.5, y=-2.0, z=0.001),
            Point(x=-2.0, y=-2.5, z=0.001),
            Point(x=-4.5, y=-3.5, z=0.001),
            Point(x=0.0, y=-4.0, z=0.001),
            Point(x=4.0, y=-4.0, z=0.001),
        ]

        # Possible spawn orientation for each spawn location
        self.ORIENTATIONS = [
            [self.UP],
            [self.RIGHT],
            [self.DOWN],
            [self.UP, self.DOWN],
            [self.RIGHT],
            [self.RIGHT],
            [self.UP],
            [self.UP],
            [self.RIGHT, self.LEFT],
            [self.UP, self.LEFT],
        ]

        # Possible spawn locations for the goal
        self.GOAL_POSITIONS = [
            Point(x=-4.62856, y=3.5, z=0.001),
            Point(x=0.95, y=3.842, z=0.001),
            Point(x=-1.0, y=-2.83, z=0.001),
            Point(x=-1.23, y=2.99, z=0.001),
            Point(x=2.0, y=1.53, z=0.001),
            Point(x=2.89, y=2.31, z=0.001),
            Point(x=-3.0, y=1.0, z=0.001),
            Point(x=-2.9, y=-0.804, z=0.001),
            Point(x=-0.347, y=-0.3895, z=0.001),
            Point(x=1.864, y=0.612, z=0.001),
            Point(x=-2.77, y=-3.097, z=0.001),
            Point(x=2.55, y=-2.62, z=0.001),
        ]

        # Redinfed for bigger goal model, {working some things out}
        self.GOAL_POSITIONS = [
            Point(x=1.9946, y=1.0198, z=0.001),
            Point(x=-2.399, y=1.411, z=0.001),
            Point(x=-0.876, y=4.17354, z=0.001),
            Point(x=-0.62, y=-0.95866, z=0.001),
            Point(x=-2.7465, y=-1.0415, z=0.001),
        ]

    def setup_variables(self) -> None:
        self.previous_goal_pose: Pose = Pose(position=self.GOAL_POSITIONS[0])
        self.current_goal_pose: Pose = Pose(position=self.GOAL_POSITIONS[0])
        self.start_position: Pose = Pose(position=self.ROBOT_POSITIONS[0])

        self.goal_counter = 0
        self.collision_counter = 0

        self.first_episode = True

    # Service callback to initialise the Node and generate goal position at the start of the training.
    def init_callback(self, request, response):
        self._generate_new_goal_position(init=True)

        self.get_logger().info("Sucessfully initialised custom Gazebo Bridge Node")
        self.get_logger().info(
            f"Initial goal pose: [{self.current_goal_pose.position.x}, {self.current_goal_pose.position.y}]"
        )
        return response

    # Service callback to reset the robot position and SLAM map after a collision or episode end.
    def gazebo_bridge_reset_callback(self, request, response):
        pose = self._reset_robot_pose()
        response.pose = pose
        self._reset_slam_map()
        if self.first_episode:
            self.first_episode = False
            return response

        if request.collision:
            self.collision_counter += 1
            self.get_logger().info(
                "\n Robot has crashed, Resetting Gazebo, collision count is: {}".format(
                    self.collision_counter
                )
            )
        else:
            self.get_logger().info("Episode timeout, Resetting Gazebo")

        return response

    # Service callback to handle the event when the robot successfully finds the goal.
    def gazebo_bridge_success_callback(self, request, response):
        self._generate_new_goal_position()
        self.goal_counter += 1
        self.get_logger().info(
            "\n Robot found a goal!, goal count is: {}, New goal position: [ {:.2f}, {:.2f} ]".format(
                self.goal_counter,
                self.current_goal_pose.position.x,
                self.current_goal_pose.position.y,
            )
        )
        return response

    # Generate a new random robot pose.
    def _generate_new_robot_poses(self):
        # Convert positions and orientations to NumPy arrays for efficient indexing
        positions = np.array(self.ROBOT_POSITIONS)
        orientations = self.ORIENTATIONS

        # Randomly select a position and orientation
        position_index = np.random.choice(len(positions))
        orientation_index = np.random.choice(len(orientations[position_index]))

        sampled_pose = Pose(
            position=positions[position_index],
            orientation=orientations[position_index][orientation_index],
        )

        return sampled_pose

    # Reset the robot's position to a new random pose.
    def _reset_robot_pose(self) -> Pose:
        req = SetEntityState.Request()
        pose = self._generate_new_robot_poses()
        req.state.name = self.ROBOT_NAME
        req.state.pose = pose

        self._wait_for_service(self.set_entity_state_client, "Set Entity State")
        self.set_entity_state_client.call_async(req)
        return pose

    # Generate a new goal position different from the previous one.
    def _generate_new_goal_position(self, init: bool = False) -> None:
        self.previous_goal_pose = self.current_goal_pose
        while self.previous_goal_pose == self.current_goal_pose:
            index = np.random.choice(len(self.GOAL_POSITIONS), replace=False)
            self.current_goal_pose = Pose(position=self.GOAL_POSITIONS[index])
        self._publish_new_goal(init)

    # Reset the map created using the SLAM toolbox
    def _reset_slam_map(self) -> None:
        time.sleep(2)
        self._wait_for_service(self.slam_reset_client, "SLAM reset")
        self.slam_reset_client.call_async(Reset.Request())

    # Pause gazebo engine
    def gazebo_bridge_pause_callback(self, request, response):
        self._wait_for_service(self.unpause_gazebo_client, "Unpause Gazebo")
        self.pause_gazebo_client.call_async(Empty.Request())
        return response

    # Unpause gazebo engine
    def gazebo_bridge_unpause_callback(self, request, response):
        self._wait_for_service(self.pause_gazebo_client, "Pause Gazebo")
        self.unpause_gazebo_client.call_async(Empty.Request())
        return response

    # Publish the goal position to the learning environment node so it can track whether the agent reaches said goal
    def _publish_new_goal(self, init=False):
        self.goal_position_publisher.publish(self.current_goal_pose)
        if init:
            self._spawn_goal_entity()
        else:
            self._move_goal_entity()

    # Spawn a new goal entity in Gazebo at the current goal position.
    def _spawn_goal_entity(self):
        req = SpawnEntity.Request()
        req.name = self.GOAL_ENTITY_NAME
        req.xml = open(self.GOAL_ENTITY, "r").read()
        req.initial_pose = self.current_goal_pose
        self._wait_for_service(self.spawn_entity_client, "Spawn Entity")
        self.spawn_entity_client.call_async(req)

    # Move the goal entity in Gazebo to the current goal position.
    def _move_goal_entity(self):
        req = SetEntityState.Request()
        req.state.name = self.GOAL_ENTITY_NAME
        req.state.pose = self.current_goal_pose
        self._wait_for_service(self.set_entity_state_client, "Set Entity State")
        self.set_entity_state_client.call_async(req)

    def _quaternion_to_theta(self, orientation):
        # Extract the quaternion components
        w, x, y, z = orientation.w, orientation.x, orientation.y, orientation.z

        # Compute yaw angle (rotation around z-axis)
        theta = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

        # Ensure theta is in the range [-pi, pi]
        theta = math.atan2(math.sin(theta), math.cos(theta))

        return theta

    def _wait_for_service(self, client, service_name: str) -> None:
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"{service_name} client not available, waiting...")


def main(args=None):
    rclpy.init(args=args)
    gazebo = GazeboBridge()
    rclpy.spin(gazebo)
    gazebo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
