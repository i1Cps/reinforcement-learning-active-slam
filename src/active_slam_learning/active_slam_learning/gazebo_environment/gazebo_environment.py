import os
import time
import random
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
from slam_toolbox.srv import DeserializePoseGraph
from active_slam_interfaces.srv import ResetGazeboEnv
import math
from geometry_msgs.msg import Pose, Point, Quaternion
from slam_toolbox.srv import Reset


# Node that handles direct communication with gazebo services and training environment, will control spawning of goal model
class GazeboEnvironment(Node):
    def __init__(self):
        super().__init__("gazebo_cli")

        # ---------------------- Publishers ------------------- #

        self.goal_position_reset_pose_publisher = self.create_publisher(
            Pose, "/goal_position_reset_pose", 10
        )
        self.robot_position_reset_pose_publisher = self.create_publisher(
            Pose, "/robot_position_reset_pose", 10
        )

        # ------------------------ subscribers ------------------------- #
        self.shutdown_node_subsriber = self.create_subscription(
            Bool, "/shutdown_rl_nodes", self.shutdown_node, 10
        )

        # Clients
        self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.pause_gazebo_client = self.create_client(Empty, "/pause_physics")
        self.unpause_gazebo_client = self.create_client(Empty, "/unpause_physics")
        self.set_entity_state_client = self.create_client(
            SetEntityState, "/set_entity_state"
        )
        self.slam_reset_client = self.create_client(Reset, "/slam_toolbox/reset")

        # Services
        self.environment_success_service = self.create_service(
            Empty, "/environment_success", self.environment_success_callback
        )

        self.environment_reset_service = self.create_service(
            ResetGazeboEnv, "/environment_reset", self.environment_reset_callback
        )

        self.environment_pause_service = self.create_service(
            Empty, "/environment_pause", self.pause_gazebo_callback
        )

        self.environment_unpause_service = self.create_service(
            Empty, "/environment_unpause", self.unpause_gazebo_callback
        )

        self.initialise_gazebo_environment = self.create_service(
            Empty, "/initialise_gazebo_environment", self.init_callback
        )

        self.get_logger().info("Sucessfully initialised Gazebo Environment Node")

        #############################################################
        #           VARIABLES AND CONSTANTS                         #
        #############################################################

        # Define quaternions for up,down,left,right orientation
        LEFT = Quaternion(z=0.0, w=1.0)
        RIGHT = Quaternion(z=math.sin(math.pi), w=math.cos(math.pi))
        UP = Quaternion(z=math.sin(math.pi / 2), w=math.cos(math.pi / 2))
        DOWN = Quaternion(z=math.sin(-math.pi / 2), w=math.cos(-math.pi / 2))

        # List of positions
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

        # List of available orientations corresponding to each position
        self.ORIENTATIONS = [
            [UP],
            [RIGHT],
            [DOWN],
            [UP, DOWN],
            [RIGHT],
            [RIGHT],
            [UP],
            [UP],
            [RIGHT, LEFT],
            [UP, LEFT],
        ]

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

        self.previous_goal_pose: Pose = Pose(position=self.GOAL_POSITIONS[0])
        self.current_goal_pose: Pose = Pose(position=self.GOAL_POSITIONS[0])
        self.start_position: Pose = Pose(position=self.ROBOT_POSITIONS[0])

        self.goal_entity_name = "goal_pad"
        self.goal_entity = os.path.join(
            get_package_share_directory("active_slam_simulations"),
            "models",
            "goal_pad",
            "model.sdf",
        )
        self.first_episode = True

    def init_callback(self, request, response):
        self._generate_new_goal_position(init=True)
        print(
            "Initial goal pose:",
            self.current_goal_pose.position.x,
            self.current_goal_pose.position.y,
        )
        return response

    def environment_reset_callback(self, request, response):
        pose = self._reset_robot_pose()
        response.pose = pose
        self._reset_slam_map()
        return response

    def environment_success_callback(self, request, response):
        self._generate_new_goal_position()
        self.get_logger().info(
            f"Episode success, New goal position: [ {self.current_goal_pose.position.x}, {self.current_goal_pose.position.y} ]"
        )
        return response

    def _generate_new_robot_poses(self, num_poses=1):
        sampled_poses = []
        # Convert positions and orientations to NumPy arrays for efficient indexing
        positions = np.array(self.ROBOT_POSITIONS)
        orientations = self.ORIENTATIONS

        # Randomly select indices for positions and orientations
        position_indices = np.random.choice(len(positions), size=num_poses)
        # print("position_indices:", position_indices)

        orientation_indices = []
        for idx in position_indices:
            orientation_list = orientations[idx]
            orientation_len = len(orientation_list)
            # print(f"Orientation list length for position {idx}: {orientation_len}")
            # if orientation_len == 0:
            # print(f"Warning: No orientations available for position {idx}")
            orientation_idx = np.random.choice(orientation_len)
            orientation_indices.append(orientation_idx)
        # print("orientation_indices:", orientation_indices)

        # Create poses using the selected indices
        sampled_poses = [
            Pose(position=positions[idx], orientation=orientations[idx][ori_idx])
            for idx, ori_idx in zip(position_indices, orientation_indices)
        ]

        return sampled_poses[0]  # Return only the first sampled pose

    def _reset_robot_pose(self):
        req = SetEntityState.Request()
        pose = self._generate_new_robot_poses()
        req.state.name = "turtlebot3_burger"
        req.state.pose = pose

        while not self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Set Entity State client not available, waiting some more!"
            )
        self.set_entity_state_client.call_async(req)
        self.robot_position_reset_pose_publisher.publish(pose)
        return pose

    def _generate_new_goal_position(self, init=False):
        self.previous_goal_pose = self.current_goal_pose
        while self.previous_goal_pose == self.current_goal_pose:
            index = np.random.choice(len(self.GOAL_POSITIONS), replace=False)
            self.current_goal_pose = Pose(position=self.GOAL_POSITIONS[index])
        self._publish_new_goal(init)

    def _reset_slam_map(self):
        time.sleep(2)
        while not self.slam_reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("slam map reset service not available")
        self.slam_reset_client.call_async(Reset.Request())

    def pause_gazebo_callback(self):
        while not self.pause_gazebo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("pause service")
        self.pause_gazebo_client.call_async(Empty.Request())

    def unpause_gazebo_callback(self):
        while not self.unpause_gazebo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("pause service")
        self.unpause_gazebo_client.call_async(Empty.Request())

    # Publish the goal position to the learning environment node so it can track whether the agent reaches said goal
    def _publish_new_goal(self, init=False):
        self.goal_position_reset_pose_publisher.publish(self.current_goal_pose)
        if init:
            self._spawn_goal_entity()
        else:
            self._move_goal_entity()

    def _spawn_goal_entity(self):
        req = SpawnEntity.Request()
        req.name = self.goal_entity_name
        req.xml = open(self.goal_entity, "r").read()
        req.initial_pose = self.current_goal_pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Spawn Entity client not available, waiting some more!"
            )
        print("spawning")
        self.spawn_entity_client.call_async(req)

    def _move_goal_entity(self):
        req = SetEntityState.Request()
        req.state.name = self.goal_entity_name
        req.state.pose = self.current_goal_pose
        while not self.set_entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Set Entity State client not available, waiting some more!"
            )
        self.set_entity_state_client.call_async(req)

    def _quaternion_to_theta(self, orientation):
        # Extract the quaternion components
        w, x, y, z = orientation.w, orientation.x, orientation.y, orientation.z

        # Compute yaw angle (rotation around z-axis)
        theta = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

        # Ensure theta is in the range [-pi, pi]
        theta = math.atan2(math.sin(theta), math.cos(theta))

        return theta

    def shutdown_node(self, msg):
        if msg.data:
            self.get_logger().info("Shutdown signal received, shutting down...")
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    gazebo = GazeboEnvironment()
    rclpy.spin(gazebo)
    gazebo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
