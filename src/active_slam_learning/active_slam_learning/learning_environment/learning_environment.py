"""
Module for simulating a learning environment in a single-robot system for RL using ROS2.

This module defines a Learning Environment node which facilitates the interaction between
a robot and a simulated environment. It manages the state updates, action processing,
and reward calculation necessary for reinforcement learning experiments.

Author: Theo Moore-Calters
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

import numpy as np
import scipy
import time
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Pose
from active_slam_interfaces.srv import StepEnv, ResetEnv, ResetGazeboEnv

from active_slam_learning.common.settings import (
    NUMBER_OF_SCANS,
    MAX_SCAN_DISTANCE,
    GOAL_PAD_RADIUS,
    MAX_CONTINUOUS_ACTIONS,
    EPISODE_STEPS,
    EPISODE_LENGTH_SEC,
    COLLISION_DISTANCE,
    REWARD_DEBUG,
)
from active_slam_learning.learning_environment.reward_function import reward_function


class LearningEnvironment(Node):
    """
    Node responsible for providing an interface for agents to take actions and receive new states, rewards, or both.
    Contains communication with Bridge Node which talks to the physics simulator, Gazebo.
    """

    def __init__(self):
        super().__init__("learning_environment")

        self._initialise_subscribers()
        self._initialise_publishers()
        self._initialise_clients()
        self._initialise_services()

        self._initialise_constants()
        self._reset_robot_variables()
        self._reset_environment_variables()

        self._initialise_gazebo_bridge_node()

    def _initialise_subscribers(self) -> None:
        self.scan_subscriber = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 1
        )
        self.covariance_matrix_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, "/pose", self.covariance_matrix_callback, 10
        )
        self.goal_position_subscriber = self.create_subscription(
            Pose,
            "/goal_position",
            self.goal_position_callback,
            10,
        )
        self.odom_subscriber = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        qos_clock = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1,
        )
        self.clock_subscriber = self.create_subscription(
            Clock, "/clock", self.clock_callback, qos_profile=qos_clock
        )

    def _initialise_publishers(self) -> None:
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

    def _initialise_clients(self) -> None:
        self.gazebo_bridge_init_client = self.create_client(
            Empty, "/gazebo_bridge_init"
        )
        self.gazebo_bridge_success_client = self.create_client(
            Empty, "/gazebo_bridge_success"
        )
        self.gazebo_bridge_reset_client = self.create_client(
            ResetGazeboEnv, "/gazebo_bridge_reset"
        )
        self.gazebo_bridge_pause_client = self.create_client(
            Empty, "/gazebo_bridge_pause"
        )
        self.gazebo_bridge_unpause_client = self.create_client(
            Empty, "/gazebo_bridge_unpause"
        )
        # Sub Node for awaiting requests within a callback
        self.sub_node = rclpy.create_node("sub_node")
        self.sub_client = self.sub_node.create_client(
            ResetGazeboEnv, "/gazebo_bridge_reset"
        )

    def _initialise_services(self) -> None:
        self.step_environment_service = self.create_service(
            StepEnv, "/step_environment", self.step_environment_callback
        )
        self.reset_environment_service = self.create_service(
            ResetEnv, "/reset_environment", self.reset_environment_callback
        )
        self.skip_envionment_frame_service = self.create_service(
            Empty,
            "/skip_environment_frame",
            self.skip_environment_frame_callback,
        )

    def _initialise_constants(self) -> None:
        # Robot CONSTANTS
        self.MAX_LINEAR_SPEED, self.MAX_ANGULAR_SPEED = MAX_CONTINUOUS_ACTIONS
        self.NUMBER_OF_SCANS = NUMBER_OF_SCANS
        self.MAX_SCAN_DISTANCE = MAX_SCAN_DISTANCE

        # Environment CONSTANTS
        self.MAX_STEPS = EPISODE_STEPS
        self.GOAL_DISTANCE = GOAL_PAD_RADIUS
        self.COLLISION_DISTANCE = COLLISION_DISTANCE
        self.EPISODE_LENGTH = EPISODE_LENGTH_SEC

        # DEBUG
        self.REWARD_DEBUG = REWARD_DEBUG

    def _reset_robot_variables(self) -> None:
        # Robot Variables
        self.actual_pose = np.full(2, None, dtype=np.float32)
        self.estimated_pose = np.full(2, None, dtype=np.float32)
        self.scan = (
            np.ones(self.NUMBER_OF_SCANS, dtype=np.float32) * self.MAX_SCAN_DISTANCE
        )
        self.d_optimality = None
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

    def _reset_environment_variables(self) -> None:
        # Environment Variables
        self.collided = False
        self.done = False
        self.found_goal = False
        self.step_counter = 0
        self.goal_counter = 0
        self.collision_counter = 0
        self.goal_position = np.array([0.0, 0.0])
        self.distance_to_goal = np.Inf
        self.episode_end_time = np.Inf
        self.reset_episode_end_time = True
        self.clock_msgs_skipped = 0
        self.current_time = 0

    def _initialise_gazebo_bridge_node(self):
        # Initialize the custom Gazebo Bridge Node
        req = Empty.Request()
        while not self.gazebo_bridge_init_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for gazebo bridge init service")
        self.gazebo_bridge_init_client.call_async(req)
        self.get_logger().info("Sucessfully initialised custom gazebo bridge node")

    # Process incoming LiDAR scan data and update the scan attribute.
    def scan_callback(self, msg: LaserScan):
        scan = np.empty(len(msg.ranges), dtype=np.float32)
        for i in range(len(msg.ranges)):
            if msg.ranges[i] == float("Inf"):
                scan[i] = self.MAX_SCAN_DISTANCE
            elif np.isnan(msg.ranges[i]):
                scan[i] = 0
            else:
                scan[i] = msg.ranges[i]
        self.scan = scan

    # Handle environment clock and episode time length
    def clock_callback(self, msg: Clock):
        self.current_time = msg.clock.sec
        if not self.reset_episode_end_time:
            return

        self.clock_msgs_skipped += 1
        if self.clock_msgs_skipped <= 15:
            return

        self.episode_end_time = self.current_time + self.EPISODE_LENGTH
        self.clock_msgs_skipped = 0
        self.reset_episode_end_time = False

    # Update the robot's actual position and distance to goal.
    def odom_callback(self, msg: Odometry):
        self.actual_pose[0] = msg.pose.pose.position.x
        self.actual_pose[1] = msg.pose.pose.position.y
        # Continuously update the robots distance to the goal
        self.distance_to_goal = np.sqrt(
            (self.goal_position[0] - self.actual_pose[0]) ** 2
            + (self.goal_position[1] - self.actual_pose[1]) ** 2
        )

    # Calculate yaw angle from quaternion orientation.
    def calculate_yaw(self, q_ang) -> float:
        return math.atan2(
            2.0 * (q_ang.w * q_ang.z + q_ang.x * q_ang.y),
            1.0 - 2.0 * (q_ang.y**2 + q_ang.z**2),
        )

    # Compute D-optimality from covariance matrix and update estimated pose.
    def covariance_matrix_callback(self, msg: PoseWithCovarianceStamped):
        EIG_TH = 1e-6
        LOW_UNCERTAINTY_VALUE = 1e-5  # Small value to indicate high uncertainty
        data = msg.pose  # questionable msg type structure

        # Extract the covariance matrix from the data
        matrix = data.covariance
        covariance_matrix = np.array(matrix).reshape((6, 6))

        # Check if covariance matrix is close to an identity matrix
        identity_covariance_matrix = False
        identity_like_pattern = np.diag([1, 1, 0, 0, 0, 1])
        if np.allclose(covariance_matrix, identity_like_pattern, atol=1e-2):
            identity_covariance_matrix = True
            covariance_matrix = np.diag([LOW_UNCERTAINTY_VALUE] * 6)

        # Calculate eigenvalues
        eigenvalues = scipy.linalg.eigvalsh(covariance_matrix)
        eigv = eigenvalues[eigenvalues > EIG_TH]

        # Calculate D-optimality ~ SLAM toolbox throws identity matrix after map reset
        if eigv.size == 0 or identity_covariance_matrix:
            d_optimality = None
        else:
            n = np.size(covariance_matrix, 1)
            d_optimality = np.exp(np.sum(np.log(eigv)) / n)

        self.d_optimality = d_optimality
        pose = data.pose
        self.estimated_pose = np.array([pose.position.x, pose.position.y])

    # Get updated goal pose
    def goal_position_callback(self, msg: Pose):
        self.goal_position[0] = msg.position.x
        self.goal_position[1] = msg.position.y
        print(f"new goal pose: [ {self.goal_position[0]} , {self.goal_position[1]} ]")

    # Reset robot and environment state
    def reset_environment_callback(self, request, response):
        self._unpause_environment()

        self._reset_robot_variables()
        new_pose = self._get_new_robot_start_position()
        self._reset_environment_variables()
        # Get new robot start position
        self.estimated_pose = np.array(
            [new_pose.position.x, new_pose.position.y], dtype=np.float32
        )
        self.actual_pose = np.array(
            [new_pose.position.x, new_pose.position.y], dtype=np.float32
        )

        # Return environment observation
        response.observation = np.concatenate(
            (
                self.scan,
                self.estimated_pose,
            )
        )

        # Allow time for gazebo bridge requests to complete
        time.sleep(2)

        self._pause_environment()

        return response

    def _get_new_robot_start_position(self):
        # Communicate with custom gazebo bridge
        req = ResetGazeboEnv.Request()
        req.collision = self.collided
        while not self.gazebo_bridge_reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "gazebo bridge reset service is not available, I'll wait"
            )
        future = self.sub_client.call_async(req)
        rclpy.spin_until_future_complete(self.sub_node, future)
        future_result: ResetGazeboEnv.Response = future.result()
        return future_result.pose

    def _pause_environment(self) -> None:
        while not self.gazebo_bridge_pause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("gazebo bridge pause service not running")
        self.gazebo_bridge_pause_client.call_async(Empty.Request())

    def _unpause_environment(self) -> None:
        while not self.gazebo_bridge_unpause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("gazebo bridge unpause service not running")
        self.gazebo_bridge_unpause_client.call_async(Empty.Request())

    # Calculate agent reward based on the state of the robot and environment.
    def _get_rewards(
        self,
        found_goal: bool,
        collided: bool,
        angular_vel: float,
        linear_vel: float,
        d_opt: float | None,
    ) -> float:
        return reward_function(
            found_goal,
            collided,
            angular_vel,
            linear_vel,
            d_opt,
            self.MAX_LINEAR_SPEED,
            self.REWARD_DEBUG,
        )

    # Set robot velocities to zero.
    def _stop_robot(self) -> None:
        desired_vel_cmd = Twist()
        desired_vel_cmd.linear.x = 0.0
        desired_vel_cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(Twist())

    # reset timer and spawn new goal object
    def _handle_found_goal(self, found_goal: bool) -> None:
        if not found_goal:
            return
        self.step_counter = 0
        self.goal_counter += 1
        self.reset_episode_end_time = True
        self.episode_end_time = np.Inf
        self.done = False
        self.collided = False
        self.truncated = False
        bridge_success_req = Empty.Request()
        while not self.gazebo_bridge_success_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "gazebo bridge success service is not available, I'll wait"
            )
        self.gazebo_bridge_success_client.call_async(bridge_success_req)

    # Populate step response.
    def _get_step_response(self, response):
        found_goal = (
            self.distance_to_goal < self.GOAL_DISTANCE and self.step_counter > 130
        )

        # Check smallest scan value to see if robot has collided
        collided = bool(np.min(self.scan) < self.COLLISION_DISTANCE)
        if collided:
            self.collision_counter += 1

        # Check environment clock
        self.truncated = self.current_time > self.episode_end_time
        self.done = collided
        self.collided = collided

        self._handle_found_goal(found_goal)

        # Build observation vector
        observation = np.concatenate(
            (
                self.scan,
                self.estimated_pose,
            ),
            dtype=np.float32,
        )

        response.state = observation
        response.reward = self._get_rewards(
            found_goal,
            collided,
            self.angular_velocity,
            self.linear_velocity,
            self.d_optimality,
        )
        response.truncated = self.truncated
        response.done = self.done
        response.info.goal_found = bool(found_goal)
        response.info.collided = collided
        response.info.distance_to_goal = self.distance_to_goal
        return response

    # Environment Step function callback
    def step_environment_callback(self, request, response):
        self.step_counter += 1

        self._unpause_environment()

        # Apply action velocities
        self.linear_velocity = request.actions[0]
        self.angular_velocity = request.actions[1]
        desired_vel_cmd = Twist()
        desired_vel_cmd.linear.x = self.linear_velocity.item()
        desired_vel_cmd.angular.z = self.angular_velocity.item()
        self.cmd_vel_publisher.publish(desired_vel_cmd)

        # Let simulation play out for a bit before observing
        time.sleep(0.01)

        self._pause_environment()

        # Return new state
        response = self._get_step_response(response)
        if self.done or self.truncated:
            self._stop_robot()
            self.reset_episode_end_time = True
            time.sleep(0.5)
        return response

    # Performs a step in the environment without collating information
    def skip_environment_frame_callback(self, request, response):
        self._unpause_environment()
        time.sleep(0.01)
        self._pause_environment()
        return response


def main(args=None):
    rclpy.init(args=args)
    env = LearningEnvironment()
    rclpy.spin(env)
    env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
