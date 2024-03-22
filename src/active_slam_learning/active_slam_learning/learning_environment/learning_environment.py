import rclpy
from rclpy.node import Node

import numpy as np
from rclpy.utilities import timeout_sec_to_nsec
import scipy
import time
import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Pose
from slam_toolbox.srv import DeserializePoseGraph
from active_slam_interfaces.srv import StepEnv, ResetEnv

from active_slam_learning.common.settings import INITIAL_POSE
from active_slam_learning.learning_environment.reward_function import reward_function


# This Node is reponsible for providing an interface for agents to take actions and recieve new states, rewards or both
# Contains direct communication with our physics simulator, gazebo.
class LearningEnvironment(Node):
    def __init__(self):
        super().__init__("learning_environment")

        ################################################################
        # Initialise subscribers, publishers, clients and services     #
        ################################################################

        # Subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 1
        )

        self.covariance_matrix_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, "/pose", self.covariance_matrix_callback, 10
        )

        self.goal_position_subscriber = self.create_subscription(
            Pose, "/goal_position", self.goal_position_callback, 10
        )

        self.odom_subscriber = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Clients
        self.load_slam_grid_client = self.create_client(
            DeserializePoseGraph, "/slam_toolbox/deserialize_map"
        )

        self.gazebo_pause = self.create_client(Empty, "/pause_physics")
        self.gazebo_unpause = self.create_client(Empty, "/unpause_physics")
        self.gazebo_new_episode_client = self.create_client(
            Empty, "/gazebo_new_episode"
        )

        ##########################################################################
        self.environment_success_client = self.create_client(
            Empty, "/environment_success"
        )
        self.environment_fail_client = self.create_client(Empty, "/environment_fail")
        ##########################################################################

        # Services
        self.environment_step = self.create_service(
            StepEnv, "/environment_step", self.environment_step_callback
        )
        self.reset_environment = self.create_service(
            ResetEnv, "/reset_environment_rl", self.reset_callback
        )

        #################################################################
        #               CONSTANTS AND VARIABLES                         #
        #################################################################

        # Robot CONSTANTS
        self.MAX_LINEAR_SPEED = 0.22
        self.MAX_ANGULAR_SPEED = 2.0
        self.NUMBER_OF_SCANS = 90
        self.MAX_SCAN_DISTANCE = 3.5
        self.INITIAL_POSE = INITIAL_POSE

        # Robot Variables
        self.collided = False
        self.found_goal = False
        self.actual_pose = self.INITIAL_POSE  # The actual Pose of the robot
        self.current_pose = self.INITIAL_POSE  # Estimated Pose using SLAM
        self.current_scan = np.ones(self.NUMBER_OF_SCANS, dtype=np.float32) * 2
        self.current_d_optimality = None
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        # Environment CONSTANTS
        self.MAX_STEPS = 500
        self.GOAL_DISTANCE = 0.1
        self.COLLISION_DISTANCE = 0.2

        # Environment Variables
        self.done = False
        self.step_counter = 0
        self.goal_position = np.array([0, 0])
        self.distance_to_goal = np.Inf

        self.get_logger().info("Sucessfully initialised Learning Environment Node")

    # Callback function for LiDAR scan subscriber
    def scan_callback(self, data):
        scan = np.empty(len(data.ranges), dtype=np.float32)
        # Resize scan data, data itself returns extra info about the scan, scan.ranges just gets.... the ranges
        for i in range(len(data.ranges)):
            if data.ranges[i] == float("Inf"):
                scan[i] = self.MAX_SCAN_DISTANCE
            elif np.isnan(data.ranges[i]):
                scan[i] = 0
            else:
                scan[i] = data.ranges[i]
        self.current_scan = scan

    def odom_callback(self, data):
        self.actual_pose[0] = data.pose.pose.position.x
        self.actual_pose[1] = data.pose.pose.position.y
        self.distance_to_goal = np.sqrt(
            (self.goal_position[0] - self.actual_pose[0]) ** 2
            + (self.goal_position[1] - self.actual_pose[1]) ** 2
        )
        print(f"Actual Robot: {self.actual_pose[0]} , {self.actual_pose[1]}")
        print(f"Distance to goal: {self.distance_to_goal}")

    # Calculates yaw angle from quaternions, they deprecated Pose2D for some reason???? so this function is useless
    def calculate_yaw(self, q_ang):
        return math.atan2(
            2.0 * (q_ang.w * q_ang.z + q_ang.x * q_ang.y),
            1.0 - 2.0 * (q_ang.y**2 + q_ang.z**2),
        )

    # Callback function for covariance matrix subscriber
    def covariance_matrix_callback(self, data):
        # Get D-Optimality
        EIG_TH = 1e-6  # or any threshold value you choose
        data = data.pose
        matrix = data.covariance
        covariance_matrix = np.array(matrix).reshape((6, 6))
        eigenvalues = scipy.linalg.eigvalsh(covariance_matrix)
        if np.iscomplex(eigenvalues.any()):
            print("Error: Complex Root")
        eigv = eigenvalues[eigenvalues > EIG_TH]
        n = np.size(covariance_matrix, 1)
        d_optimality = np.exp(np.sum(np.log(eigv)) / n)
        self.current_d_optimality = d_optimality

        # Get Pose
        pose = data.pose
        self.current_pose = np.array([pose.position.x, pose.position.y])

    def goal_position_callback(self, data):
        self.goal_position[0] = data.position.x
        self.goal_position[1] = data.position.y
        print(f"new goal pose: [ {self.goal_position[0]} , {self.goal_position[1]} ]")

    def reset_callback(self, request, response):
        # Reset robot variables to prevent reset loop
        self.current_scan = np.ones(self.NUMBER_OF_SCANS, dtype=np.float32) * 2
        self.current_pose = self.INITIAL_POSE
        self.current_d_optimality = None
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.step_counter = 0
        # Reset robots velocity
        desired_vel_cmd = Twist()
        desired_vel_cmd.linear.x = 0.0
        desired_vel_cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(Twist())

        # This is debugging, if you are seeing this ignore the file it is probably broken
        if self.done:
            if self.collided:
                environment_fail_req = Empty.Request()
                while not self.environment_fail_client.wait_for_service(
                    timeout_sec=1.0
                ):
                    self.get_logger().info(
                        "Environment fail service is not available, I'll wait"
                    )
                self.environment_fail_client.call_async(environment_fail_req)
            else:
                environment_success_req = Empty.Request()
                while not self.environment_success_client.wait_for_service(
                    timeout_sec=1.0
                ):
                    self.get_logger().info(
                        "Environment success service is not available, I'll wait"
                    )
                self.environment_success_client.call_async(environment_success_req)

        self.done = False
        self.truncated = False
        self.collided = False
        response.observation = np.concatenate(
            (self.current_scan, self.current_pose), dtype=np.float32
        )

        # TODO: Use ros approved variation of time.sleep()
        time.sleep(3)
        # Pause execution

        return response

    # Calculates the rewards based on collisions, angular velocity and map certainty
    def get_rewards(
        self,
        found_goal: bool,
        collided: bool,
        angular_vel: float,
        linear_vel: float,
        d_opt: float | None,
    ) -> float:
        return reward_function(
            found_goal, collided, angular_vel, linear_vel, d_opt, self.MAX_LINEAR_SPEED
        )

    # Check if the given scan shows a collision
    def has_collided(self):
        return bool(self.COLLISION_DISTANCE > np.min(self.current_scan))

    def has_found_goal(self):
        return self.distance_to_goal < self.GOAL_DISTANCE

    def handle_found_goal(self):
        self.step_counter = 0
        self.done = False
        self.truncated = False
        environment_success_req = Empty.Request()
        while not self.environment_success_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Environment success service is not available, I'll wait"
            )
        self.environment_success_client.call_async(environment_success_req)

    # Populates the srv message response with obs,reward,truncated and, done
    def get_step_response(self, response):
        found_goal = self.has_found_goal() and self.step_counter > 130
        collided = self.has_collided()
        self.truncated = self.step_counter > self.MAX_STEPS
        self.done = collided

        # if found_goal:
        #    self.handle_found_goal()
        observation = np.concatenate(
            (self.current_scan, self.current_pose), dtype=np.float32
        )
        response.state = observation
        response.reward = self.get_rewards(
            found_goal,
            collided,
            # Below just ensures the reward is based on the raw output of the model and not the noise,
            # The noise kind of interfers with the reward function, will streamline later
            # Raw model output is [-1,1] lin and ang vel
            max(
                min(self.current_angular_velocity, self.MAX_ANGULAR_SPEED),
                -self.MAX_ANGULAR_SPEED,
            ),
            max(
                min(self.current_linear_velocity, self.MAX_LINEAR_SPEED),
                -self.MAX_LINEAR_SPEED,
            ),
            self.current_d_optimality,
        )
        response.truncated = self.truncated
        response.done = self.done
        return response

    # Agent will make a request to environment containing its chosen actions,
    # Environment service will return new state that proceeds from said action
    # Very simular to typical step function in normal Reinforcement Learning environments.
    def environment_step_callback(self, request, response):
        self.step_counter += 1
        self.current_linear_velocity = request.actions[0] * self.MAX_LINEAR_SPEED
        self.current_angular_velocity = request.actions[1] * self.MAX_ANGULAR_SPEED
        desired_vel_cmd = Twist()
        desired_vel_cmd.linear.x = self.current_linear_velocity
        desired_vel_cmd.angular.z = self.current_angular_velocity
        self.cmd_vel_publisher.publish(desired_vel_cmd)

        # Let simulation play out for a bit before observing
        time.sleep(0.1)

        # Return new state
        response = self.get_step_response(response)
        if self.done or self.truncated:
            self.stop_robots()
        return response


def main(args=None):
    rclpy.init(args=args)
    env = LearningEnvironment()
    rclpy.spin(env)
    env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
