import rclpy
from rclpy.node import Node

import numpy as np
import scipy
import time
import math

from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from slam_toolbox.srv import DeserializePoseGraph, SerializePoseGraph
from active_slam_interfaces.srv import StepEnv, ResetEnv

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

        # Services
        self.environment_step = self.create_service(
            StepEnv, "/environment_step", self.environment_step_callback
        )
        self.reset_environment = self.create_service(
            ResetEnv, "/reset_environment_rl", self.reset_callback
        )

        # Robot CONSTANTS
        self.MAX_LINEAR_SPEED = 0.22
        self.MAX_ANGULAR_SPEED = 2.0
        self.NUMBER_OF_SCANS = 90
        self.COLLISION_DISTANCE = 0.2
        self.MAX_SCAN_DISTANCE = 3.5
        self.INITIAL_POSE = np.array([-2.0, -0.5])

        # Robot Variables
        self.collided = False
        self.current_pose = self.INITIAL_POSE
        self.current_scan = np.ones(self.NUMBER_OF_SCANS, dtype=np.float32) * 2
        self.current_d_optimality = None
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        # Episode CONSTANTS
        self.MAX_STEPS = 500

        # Episode Variables
        self.step_counter = 0

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

    def reset_callback(self, request, response):
        # Reset robot variables to prevent reset loop
        self.current_scan = np.ones(self.NUMBER_OF_SCANS, dtype=np.float32) * 2
        self.current_pose = self.INITIAL_POSE
        self.current_d_optimality = None
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.collided = False
        self.done = False
        self.truncated = False
        self.step_counter = 0
        # Reset robots velocity
        desired_vel_cmd = Twist()
        desired_vel_cmd.linear.x = 0.0
        desired_vel_cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(Twist())

        # Reset robot position
        gazebo_reset_req = Empty.Request()
        while not self.gazebo_new_episode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "gazebo reset service not available, waiting again..."
            )
        self.gazebo_new_episode_client.call_async(gazebo_reset_req)

        response.observation = np.concatenate(
            (self.current_scan, self.current_pose), dtype=np.float32
        )

        # This is a hack until slam_toolbox release reset() service
        load_slam_map_request = DeserializePoseGraph.Request()
        load_slam_map_request.filename = "src/active_slam_learning/config/initial_map"
        load_slam_map_request.match_type = 2
        load_slam_map_request.initial_pose.x = self.INITIAL_POSE[0]
        load_slam_map_request.initial_pose.y = self.INITIAL_POSE[1]
        load_slam_map_request.initial_pose.theta = 0.0
        while not self.load_slam_grid_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "slam map reset service not available, waiting again..."
            )
        self.load_slam_grid_client.call_async(load_slam_map_request)

        # TODO: Use ros approved variation of time.sleep()
        self.get_logger().info("pausing")
        time.sleep(3)
        self.get_logger().info("pausing")
        # Pause execution

        return response

    # Calculates the rewards based on collisions, angular velocity and map certainty
    def get_rewards(
        self, collided: bool, angular_vel: float, linear_vel: float, d_opt: float | None
    ) -> float:
        return reward_function(
            collided, angular_vel, linear_vel, d_opt, self.MAX_LINEAR_SPEED
        )
        # If the environment hasnt calculated a D-Optimality score yet, return 0. (The whole reward system is centered around it)
        if d_opt is None:
            return 0.0

        # Eta is a coefficent for scaling the map uncertainty value, this really should be a constant but eh
        eta = 0.01

        # Negative Reward
        linear_vel_reward = -1 * (((self.MAX_LINEAR_SPEED - linear_vel) * 10) ** 2)
        # self.get_logger().info("linear_vel_reward: {}".format(linear_vel_reward))

        # Negative Reward
        angular_vel_reward = -1 * (angular_vel**2)
        # self.get_logger().info("angular_vel_reward: {}".format(angular_vel_reward))

        # Negative Reward
        collision_reward = -800 if collided else 0

        # Positive Reward
        d_optimality_reward = np.tanh(eta / d_opt)
        return (
            angular_vel_reward
            + collision_reward
            + d_optimality_reward
            + linear_vel_reward
        )

    # Check if the given scan shows a collision
    def hasCollided(self, scan_range):
        if self.COLLISION_DISTANCE > np.min(scan_range):
            return True
        return False

    # Populates the srv message response with obs,reward,truncated and, done
    def get_step_response(self, response):
        observation = np.concatenate(
            (self.current_scan, self.current_pose), dtype=np.float32
        )
        response.state = observation
        collided = self.hasCollided(self.current_scan)
        response.reward = self.get_rewards(
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

        # Return truncated true if max steps reached
        self.truncated = self.step_counter > self.MAX_STEPS
        self.done = collided
        response.truncated = self.truncated
        response.done = collided
        # Will change to include a goal
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
        # if self.done or self.truncated:
        #    # Debugging
        #    self.get_logger().info(
        #        "done {}, or truncated: {}".format(self.done, self.truncated)
        #    )
        return response


def main(args=None):
    rclpy.init(args=args)
    env = LearningEnvironment()
    rclpy.spin(env)
    env.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
