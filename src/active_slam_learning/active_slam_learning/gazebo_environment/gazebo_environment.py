import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
from slam_toolbox.srv import DeserializePoseGraph
from active_slam_learning.common.settings import INITIAL_POSE


# Node that handles direct communication with gazebo services and training environment, will control spawning of goal model
class GazeboEnvironment(Node):
    def __init__(self):
        super().__init__("gazebo_cli")

        ################################################################
        # Initialise subscribers, publishers, clients and services     #
        ################################################################

        # Publishers
        self.goal_position_publisher = self.create_publisher(Pose, "/goal_position", 10)

        # Clients
        self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.reset_world_client = self.create_client(Empty, "/reset_simulation")
        self.pause_gazebo_client = self.create_client(Empty, "/pause_physics")
        self.load_slam_map_client = self.create_client(
            DeserializePoseGraph, "/slam_toolbox/deserialize_map"
        )
        self.set_robot_pose = self.create_client(SetEntityState, "/set_entity_state")

        # Services
        self.environment_success_service = self.create_service(
            Empty, "/environment_success", self.environment_success_callback
        )
        self.environment_fail_service = self.create_service(
            Empty, "/environment_fail", self.environment_fail_callback
        )

        self.get_logger().info("Sucessfully initialised Gazebo Environment Node")

        #############################################################
        #           VARIABLES AND CONSTANTS                         #
        #############################################################

        self.previous_goal_position_x = -22.0
        self.previous_goal_position_y = -50.0
        self.goal_position_x = -2.0
        self.goal_position_y = 0.5
        self.goal_entity_name = "goal_pad"
        self.goal_entity = os.path.join(
            get_package_share_directory("active_slam_simulations"),
            "models",
            "goal_pad",
            "model.sdf",
        )
        self.first_episode = True
        self.init_callback()

    def init_callback(self):
        self.delete_entity()
        self.reset_simulation()
        self.publish_new_goal()
        print("Init, goal pose:", self.goal_position_x, self.goal_position_y)
        time.sleep(1)

    def environment_fail_callback(self, request, response):
        self.delete_entity()
        self.reset_simulation()
        self.reset_slam_map()
        self.publish_new_goal()
        # self.generate_new_goal_position()
        if self.first_episode:
            self.get_logger().info("Resetting The Environment")
            self.first_episode = False
        else:
            self.get_logger().info("Episode was unsuccesful, Resetting The Environment")
        return response

    def environment_success_callback(self, request, response):
        self.delete_entity()
        self.generate_new_goal_position()
        self.get_logger().info(
            f"Episode success, New goal position: [ {self.goal_position_x}, {self.goal_position_y} ]"
        )
        return response

    def generate_new_goal_position(self):
        self.previous_goal_position_x = self.goal_position_x
        self.previous_goal_position_y = self.goal_position_y
        generation_attemps = 0

        while (
            abs(self.previous_goal_position_x - self.goal_position_x)
            + abs(self.previous_goal_position_y - self.goal_position_y)
        ) < -0.5:
            map_goal_poses = [
                [2.0, 0.0],
                [0.0, 2.0],
                [0.0, -2.0],
                [-0.5, 0.5],
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [1.5, -1.5],
                [-1.5, -1.5],
                [-1.5, 1.5],
                [1.5, 1.5],
            ]
            debug_goal_poses = [[-1.1, -0.5], [-2, 0.5]]
            # index = np.random.randint(0, len(map_goal_poses))
            index = np.random.randint(0, len(debug_goal_poses))
            print(index)
            self.goal_position_x = float(debug_goal_poses[index][0])
            self.goal_position_y = float(debug_goal_poses[index][1])
            generation_attemps += 1
            if generation_attemps > 100:
                print("ERROR generating new goal")
                break
        self.publish_new_goal()

    def reset_simulation(self):
        req = Empty.Request()
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.reset_world_client.call_async(req)

    def reset_slam_map(self):
        # This is a hack until the slam_toolbox releases reset() service
        load_slam_map_request = DeserializePoseGraph.Request()
        load_slam_map_request.filename = "src/active_slam_learning/config/initial_map"
        load_slam_map_request.match_type = 2
        load_slam_map_request.initial_pose.x = INITIAL_POSE[0]
        load_slam_map_request.initial_pose.y = INITIAL_POSE[1]
        load_slam_map_request.initial_pose.theta = 0.0
        while not self.load_slam_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "slam map reset service not available, waiting again..."
            )
        self.load_slam_map_client.call_async(load_slam_map_request)

    # Publish the goal position to the learning environment node so it can track whether the agent reaches said goal
    def publish_new_goal(self):
        goal_position = Pose()
        goal_position.position.x = self.goal_position_x
        goal_position.position.y = self.goal_position_y
        self.goal_position_publisher.publish(goal_position)
        self.spawn_goal_entity()

    def spawn_goal_entity(self):
        goal_position = Pose()
        goal_position.position.x = self.goal_position_x
        goal_position.position.y = self.goal_position_y
        req = SpawnEntity.Request()
        req.name = self.goal_entity_name
        req.xml = open(self.goal_entity, "r").read()
        req.initial_pose = goal_position
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Spawn Entity client not available, waiting some more!"
            )
        print("spawning")
        self.spawn_entity_client.call_async(req)

    def delete_entity(self):
        req = DeleteEntity.Request()
        req.name = self.goal_entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Delete entity service is not available")
        self.delete_entity_client.call_async(req)


def main():
    rclpy.init()
    gazebo = GazeboEnvironment()
    rclpy.spin(gazebo)
    gazebo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
