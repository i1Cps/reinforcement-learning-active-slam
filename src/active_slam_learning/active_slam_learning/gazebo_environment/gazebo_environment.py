import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import DeleteEntity, SpawnEntity


# Node that handles direct communication with gazebo services and training environment, will control spawning of goal model
class GazeboEnvironment(Node):
    def __init__(self):
        super().__init__("gazebo_cli")

        ################################################################
        # Initialise subscribers, publishers, clients and services     #
        ################################################################

        # Publishers
        self.goal_position_publisher = self.create_publisher(Pose, "/goal_pose", 10)

        # Clients
        self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_entity_client = self.create_client(SpawnEntity, "spawn_entity")
        self.reset_world_client = self.create_client(Empty, "/reset_world")
        self.pause_gazebo_client = self.create_client(Empty, "/pause_physics")

        # Services
        self.environment_success_service = self.create_service(
            Empty, "/environment_success", self.environment_success_callback
        )
        self.environment_fail_service = self.create_service(
            Empty, "/environment_fail", self.environment_fail_callback
        )

        self.new_episode = self.create_service(
            Empty, "/gazebo_new_episode", self.new_episode_callback
        )
        self.get_logger().info("Sucessfully initialised Gazebo Environment Node")

    def environment_fail_callback(self, request, response):
        self.delete_entity()
        self.reset_simulation()
        self.generate_new_goal_position()
        self.get_logger().info(
            "Episode fail, resetting environment.... New goal position:[ {self.goal_position_x}, {self.goal_position_y} ]"
        )
        return response

    def environment_success_callback(self, request, response):
        self.delete_entity()
        self.generate_new_goal_position()
        self.get_logger().info(
            "Episode success, New goal position: [ {self.goal_position_x}, {self.goal_position_y} ]"
        )
        return response

    def generate_new_goal_position(self):
        print("Coming SOON")

    def reset_simulation(self):
        print("Coming SOON")

    def delete_entity(self):
        print("Coming SOON")

    def publish_new_goal(self):
        goal_position = Pose()
        goal_position.position.x = self.goal_position_x
        goal_position.position.y = self.goal_position_y
        self.goal_position_publisher.publish(goal_position)
        self.spawn_entity_client()

    def spawn_goal_entity(self):
        print("Coming SOON")

    def new_episode_callback(self, request, response):
        self.get_logger().info(
            "new episode, sending reset simulation request to gazebo"
        )
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.reset_world_client.call_async(request)
        return response


def main():
    rclpy.init()
    gazebo = GazeboEnvironment()
    rclpy.spin(gazebo)
    gazebo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
