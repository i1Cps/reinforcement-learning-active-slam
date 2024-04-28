import rclpy
import torch
import numpy as np
from std_srvs.srv import Empty
from active_slam_interfaces.srv import StepEnv, ResetEnv


# Pause Gazebo Physics simulation
def pause_simulation(agent_self):
    while not agent_self.gazebo_pause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "pause gazebo service not available, waiting again..."
        )
    future = agent_self.gazebo_pause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return


# Unpause Gazebo Physics simulation
def unpause_simulation(agent_self):
    while not agent_self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "unpause gazebo service not available, waiting again..."
        )
    future = agent_self.gazebo_unpause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return


# Function communicates with environment node, it request action and expects response
# containing observation, reward, and whether episode has finished or truncated
def step(agent_self, action, discrete=False):
    req = StepEnv.Request()
    actions = np.array((action[0], action[1]), dtype=np.float32)
    req.actions = actions

    while not agent_self.environment_step_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "environment step service not available, waiting again..."
        )
    future = agent_self.environment_step_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.state, res.reward, res.done, res.truncated
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error getting step service response, this wont output anywhere")
    print("next")


# Make this spin til complete
def reset(agent_self):
    req = ResetEnv.Request()
    while not agent_self.reset_environment_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "reset environment service not available, waiting again..."
        )
    future = agent_self.reset_environment_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.observation
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error resetting the env, if thats even possible ")
