import rclpy
import numpy as np
from std_srvs.srv import Empty
from active_slam_interfaces.srv import StepEnv, ResetEnv


# Communicates to the Learning Environment Node that it should step the environment using these actions
# It then expects typical reinforcement learning items back
def step(agent_self, action):
    req = StepEnv.Request()
    actions = np.array((action[0], action[1]), dtype=np.float32)
    req.actions = actions

    while not agent_self.step_environment_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info(
            "environment step service not available, waiting again..."
        )
    future = agent_self.step_environment_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                info = {
                    "collided": res.info.collided,
                    "goal_found": res.info.goal_found,
                    "distance_to_goal": res.info.distance_to_goal,
                }
                if res.info.goal_found:
                    print(info)
                    print(info["goal_found"])
                return res.state, res.reward, res.done, res.truncated, info
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error getting step service response, this wont output anywhere")


# Communicates to the Learning Environment Node that it should reset everything
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


# Communicates to the Learning Environment Node that it should skip a frame
def skip_frame(agent_self):
    req = Empty.Request()
    while not agent_self.skip_environment_frame_client.wait_for_service(
        timeout_sec=1.0
    ):
        agent_self.get_logger().info("Frame skip service not available, waiting ")
    future = agent_self.skip_environment_frame_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                return
            else:
                agent_self.get_logger().error(
                    "Exception while calling service: {0}".format(future.exception())
                )
                print("Error skipping the env, if thats even possible ")
