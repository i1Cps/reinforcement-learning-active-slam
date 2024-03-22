import numpy as np


def reward_function(
    found_goal: bool,
    collided: bool,
    angular_vel: float,
    linear_vel: float,
    d_opt: float | None,
    max_speed: float,
) -> float:
    # If the environment hasnt calculated a D-Optimality score yet, return 0. (The whole reward system is centered around it)
    if d_opt is None:
        return 0.0

    # Eta is a coefficent for scaling the map uncertainty value, this really should be a constant but eh
    eta = 0.01

    # Negative Reward
    linear_vel_reward = -1 * (((max_speed - linear_vel) * 10) ** 2)
    # self.get_logger().info("linear_vel_reward: {}".format(linear_vel_reward))
    # print("linear_vel_reward: {}".format(linear_vel_reward))

    # Negative Reward
    angular_vel_reward = -1 * (angular_vel**2)
    # self.get_logger().info("angular_vel_reward: {}".format(angular_vel_reward))
    # print("angular_vel_reward: {}".format(angular_vel_reward))

    # Negative Reward
    collision_reward = -800 if collided else 0

    # Positive Reward
    goal_reward = 1000 if found_goal else 0

    # Positive Reward
    d_optimality_reward = np.tanh(eta / d_opt)
    return (
        angular_vel_reward
        + goal_reward
        + collision_reward
        + d_optimality_reward
        + linear_vel_reward
    )
