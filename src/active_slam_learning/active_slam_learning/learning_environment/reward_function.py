import numpy as np


def reward_function(
    found_goal: bool,
    collided: bool,
    angular_vel: float,
    linear_vel: float,
    d_opt: float | None,
    max_speed: float,
    log=False,
) -> float:
    """
    Calculate the reward for the agent based on its current state and actions.

    Parameters:
    - found_goal (bool): Whether the goal has been found.
    - collided (bool): Whether the robot has collided.
    - angular_vel (float): The angular velocity of the robot.
    - linear_vel (float): The linear velocity of the robot.
    - d_opt (float | None): D-optimality measure for map uncertainty.
    - max_speed (float): Maximum speed of the robot.
    - log (bool): Whether to log debug information.

    Returns:
    - float: The calculated reward.
    """

    initial_reward = -0.4

    # Negative Reward: Encourage higher linear velocity
    # Range: [-0.4 , 0] for linear velocity in range [-0.2,0.2]
    linear_vel_reward = -3 * (max_speed - linear_vel)

    # Negative Reward: Penalise higher angular velocity
    # Range: [-0.484, 0] for angular velocity in range [-2.2, 2.2]
    angular_vel_reward = -0.2 * (angular_vel**2)

    # Negative Reward: High penalty for collisions
    collision_reward = -1000 if collided else 0

    # Positive Reward: High reward for reaching the goal
    goal_reward = 1000 if found_goal else 0

    if log:
        print(d_opt)

    # Positive Intrinsic Reward: Reward for map uncertainty
    if d_opt is None:
        d_optimality_reward = 0
    else:
        # Eta is a coefficent for scaling the map uncertainty value
        eta = 0.01
        d_optimality_reward = np.tanh(eta / d_opt)

    if log:
        print(
            "Rewards: lv: {:.4f} av: {:.4f} d_opt: {:.4f}".format(
                linear_vel_reward, angular_vel_reward, d_optimality_reward
            )
        )
    return (
        initial_reward
        + angular_vel_reward
        + goal_reward
        + collision_reward
        + d_optimality_reward
        + linear_vel_reward
    )
