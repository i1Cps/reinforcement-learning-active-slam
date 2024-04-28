import torch as T
import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import Bool
from active_slam_interfaces.srv import StepEnv, ResetEnv
from active_slam_learning.learning.ppo.agent import Agent
from active_slam_learning.learning.ppo.utils import plot_learning_curve
from active_slam_learning.common import utilities as util
from active_slam_learning.common.settings import (
    ALPHA,
    BATCH_SIZE,
    FC1_DIMS,
    FC2_DIMS,
    RANDOM_STEPS,
    ENVIRONMENT_OBSERVATION_SPACE,
    TRAINING_EPISODES,
    MAX_CONTINUOUS_ACTIONS,
)


class LearningPPO(Node):
    def __init__(self):
        super().__init__("learning_ppo")
        # Variables
        self.episode_number = 1
        self.max_action = MAX_CONTINUOUS_ACTIONS
        self.best_score = 0
        self.total_steps = 0
        self.score_history = []
        self.training_episodes = TRAINING_EPISODES

        # PPO variables
        self.N = 2048
        self.n_epochs = 10
        self.trajectory_len = 0

        self.model = Agent(
            alpha=3e-4,
            input_dims=(ENVIRONMENT_OBSERVATION_SPACE,),
            n_actions=2,
            batch_size=BATCH_SIZE,
            fc1_dims=FC1_DIMS,
            fc2_dims=FC2_DIMS,
        )

        # --------------------- Clients ---------------------------#

        self.environment_step_client = self.create_client(StepEnv, "/environment_step")
        self.reset_environment_client = self.create_client(
            ResetEnv, "/reset_environment_rl"
        )

        # --------------------- Check GPU Availability ---------------- #

        if T.cuda.is_available:
            self.get_logger().info("GPU AVAILABLE")
        else:
            self.get_logger().info("GPU UNAVAILABLE")

        # Start Reinforcement Learning
        self.start(self.training_episodes)

    def start(self, n_games):
        self.get_logger().info("Starting the learning loop")
        for _ in range(n_games):
            observation = util.reset(self)
            done = False
            score = 0
            while not done:
                """
                if self.total_steps < RANDOM_STEPS:
                    action = self.model.choose_random_action()
                else:
                    action = self.model.choose_action(observation)
                """
                action, prob = self.model.choose_action(observation)
                adapted_action = self.action_adapter(action, self.max_action)
                next_obs, reward, terminal, truncated = util.step(self, adapted_action)
                score += reward
                done = terminal or truncated
                self.model.remember(observation, action, reward, next_obs, done, prob)
                self.total_steps += 1
                self.trajectory_len += 1
                if self.trajectory_len % self.N == 0:
                    self.model.learn()
                    self.trajectory_len = 0
                observation = next_obs
                """
                if self.total_steps >= RANDOM_STEPS:
                    self.model.learn()
                """
            self.finish_episode(score)

        x = [i + 1 for i in range(n_games)]
        filename = (
            "./src/active_slam_learning/active_slam_learning/learning/ppo/plots/ppo.png"
        )
        plot_learning_curve(x, self.score_history, filename)

    # Handles end of episode (nice, clean and modular)
    def finish_episode(self, score):
        self.episode_number += 1
        # Calculate the robot avearage score
        self.score_history.append(score)
        # Average the last 100 recent scores
        avg_score = np.mean(self.score_history[-100:])
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.model.save_models()

        self.get_logger().info(
            "Episode: {}, score: {}, Average Score: {:.1f}".format(
                self.episode_number, score, avg_score
            )
        )

    # Convert action range 0-1 -> -max_action, max_action
    def action_adapter(self, a, max_a):
        return 2 * (a - 0.5) * max_a


def main():
    rclpy.init()
    learning_ppo = LearningPPO()
    rclpy.spin(learning_ppo)
    learning_ppo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
