import torch as T
import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import Bool
from active_slam_interfaces.srv import StepEnv, ResetEnv
from active_slam_learning.learning.ddpg.agent import Agent
from active_slam_learning.learning.ddpg.utils import plot_learning_curve
from active_slam_learning.common import utilities as util
from active_slam_learning.common.settings import (
    TAU,
    BETA,
    ALPHA,
    BATCH_SIZE,
    FC1_DIMS,
    FC2_DIMS,
    MAX_SIZE,
    RANDOM_STEPS,
    MAX_CONTINUOUS_ACTIONS,
    ENVIRONMENT_OBSERVATION_SPACE,
    TRAINING_EPISODES,
)


class LearningDDPG(Node):
    def __init__(self):
        super().__init__("learning_ddpg")
        # Variables
        self.episode_number = 0
        self.best_score = 0
        self.total_steps = 0
        self.score_history = []
        self.training_episodes = TRAINING_EPISODES
        self.training = True
        self.print_interval = 1
        self.model_is_learning = False
        self.model = Agent(
            alpha=ALPHA,
            beta=BETA,
            input_dims=(ENVIRONMENT_OBSERVATION_SPACE,),
            tau=TAU,
            n_actions=2,
            max_action=MAX_CONTINUOUS_ACTIONS,
            batch_size=BATCH_SIZE,
            fc1_dims=FC1_DIMS,
            fc2_dims=FC2_DIMS,
            max_size=MAX_SIZE,
            logger=self.get_logger(),
        )

        # -------------------- Publisher ------------------------ #

        self.shutdown_publisher = self.create_publisher(Bool, "/shutdown_rl_nodes", 10)

        # --------------------- Clients ---------------------------#
        #
        self.environment_step_client = self.create_client(StepEnv, "/environment_step")
        self.reset_environment_client = self.create_client(
            ResetEnv, "/reset_environment_rl"
        )

        if T.cuda.is_available:
            self.get_logger().info("GPU AVAILABLE")
        else:
            self.get_logger().info("GPU UNAVAILABLE")

        # Start Reinforcement Learning
        self.start(self.training_episodes)

    # Main learning loop
    def start(self, n_games):
        self.get_logger().info("Starting the learning loop")
        for _ in range(n_games):
            # Reset episode
            observation = util.reset(self)
            done = False
            score = 0
            while not done:
                if self.total_steps < RANDOM_STEPS:
                    action = self.model.choose_random_action()
                else:
                    action = self.model.choose_action(observation)
                next_obs, reward, terminal, truncated = util.step(self, action)
                self.total_steps += 1
                done = terminal or truncated
                self.model.store_transition(observation, action, reward, next_obs, done)
                if self.total_steps >= RANDOM_STEPS:
                    self.model.learn()
                score += reward
                observation = next_obs
            self.finish_episode(score)

        x = [i + 1 for i in range(n_games)]
        filename = "./src/active_slam_learning/active_slam_learning/learning/ddpg/plots/ddpg.png"
        plot_learning_curve(x, self.score_history, filename)
        # self.shutdown_nodes()

    # Handles end of episode (nice, clean and modular)
    def finish_episode(self, score):
        self.episode_number += 1
        # Calculate the robot avearage score
        self.score_history.append(score)
        # Average the last 100 recent scores
        avg_score = np.mean(self.score_history[-100:])
        if self.training:
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.model.save_models()

        if self.episode_number % self.print_interval == 0 and self.episode_number > 0:
            self.get_logger().info(
                "Episode: {}, score: {}, Average Score: {:.1f}".format(
                    self.episode_number, score, avg_score
                )
            )

    def shutdown_nodes(self):
        self.get_logger().info("Shutting down the node and ROS.")
        shutdown_message = Bool(data=True)
        self.shutdown_publisher.publish(shutdown_message)
        self.destroy_node()
        rclpy.shutdown()


def main():
    rclpy.init()
    learning_ddpg = LearningDDPG()
    rclpy.spin(learning_ddpg)
    # learning_ddpg.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
