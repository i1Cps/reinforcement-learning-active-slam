import torch as T
import rclpy
from rclpy.node import Node

import numpy as np

from std_srvs.srv import Empty
from slam_toolbox.srv import SerializePoseGraph
from active_slam_interfaces.srv import StepEnv, ResetEnv
from active_slam_learning.learning.td3.agent import Agent
from active_slam_learning.learning.td3.utils import plot_learning_curve
from active_slam_learning.common import utilities as util
from active_slam_learning.common.settings import (
    TAU,
    BETA,
    ALPHA,
    BATCH_SIZE,
    FC1_DIMS,
    FC2_DIMS,
    MAX_SIZE,
    MAX_CONTINUOUS_ACTIONS,
    ENVIRONMENT_OBSERVATION_SPACE,
)


class LearningTD3(Node):
    def __init__(self):
        super().__init__("learning_td3")
        # Variables
        self.episode_number = 0
        self.best_score = 0
        self.total_steps = 0
        self.score_history = []
        self.training_episodes = 2000
        self.training = True
        self.print_interval = 1
        self.model_is_learning = False
        self.model = Agent(
            alpha=0.0001,
            beta=0.001,
            input_dims=(ENVIRONMENT_OBSERVATION_SPACE,),
            tau=0.001,
            max_action_values=MAX_CONTINUOUS_ACTIONS,
            batch_size=64,
            layer1_size=300,
            layer2_size=300,
            max_size=1000000,
            warmup=100000,
            n_actions=2,
            logger=self.get_logger(),
        )

        # Clients
        self.environment_step_client = self.create_client(StepEnv, "/environment_step")
        self.reset_environment_client = self.create_client(
            ResetEnv, "/reset_environment_rl"
        )
        self.gazebo_pause = self.create_client(Empty, "/pause_physics")
        self.gazebo_unpause = self.create_client(Empty, "/unpause_physics")
        self.save_slam_grid_client = self.create_client(
            SerializePoseGraph, "/slam_toolbox/serialize_map"
        )

        # Immediately save the empty slam initial_map. This is a quick hack until slam_toolbox release reset() service
        req = SerializePoseGraph.Request()
        req.filename = "./src/active_slam_learning/config/initial_map"
        while not self.save_slam_grid_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "save slam grid service not available, waiting again..."
            )
        self.save_slam_grid_client.call_async(req)

        if T.cuda.is_available:
            self.get_logger().info("GPU AVAILABLE")
        else:
            self.get_logger().info("GPU UNAVAILABLE")

        # Start Reinforcement Learning
        self.start(self.training_episodes)

    # Main learning loop
    def start(self, n_games):
        self.get_logger().info("Starting the learning loop")
        # Pause the simulation while we intialise episode
        # util.pause_simulation(self)
        for i in range(n_games):
            # Reset episode variables
            terminal = False
            truncated = False
            score = 0

            # potentially dont reset
            self.model.reset_noise()
            # Get initial observation state
            obs = util.reset(self)

            # Unpause sim and sleep for half a second to allow for gazebo to catch up
            # util.unpause_simulation(self)

            # Episode loop, This loop ends when episode is 'done' or is 'truncated'
            while not (terminal or truncated):
                action = self.model.choose_action(obs)

                # Send action to environment and get returned state
                next_obs, reward, terminal, truncated = util.step(self, action)

                # Add to memory buffer
                if self.training:
                    self.model.remember(
                        obs, action, reward, next_obs, truncated or terminal
                    )
                    self.model.learn()

                obs = next_obs
                score += reward
                self.total_steps += 1

            # Episode done
            # util.pause_simulation(self)

            self.finish_episode(score)

        x = [i + 1 for i in range(n_games)]
        filename = (
            "./src/active_slam_learning/active_slam_learning/learning/td3/plots/td3.png"
        )
        plot_learning_curve(x, self.score_history, filename)

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


def main():
    rclpy.init()
    learning_td3 = LearningTD3()
    rclpy.spin(learning_td3)
    learning_td3.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
