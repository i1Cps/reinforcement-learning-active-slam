import torch as T
import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import Bool
from active_slam_interfaces.srv import StepEnv, ResetEnv
from active_slam_learning.learning.ddpg.agent import Agent
from active_slam_learning.learning.ddpg.utils import plot_learning_curve
from active_slam_learning.common import utilities as util
from active_slam_learning.learning.ddpg.replay_memory import ReplayBuffer
from active_slam_learning.common.settings import (
    BETA_DDPG,
    ALPHA_DDPG,
    BATCH_SIZE_DDPG,
    ENVIRONMENT_ACTION_SPACE,
    TAU,
    GAMMA_DDPG,
    ACTOR_DDPG_FC1,
    ACTOR_DDPG_FC2,
    CRITIC_DDPG_FC1,
    CRITIC_DDPG_FC2,
    MAX_MEMORY_SIZE,
    RANDOM_STEPS,
    MAX_CONTINUOUS_ACTIONS,
    ENVIRONMENT_OBSERVATION_SPACE,
    TRAINING_EPISODES,
)


class LearningDDPG(Node):
    def __init__(self):
        super().__init__("learning_ddpg")
        self.episode_number = 0
        self.best_score = 0
        self.total_steps = 0
        self.score_history, self.step_history = [], []

        self.training_episodes = TRAINING_EPISODES
        self.training = True
        self.model_is_learning = False

        self.actor_dims = ENVIRONMENT_OBSERVATION_SPACE
        self.critic_dims = ENVIRONMENT_OBSERVATION_SPACE + ENVIRONMENT_ACTION_SPACE

        self.model = Agent(
            actor_dims=self.actor_dims,
            critic_dims=self.critic_dims,
            n_actions=ENVIRONMENT_ACTION_SPACE,
            max_actions=MAX_CONTINUOUS_ACTIONS,
            min_actions=MAX_CONTINUOUS_ACTIONS * -1,
            alpha=ALPHA_DDPG,
            beta=BETA_DDPG,
            tau=TAU,
            gamma=GAMMA_DDPG,
            actor_fc1=ACTOR_DDPG_FC1,
            actor_fc2=ACTOR_DDPG_FC2,
            critic_fc1=CRITIC_DDPG_FC1,
            critic_fc2=CRITIC_DDPG_FC2,
            batch_size=BATCH_SIZE_DDPG,
            checkpoint_dir="training_data/models",
            scenario="robotic_exploration",
        )

        self.memory = ReplayBuffer(
            MAX_MEMORY_SIZE, ENVIRONMENT_OBSERVATION_SPACE, ENVIRONMENT_ACTION_SPACE
        )

        # Reward normalization variables
        self.reward_mean = 0
        self.reward_var = 1
        self.reward_count = 1

        # --------------------- Clients ---------------------------#

        self.environment_step_client = self.create_client(StepEnv, "/environment_step")
        self.reset_environment_client = self.create_client(
            ResetEnv, "/reset_environment_rl"
        )

        # Check for GPU availability
        self.get_logger().info(
            "GPU AVAILABLE" if T.cuda.is_available() else "GPU UNAVAILABLE"
        )

        # Start Reinforcement Learning
        self.get_logger().info("Starting the learning loop")
        self.start(self.training_episodes)

    # Main learning loop
    def start(self, n_games):
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

                # Normalize reward
                self.update_reward_statistics(reward)
                norm_reward = (reward - self.reward_mean) / (
                    self.reward_var**0.5 + 1e-5
                )

                self.memory.store_transition(
                    observation, action, norm_reward, next_obs, done
                )

                if self.total_steps >= RANDOM_STEPS:
                    self.model.learn(self.memory)
                score += reward
                observation = next_obs
            self.score_history.append(score)
            self.step_history.append(self.total_steps)
            self.finish_episode(score)

        x = [i + 1 for i in range(n_games)]
        np.save(
            "training_data/raw_data/mappo_scores.npy",
            np.array(self.score_history),
        )
        np.save(
            "training_data/raw_data/mappo_steps.npy",
            np.array(self.step_history),
        )

    def update_reward_statistics(self, reward):
        # Incremental calculation of mean and variance
        self.reward_count += 1
        last_mean = self.reward_mean
        self.reward_mean += (reward - self.reward_mean) / self.reward_count
        self.reward_var += (reward - last_mean) * (reward - self.reward_mean)

    # Handles end of episode (nice, clean and modular)
    def finish_episode(self, score):
        self.episode_number += 1
        avg_score = np.mean(self.score_history[-100:])
        if self.training:
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.model.save_models()

        self.get_logger().info(
            "Episode: {}, score: {}, Average Score: {:.1f}".format(
                self.episode_number, score, avg_score
            )
        )


def main():
    rclpy.init()
    learning_ddpg = LearningDDPG()
    rclpy.spin(learning_ddpg)
    learning_ddpg.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
