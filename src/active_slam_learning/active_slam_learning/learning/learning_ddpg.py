from pathlib import Path
import time
import torch as T
import rclpy
from rclpy.node import Node

import numpy as np

from std_srvs.srv import Empty

from active_slam_interfaces.srv import StepEnv, ResetEnv
from active_slam_learning.learning.ddpg.agent import Agent
from active_slam_learning.common import utilities as util
from active_slam_learning.learning.ddpg.replay_memory import ReplayBuffer
from active_slam_learning.common.settings import (
    MODEL_PATH,
    BETA_DDPG,
    ALPHA_DDPG,
    BATCH_SIZE_DDPG,
    FRAME_BUFFER_DEPTH,
    FRAME_BUFFER_SKIP,
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
    ENVIRONMENT_ACTION_SPACE,
    MAX_SCAN_DISTANCE,
    TRAINING_EPISODES,
    TRAINING_STEPS,
    LOAD_MODEL,
)


class LearningDDPG(Node):
    def __init__(self):
        super().__init__("learning_ddpg")

        self.initialise_parameters()
        self.model = self.initialise_model()
        self.memory = self.initialise_memory()
        self.initialise_clients()

        # Check for GPU availability
        self.get_logger().info(
            "GPU AVAILABLE" if T.cuda.is_available() else "WARNING GPU UNAVAILABLE"
        )

        # Start Reinforcement Learning
        self.start_training()
        self.end_training()
        # Save data
        self.save_training_data()

    def initialise_parameters(self):
        self.episode_number = 0
        self.total_steps = 0

        self.score_history = []
        self.step_history = []
        self.goal_history = []
        self.collision_history = []
        self.best_score = -np.Infinity

        self.training_start_time = time.perf_counter()
        self.training_episodes = TRAINING_EPISODES
        self.training_steps = TRAINING_STEPS
        self.load_model = LOAD_MODEL

        # Create Directory in user system
        self.model_path = Path(MODEL_PATH)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Frame stacking
        self.stack_depth = FRAME_BUFFER_DEPTH
        self.frame_skip = FRAME_BUFFER_SKIP
        self.frame_buffer = np.full(
            (self.stack_depth * ENVIRONMENT_OBSERVATION_SPACE),
            MAX_SCAN_DISTANCE,
            dtype=np.float32,
        )

        # Network Dimensions
        self.actor_dims = ENVIRONMENT_OBSERVATION_SPACE * self.stack_depth
        self.critic_dims = (
            ENVIRONMENT_OBSERVATION_SPACE * self.stack_depth + ENVIRONMENT_ACTION_SPACE
        )

        # Reward normalisation
        self.reward_mean = 0
        self.reward_var = 1
        self.reward_count = 1

    def initialise_model(self) -> Agent:
        model = Agent(
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
        )

        if self.load_model:
            model.load(self.model_path)
        return model

    def initialise_memory(self) -> ReplayBuffer:
        return ReplayBuffer(
            MAX_MEMORY_SIZE,
            ENVIRONMENT_OBSERVATION_SPACE * self.stack_depth,
            ENVIRONMENT_ACTION_SPACE,
        )

    def initialise_clients(self) -> None:
        self.step_environment_client = self.create_client(StepEnv, "/step_environment")
        self.reset_environment_client = self.create_client(
            ResetEnv, "/reset_environment"
        )
        self.skip_environment_frame_client = self.create_client(
            Empty, "/skip_environment_frame"
        )

    # Main learning loop
    def start_training(self):
        self.get_logger().info("Starting the Reinforcement Learning")
        while self.total_steps < self.training_steps:
            # Reset episode
            observation = util.reset(self)

            # Prepare frame buffer
            self.frame_buffer.fill(MAX_SCAN_DISTANCE)
            _ = self.update_frame_buffer(observation)

            current_frame = 0
            done = False
            score = 0
            goals_found = 0
            action = self.model.choose_random_action()

            while not done:
                current_frame += 1
                if current_frame % self.frame_skip == 0:
                    # Choose actions
                    if self.total_steps < RANDOM_STEPS:
                        action = self.model.choose_random_action()
                    else:
                        action = self.model.choose_action(self.frame_buffer)

                    # Step the environment
                    next_obs, reward, terminal, truncated, info = util.step(
                        self, action
                    )

                    # bookkeep goals found per episode
                    goals_found += info["goal_found"]

                    # Check for episode termination
                    done = terminal or truncated
                    self.collision_history.append(int(terminal))

                    # Normalize reward
                    self.update_reward_statistics(reward)
                    norm_reward = (reward - self.reward_mean) / (
                        self.reward_var**0.5 + 1e-5
                    )

                    # Store the current state of the buffer
                    current_frame_buffer = self.frame_buffer.copy()
                    # Update the frame buffer with next_obs and store it too
                    next_frame_buffer = self.update_frame_buffer(next_obs)

                    # Store the transition in a memory buffer for sampling
                    self.memory.store_transition(
                        state=current_frame_buffer,
                        action=action,
                        reward=self.clip_reward(reward),
                        new_state=next_frame_buffer,
                        terminal=done,
                    )

                    # Learn
                    if self.total_steps >= RANDOM_STEPS:
                        self.model.learn(self.memory)

                    # Accumulate rewards per step for each episode
                    score += reward
                    observation = next_obs
                    self.total_steps += 1
                else:
                    util.skip_frame(self)
                    # Learn ~ DDPG is off-policy so agent can learn on skipped frames
                    if self.total_steps >= RANDOM_STEPS:
                        self.model.learn(self.memory)

            # Reset noise correlation per episode
            self.model.ou_noise.reset()
            self.model.pink_noise.reset()

            # Bookkeep scores, goals and step history for plots
            self.score_history.append(score)
            self.goal_history.append(goals_found)
            self.step_history.append(self.total_steps)

            self.finish_episode(score, goals_found)

    # Handles end of episode (nice, clean and modular)
    def finish_episode(self, score, goals_found):
        self.episode_number += 1
        episode_finish_time_sec = time.perf_counter() - self.training_start_time
        episode_finish_time_min = episode_finish_time_sec / 60
        episode_finish_time_hour = episode_finish_time_min / 60
        avg_score = np.mean(self.score_history[-100:])
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.model.save(self.model_path)

        self.get_logger().info(
            "\nEpisode: {}, Steps: {}/{}, Training Time Elaspsed: {:.2f} \n Score: {:.2f}, Average Score: {:.2f}, Goals Found: {}".format(
                self.episode_number,
                self.total_steps,
                self.training_steps,
                episode_finish_time_min,
                score,
                avg_score,
                goals_found,
            )
        )

    def save_training_data(self):
        raw_data_dir = Path("src/active_slam_learning/raw_data")
        plot_dir = Path("src/active_slam_learning/plots")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            raw_data_dir / "ddpg_scores.npy",
            np.array(self.score_history),
        )
        np.save(
            raw_data_dir / "ddpg_steps.npy",
            np.array(self.step_history),
        )
        np.save(raw_data_dir / "ddpg_goals_found.npy", np.array(self.goal_history))
        np.save(raw_data_dir / "ddpg_collision.npy", np.array(self.collision_history))

        self.get_logger().info(
            "\n\n\nTraining has finished! raw data is available in the workspace src/training_data/raw_data/ "
        )

        # Plot the data
        util.plot_training_data(
            steps_file=raw_data_dir / "ddpg_steps.npy",
            scores_file=raw_data_dir / "dppg_scores.npy",
            goal_history_file=raw_data_dir / "ddpg_goals_found.npy",
            learning_plot_filename=plot_dir / "ddpg_learning_plot",
            goals_plot_filename=plot_dir / "ddpg_returns_plot",
            goals_title="ddpg goals found",
            learning_title="ddpg returns",
        )

    # Helper function for reward normalisation
    def update_reward_statistics(self, reward: float):
        # Incremental calculation of mean and variance
        self.reward_count += 1
        last_mean = self.reward_mean
        self.reward_mean += (reward - self.reward_mean) / self.reward_count
        self.reward_var += (reward - last_mean) * (reward - self.reward_mean)

    # Drastically increases performance
    def clip_reward(self, reward: float) -> float:
        if reward < -10:
            return -10
        elif reward > 10:
            return 10
        else:
            return reward

    def update_frame_buffer(self, observation):
        self.frame_buffer = np.roll(self.frame_buffer, -ENVIRONMENT_OBSERVATION_SPACE)
        self.frame_buffer[-ENVIRONMENT_OBSERVATION_SPACE:] = observation
        return self.frame_buffer

    def end_training(self):
        # Print results
        print(
            "\n\n\nResults: "
            + "\nGoals found: {}".format(sum(self.goal_history))
            + "\nCollisions:  {}".format(sum(self.collision_history))
            + "\nBest Score:  {:.2f}".format(self.best_score)
            + "\nTotal Time (hours): {:.2f}".format(
                ((time.perf_counter() - self.training_start_time) / 60) / 60,
            )
        )

        # Remind user of hyperparameters used
        print(
            "\n\nHyperparameters: "
            + "\nAlpha: {}".format(ALPHA_DDPG)
            + "\nBeta:  {}".format(BETA_DDPG)
            + "\nTau:   {}".format(TAU)
            + "\nGamma: {}".format(GAMMA_DDPG)
            + "\nActor Fully Connected Dims:  {}".format(ACTOR_DDPG_FC1)
            + "\nCritic Fully Connected Dims: {}".format(CRITIC_DDPG_FC1)
            + "\nBatch Size: {}".format(BATCH_SIZE_DDPG)
        )


def main():
    rclpy.init()
    learning_ddpg = LearningDDPG()
    rclpy.spin(learning_ddpg)
    learning_ddpg.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
