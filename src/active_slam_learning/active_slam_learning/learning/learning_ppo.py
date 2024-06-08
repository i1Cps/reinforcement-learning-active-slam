from pathlib import Path
import time
import torch as T
import rclpy
from rclpy.node import Node

import numpy as np

from std_srvs.srv import Empty

from active_slam_interfaces.srv import StepEnv, ResetEnv
from active_slam_learning.learning.ppo.agent import Agent
from active_slam_learning.learning.ppo.memory import PPOMemory
from active_slam_learning.common import utilities as util
from active_slam_learning.common.settings import (
    ENTROPY_COEFFICIENT,
    GAE_LAMBDA,
    MODEL_PATH,
    ALPHA_PPO,
    BETA_PPO,
    FRAME_BUFFER_DEPTH,
    FRAME_BUFFER_SKIP,
    GAMMA_PPO,
    ACTOR_PPO_FC1,
    ACTOR_PPO_FC2,
    CRITIC_PPO_FC1,
    CRITIC_PPO_FC2,
    MAX_CONTINUOUS_ACTIONS,
    ENVIRONMENT_OBSERVATION_SPACE,
    ENVIRONMENT_ACTION_SPACE,
    MAX_SCAN_DISTANCE,
    N_EPOCHS,
    NUM_MINI_BATCHES,
    POLICY_CLIP,
    TRAINING_STEPS,
    LOAD_MODEL,
    TRAJECTORY,
)


class LearningPPO(Node):
    def __init__(self):
        super().__init__("learning_ppo")

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
        self.total_steps = 0
        self.episode_number = 0

        self.score_history = []
        self.step_history = []
        self.goal_history = []
        self.collision_history = []
        self.best_score = -np.Infinity

        self.training_start_time = time.perf_counter()
        self.training_steps = TRAINING_STEPS
        self.load_model = LOAD_MODEL

        # Create Directory in user system
        self.model_path = Path(MODEL_PATH)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Frame stacking
        self.stack_depth = FRAME_BUFFER_DEPTH
        self.frame_skip = FRAME_BUFFER_SKIP
        self.current_frame = 0
        self.frame_buffer = np.full(
            (self.stack_depth * ENVIRONMENT_OBSERVATION_SPACE),
            MAX_SCAN_DISTANCE,
            dtype=np.float32,
        )

        # Network Dimensions
        self.actor_dims = ENVIRONMENT_OBSERVATION_SPACE * self.stack_depth
        self.critic_dims = ENVIRONMENT_OBSERVATION_SPACE * self.stack_depth

        # Reward normalisation
        self.reward_mean = 0
        self.reward_var = 1
        self.reward_count = 1

    def initialise_model(self) -> Agent:
        model = Agent(
            actor_dims=self.actor_dims,
            critic_dims=self.critic_dims,
            n_actions=ENVIRONMENT_ACTION_SPACE,
            alpha=ALPHA_PPO,
            beta=BETA_PPO,
            entropy_coefficient=ENTROPY_COEFFICIENT,
            gae_lambda=GAE_LAMBDA,
            policy_clip=POLICY_CLIP,
            n_epochs=N_EPOCHS,
            gamma=GAMMA_PPO,
            actor_fc1=ACTOR_PPO_FC1,
            actor_fc2=ACTOR_PPO_FC2,
            critic_fc1=CRITIC_PPO_FC1,
            critic_fc2=CRITIC_PPO_FC2,
        )

        if self.load_model:
            model.load(self.model_path)
        return model

    def initialise_memory(self) -> PPOMemory:
        return PPOMemory(
            T=TRAJECTORY,
            input_dims=self.actor_dims,
            num_mini_batch=NUM_MINI_BATCHES,
            n_actions=ENVIRONMENT_ACTION_SPACE,
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
        trajectory_len = 0
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
            action = self.model.choose_action(self.frame_buffer)

            while not done:
                current_frame += 1
                if current_frame % self.frame_skip == 0:
                    # Choose actions and convert to appropriate action space
                    action, prob = self.model.choose_action(self.frame_buffer)
                    adapated_action = self.action_adapter(
                        action, MAX_CONTINUOUS_ACTIONS
                    )

                    # Step the environment
                    next_obs, reward, terminal, truncated, info = util.step(
                        self, adapated_action
                    )

                    # book keep goals found per episode
                    goals_found += info["goal_found"]

                    # Check for episode termination
                    done = terminal or truncated
                    self.collision_history.append(int(terminal))

                    clipped_reward = self.clip_reward(reward)

                    # Store the current state of the buffer
                    current_frame_buffer = self.frame_buffer.copy()
                    # Update the frame buffer with next_obs and store it too
                    next_frame_buffer = self.update_frame_buffer(next_obs)

                    # Store the transition in a memory buffer for sampling
                    self.memory.store_memory(
                        state=current_frame_buffer,
                        action=action,
                        reward=clipped_reward,
                        next_state=next_frame_buffer,
                        terminal=done,
                        prob=prob,
                    )

                    self.total_steps += 1
                    trajectory_len += 1

                    if trajectory_len % TRAJECTORY == 0:
                        self.model.learn(self.memory)
                        trajectory_len = 0

                    # Accumulate rewards per step for each episode
                    score += reward
                    observation = next_obs
                else:
                    util.skip_frame(self)

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
            raw_data_dir / "ppo_scores.npy",
            np.array(self.score_history),
        )
        np.save(
            raw_data_dir / "ppo_steps.npy",
            np.array(self.step_history),
        )
        np.save(raw_data_dir / "ppo_goals_found.npy", np.array(self.goal_history))
        np.save(raw_data_dir / "ppo_collision.npy", np.array(self.collision_history))

        self.get_logger().info(
            "\n\n\nTraining has finished! raw data is available in the workspace src/training_data/raw_data/ "
        )

        util.plot_training_data(
            steps_file=raw_data_dir / "ppo_steps.npy",
            scores_file=raw_data_dir / "ppo_scores.npy",
            goal_history_file=raw_data_dir / "ppo_goals_found.npy",
            learning_plot_filename=plot_dir / "ppo_learning_plot",
            goals_plot_filename=plot_dir / "ppo_goals_plot",
            goals_title="ppo goals found",
            learning_title="ppo returns",
        )

    # Drastically increases performance
    def clip_reward(self, reward: float) -> float:
        if reward < -1:
            return -1
        elif reward > 1:
            return 1
        else:
            return reward

    # Convert action range 0-1 -> -max_action, max_action
    def action_adapter(self, a, max_a):
        return 2 * (a - 0.5) * max_a

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
            + "\nAlpha: {}".format(ALPHA_PPO)
            + "\nBeta:  {}".format(BETA_PPO)
            + "\nGamma: {}".format(GAMMA_PPO)
            + "\nPolicy Clip: {}".format(POLICY_CLIP)
            + "\nGAE Lambda:  {}".format(GAE_LAMBDA)
            + "\nTrajectory:  {}".format(TRAJECTORY)
            + "\nNumber of Epochs:     {}".format(N_EPOCHS)
            + "\nEntropy Coefficient:  {}".format(ENTROPY_COEFFICIENT)
            + "\nNumber of Mini-Batches:  {}".format(NUM_MINI_BATCHES)
            + "\nActor Fully Connected Dims:  {}".format(ACTOR_PPO_FC1)
            + "\nCritic Fully Connected Dims: {}".format(CRITIC_PPO_FC1)
        )


def main():
    rclpy.init()
    learning_ppo = LearningPPO()
    rclpy.spin(learning_ppo)
    learning_ppo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
