import numpy as np
from typing import Tuple


class PPOMemory:
    def __init__(
        self,
        T: int,
        input_dims: int,
        num_mini_batch: int,
        n_actions: int,
    ):
        # Initialise arrays to store states, actions, rewards, next states, done flags, and probabilities
        self.states = np.zeros((T, input_dims), dtype=np.float32)
        self.actions = np.zeros((T, n_actions), dtype=np.float32)
        self.rewards = np.zeros(T, dtype=np.float32)
        self.next_states = np.zeros((T, input_dims), dtype=np.float32)
        self.terminals = np.zeros(T, dtype=np.float32)
        self.probs = np.zeros((T, n_actions), dtype=np.float32)

        # Initialise counters and configuration parameters
        self.memory_counter = 0
        self.num_mini_batch = num_mini_batch
        self.T = T
        self.input_dims = input_dims
        self.n_actions = n_actions

    def recall(self) -> Tuple:
        # Return the stored memory arrays
        return (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.terminals,
            self.probs,
        )

    def generate_batches(self):
        # Get the actual length of the episode based on stored rewards ~ (In case of vectorised environment addition)
        episode_length = len(self.rewards)
        batch_size = episode_length
        # Calculate the mini_batch_size based on the num_mini_batches we want
        mini_batch_size = batch_size // self.num_mini_batch

        # Generate shuffled indices for batching
        indices = np.arange(batch_size, dtype=np.int64)
        np.random.shuffle(indices)

        # Create mini-batches of indices
        mini_batches = [
            indices[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(self.num_mini_batch)
        ]
        return mini_batches

    def store_memory(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        prob: float,
    ):
        # Store the experience in the memory arrays
        index = self.memory_counter % self.T
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = terminal
        self.probs[index] = prob

        self.memory_counter += 1

    # Reset all memory arrays and the counter
    def clear_memory(self):
        self.states = np.zeros((self.T, self.input_dims), dtype=np.float32)
        self.actions = np.zeros((self.T, self.n_actions), dtype=np.float32)
        self.rewards = np.zeros(self.T, dtype=np.float32)
        self.next_states = np.zeros((self.T, self.input_dims), dtype=np.float32)
        self.terminals = np.zeros(self.T, dtype=np.bool_)
        self.probs = np.zeros((self.T, self.n_actions), dtype=np.float32)
        self.memory_counter = 0

    # Check if enough samples are collected to form the specified number of mini-batches
    def is_ready(self):
        return self.memory_counter >= self.num_mini_batch
