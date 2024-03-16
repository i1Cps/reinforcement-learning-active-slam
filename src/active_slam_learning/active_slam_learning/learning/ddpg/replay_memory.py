import numpy as np


# This class is responsible for storing and sampling the agents memory of previous states, actions and rewards
# A key component to a majority of modern day reinforcement learning algorithms, PER and HER to come.
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, new_state, terminal):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = terminal

        self.mem_counter += 1

    def sample_memory(self, batch_size):
        max_memory = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_memory, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals
