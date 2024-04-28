import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        """
        Initialize a replay buffer for storing transitions.

        Parameters:
        - max_size (int): Maximum capacity of the replay buffer.
        - input_shape (tuple): Shape of the state observations.
        - n_actions (int): Number of possible actions.

        Initializes arrays to store states, actions, rewards, new states, and terminal flags.
        """
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, next_state, terminal):
        """
        Store a transition tuple in the replay buffer.

        Parameters:
        - state (array): Current state observation.
        - action (array): Action taken in the current state.
        - reward (float): Reward received after taking the action.
        - next_state (array): Next state observation after taking the action.
        - terminal (bool): Flag indicating whether the episode terminated after this transition.

        Stores the transition tuple (state, action, reward, next_state, terminal) in the replay buffer.
        """
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = terminal

        self.mem_counter += 1

    def sample_memory(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Parameters:
        - batch_size (int): Number of transitions to sample.

        Returns:
        - states (array): Batch of current state observations.
        - actions (array): Batch of actions taken.
        - rewards (array): Batch of rewards received.
        - new_states (array): Batch of new state observations.
        - terminals (array): Batch of terminal flags.

        Randomly samples a batch of transitions from the replay buffer and returns them as separate arrays.
        """
        max_memory = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_memory, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminals
