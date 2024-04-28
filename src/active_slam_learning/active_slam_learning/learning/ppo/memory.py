import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.new_states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []
        self.probs = []

        self.batch_size = batch_size

    def recall(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.terminals),
            np.array(self.probs),
        )

    # Generate batches, but we will return the indices for the learn function to index.
    # Reason being is that the learn function will first call "recall"

    def generate_batches(self):
        n_states = len(self.states)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [
            indices[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(n_batches)
        ]
        return batches

    def store_memory(self, state, action, reward, next_state, terminal, prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(terminal)
        self.probs.append(prob)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []
        self.probs = []
