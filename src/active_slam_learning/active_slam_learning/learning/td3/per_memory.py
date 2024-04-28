from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Node:
    """
    Node class representing a node in the sum tree.

    Attributes:
        - value (float): Priority value of the node.
        - total (float): Total priority value of the subtree rooted at this node.
    """

    value: float = 0.01
    total: float = 0.01

    def update_priority(self, priority: float):
        """
        Update the priority value of the node and return the change in priority.

        Parameters:
            - priority (float): New priority value.

        Returns:
            - delta (float): Change in priority value.
        """

        delta = priority - self.value
        self.value = priority
        self.total += delta
        return delta

    def update_total(self, delta: float):
        """
        Update the total priority value of the subtree rooted at this node.

        Parameters:
            - delta (float): Change in priority value to propagate.
        """

        self.total += delta


class SumTree:
    """
    SumTree class implementing the sum tree data structure for prioritized experience replay.

    Attributes:
        - max_size (int): Maximum size of the memory.
        - batch_size (int): Size of the batch sampled from memory.
        - alpha (float): Alpha parameter for prioritization.
        - beta (float): Beta parameter for importance sampling.
        - alpha_start (float): Initial value of alpha.
        - beta_start (float): Initial value of beta.
        - memory_counter (int): Counter to keep track of the number of experiences stored in memory.
        - sum_tree (list): List representing the sum tree.
        - states (ndarray): Array to store states.
        - actions (ndarray): Array to store actions.
        - rewards (ndarray): Array to store rewards.
        - new_states (ndarray): Array to store new states.
        - terminals (ndarray): Array to store terminal flags.
    """

    def __init__(
        self,
        input_shape,
        n_actions,
        max_size: int = 1_00_000,
        batch_size: int = 32,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        self.memory_counter = 0
        self.max_size = max_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.alpha_start = alpha
        self.beta_start = beta

        self.sum_tree = []
        self.states = np.zeros(shape=(max_size, *input_shape), dtype=np.float32)
        self.actions = np.zeros(shape=(max_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros(shape=(max_size,), dtype=np.float32)
        self.new_states = np.zeros(shape=(max_size, *input_shape), dtype=np.float32)
        self.terminals = np.zeros(shape=(max_size,), dtype=np.bool_)

    def _insert(self, transition: List):
        """
        Insert a new transition into the memory.

        Parameters:
            - transition (list): Experience tuple (state, action, reward, new_state, terminal).
        """

        state, action, reward, new_state, terminal = transition
        index = self.memory_counter % self.max_size
        # Store the transition in memory
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.terminals[index] = terminal
        if self.memory_counter < self.max_size:
            self.sum_tree.append(Node())
        self.memory_counter += 1

    def store_transition(self, transition):
        self._insert(transition)

    def _calculate_parents(self, index: int):
        """
        Calculate the chain of parent indices for a given index in the sum tree.

        Parameters:
            - index (int): Index of the node in the sum tree.

        Returns:
            - parents (list): List of parent indices.
        """
        parents = []
        while index > 0:
            parents.append(int((index - 1) // 2))
            index = int((index - 1) // 2)
        return parents

    def update_priorities(self, indices: List, priorities: List):
        self._propagate_changes(indices, priorities)

    def _propagate_changes(self, indices: List, priorities: List):
        """
        Update priorities and total priority values in the sum tree.

        Parameters:
            - indices (list): List of indices corresponding to sampled experiences.
            - priorities (list): List of updated priorities for each sampled experience.
        """
        for idx, p in zip(indices, priorities):
            # Update priority and calculate change in priority
            delta = self.sum_tree[idx].update_priority(p + 1e-3**self.alpha)
            # Calculate parent indices and propagate changes
            parents = self._calculate_parents(idx)
            for parent in parents:
                self.sum_tree[parent].update_total(delta)

    def _sample(self):
        """
        Sample experiences from the sum tree.

        Returns:
            - samples (list): List of sampled indices.
            - probs (list): List of probabilities corresponding to the sampled experiences.
        """
        total_weight = self.sum_tree[0].total

        if total_weight == 0.01:
            # Uniform sampling if sum tree is empty
            samples = np.random.choice(self.batch_size, self.batch_size, replace=False)
            probs = [1 / self.batch_size for _ in range(self.batch_size)]
            return samples, probs

        samples, probs, n_samples = [], [], 1
        index = self.memory_counter % self.max_size - 1
        samples.append(index)
        probs.append(self.sum_tree[index].value / self.sum_tree[0].total)
        while n_samples < self.batch_size:
            index = 0
            target = total_weight * np.random.random()
            while True:
                # Traverse the sum tree to find the sample index
                left = 2 * index + 1
                right = 2 * index + 2
                if left > len(self.sum_tree) - 1 or right > len(self.sum_tree) - 1:
                    break
                left_sum = self.sum_tree[left].total
                if target < left_sum:
                    index = left
                    continue
                target -= left_sum
                right_sum = self.sum_tree[right].total
                if target < right_sum:
                    index = right
                    continue
                target -= right_sum
                break
            samples.append(index)
            n_samples += 1
            probs.append(self.sum_tree[index].value / self.sum_tree[0].total)
        return samples, probs

    def sample(self):
        """
        Sample experiences from the sum tree and calculate importance sampling weights.

        Returns:
            - mems (list): List of sampled experiences.
            - samples (list): List of sampled indices.
            - weights (ndarray): Importance sampling weights for the sampled experiences.
        """
        samples, probs = self._sample()
        weights = self._calculate_weights(probs)
        mems = [
            self.states[samples],
            self.actions[samples],
            self.rewards[samples],
            self.new_states[samples],
            self.terminals[samples],
        ]
        return mems, samples, weights

    def _calculate_weights(self, probs: List):
        """
        Calculate importance sampling weights.

        Parameters:
            - probs (list): List of probabilities corresponding to the sampled experiences.

        Returns:
            - weights (ndarray): Importance sampling weights for the sampled experiences.
        """

        weights = np.array(
            [(1 / self.memory_counter * 1 / prob) ** self.beta for prob in probs]
        )
        weights *= 1 / max(weights)
        return weights

    def ready(self):
        """
        Check if the memory contains enough experiences for sampling.

        Returns:
            - ready (bool): True if memory contains enough experiences, False otherwise.
        """
        return self.memory_counter >= self.batch_size

    def anneal_beta(self, ep: int, ep_max: int):
        """
        Anneal the beta parameter for importance sampling.

        The beta parameter controls how much importance sampling affects the updates of the Q-values
        during training. By annealing beta, we gradually increase its influence over time, allowing
        the agent to focus more on experiences with higher TD-errors.

        Parameters:
            - ep (int): Current episode number.
            - ep_max (int): Maximum number of episodes.
        """
        self.beta = self.beta_start + ep / ep_max * (1 - self.beta_start)

    def anneal_alpha(self, ep: int, ep_max: int):
        """
        Anneal the alpha parameter for prioritization.

        The alpha parameter controls how much prioritization is applied to the replay buffer
        during sampling. By annealing alpha, we gradually decrease its influence over time, reducing
        the degree of prioritization as training progresses.

        Parameters:
            - ep (int): Current episode number.
            - ep_max (int): Maximum number of episodes.
        """
        self.alpha = self.alpha_start * (1 - ep / ep_max)
