import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    """
    Critic Network for DDPG, evaluates the action-value function Q(s, a).
    """

    def __init__(
        self,
        input_dims: int,
        learning_rate: float,
        fc1: int,
        fc2: int,
    ):
        super(CriticNetwork, self).__init__()

        # input_dims = obs_space_dims + action_space_dims
        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.q = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        # There are many variations of the critic network in DDPG, I went with the TD3 style
        state_action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        state_action_value = F.relu(self.fc2(state_action_value))
        return self.q(state_action_value)


class ActorNetwork(nn.Module):
    """
    Actor Network for DDPG, determines the best action to take given a state.
    """

    def __init__(
        self,
        input_dims: int,
        learning_rate: float,
        fc1: int,
        fc2: int,
        n_actions: int,
        max_actions: np.ndarray,
    ):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.mu = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        self.max_actions = T.tensor(max_actions, dtype=T.float, device=self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = T.tanh(self.mu(x))
        return self.max_actions * x
