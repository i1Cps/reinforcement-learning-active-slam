import torch as T
import torch.nn.functional as F
from torch.distributions import Beta
import torch.optim as optim
import torch.nn as nn


# Models the value of a state
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate: float, fc1: int, fc2: int):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.v = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        return self.v(x)


# With PPO we sample actions using a distrbutions, Here we use the Beta distributions which takes parameters, alpha and beta
class ActorNetwork(nn.Module):
    def __init__(
        self, input_dims: int, learning_rate: float, fc1: int, fc2: int, n_actions: int
    ):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.alpha = nn.Linear(fc2, n_actions)
        self.beta = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor) -> Beta:
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        # Compute alpha parameter, ensure positivity
        alpha = F.relu(self.alpha(x)) + 1.0
        # Compute beta parameter, ensure positivity
        beta = F.relu(self.beta(x)) + 1.0
        # Create and return Beta distribution with alpha and beta
        dist = Beta(alpha, beta)
        return dist
