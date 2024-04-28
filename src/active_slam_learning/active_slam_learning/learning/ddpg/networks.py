import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, beta, name):
        super(CriticNetwork, self).__init__()
        self.beta = beta
        self.chkpt_dir = (
            "./src/active_slam_learning/active_slam_learning/learning/ddpg/models"
        )
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + "_ddpg")

        # Three Fully Connected Layers
        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)  # State-Action Value ~ Q Value

        # Adam Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    # Concatanate State and Action in first layer
    def forward(self, state, action):
        state_action_value = F.relu(self.fc1(T.cat([state, action], 1)))
        state_action_value = F.relu(self.fc2(state_action_value))
        return self.q(state_action_value)

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(
        self, input_dims, fc1_dims, fc2_dims, n_actions, max_action, alpha, name
    ):
        super(ActorNetwork, self).__init__()
        self.alpha = alpha
        self.chkpt_dir = (
            "./src/active_slam_learning/active_slam_learning/learning/ddpg/models"
        )
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + "_ddpg")

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        # Adam Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        self.max_action = T.tensor(max_action, dtype=T.float, device=self.device)

    # Get Action from State
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = T.tanh(self.mu(x))
        return self.max_action * x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
