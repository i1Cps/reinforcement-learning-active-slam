import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, beta, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.beta = beta
        self.chkpt_dir = (
            "./src/active_slam_learning/active_slam_learning/learning/ddpg/models"
        )
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + "_ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Batch Normalization vs Layer Normalization
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Paper says not to include actions until second hidden layer
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        # We want a single value for our chosen action given the state Q(s,a) = value
        self.q = nn.Linear(self.fc2_dims, 1)

        # Initalize weights
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1.0 / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta, weight_decay=0.01)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    # Remember Critic takes action and state and spits out a value based on how good the state, action pair is
    def forward(self, state, action):
        # Calculate state value first, then add action in the second hidden layer
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        # Activate after batch normalization to preserve negative state values
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        # Introduce action into critique
        action_value = self.action_value(action)
        # Add instead of concatonate to preserve dimensionality
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, alpha, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.chkpt_dir = (
            "./src/active_slam_learning/active_slam_learning/learning/ddpg/models"
        )
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + "_ddpg")

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Batch Normalization vs Layer Normalization
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        # self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        # We want a single value for our chosen action given the state Q(s,a) = value
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        # Initalize weights
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    # Takes state and spits out action, network to model policy
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # Tan hyperbolic is +1 to -1, make sure to multiply by custom bounds if you need different action high and low
        x = T.tanh(self.mu(x))

        return x

    # Saving functions
    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
