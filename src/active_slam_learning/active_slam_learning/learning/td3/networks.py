import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# This network is equivalent to our Q Function, It evaluates state-action pairs.
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        """
        Initializes the Critic Network.

        Parameters:
        - beta (float): Learning rate for the optimizer.
        - input_dims (tuple): Dimensions of the input state.
        - fc1_dims (int): Number of neurons in the first fully connected layer.
        - fc2_dims (int): Number of neurons in the second fully connected layer.
        - n_actions (int): Number of actions in the action space.
        - name (str): Name of the network used for checkpointing.
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.beta = beta  # Learning rate
        self.chkpt_dir = (
            "./src/active_slam_learning/active_slam_learning/learning/ddpg/models"
        )
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + "_td3")

        # Define fully connected layers, Very MESSY <----------------------------
        self.fc1 = nn.Linear(self.input_dims[0], self.fc1_dims)  # state layer
        self.fc2 = nn.Linear(self.fc1_dims, int(self.fc2_dims / 2))  # state layer

        self.action_value = nn.Linear(n_actions, int(self.fc2_dims / 2))  # action layer

        self.fc3 = nn.Linear(self.fc2_dims, self.fc2_dims)  # state action layer

        # Output layer for Q-value estimation
        self.output = nn.Linear(self.fc2_dims, 1)

        # Initalize weights
        # First layer
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        # Second layer
        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        # Third layer
        f3 = 1.0 / np.sqrt(self.fc3.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)

        f_output = 0.003
        self.output.weight.data.uniform_(-f_output, f_output)
        self.output.bias.data.uniform_(-f_output, f_output)

        # Action layer
        f4 = 1.0 / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        # Optimizer for updating network parameters
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Device selection (GPU if available, otherwise CPU)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # Send model to device
        self.to(self.device)

    # Different to DDPG, we include and concatonate the action in the first layer
    def forward(self, state, action):
        """
        Forward pass through the critic network.

        Parameters:
        - state (torch.Tensor): Input state tensor.
        - action (torch.Tensor): Input action tensor.

        Returns:
        - q_output (torch.Tensor): Estimated Q-value for the given state-action pair.
        """
        ########## DEBUG
        """
        # Concatenate state and action tensors
        state_action = T.cat([state, action], dim=1)

        # Pass through first fully connected layer with ReLU activation
        state_action_value = F.relu(self.fc1(state_action))

        # Pass through second fully connected layer with ReLU activation
        state_action_value = F.relu(self.fc2(state_action_value))

        # Debug
        state_action_value = F.relu(self.fc3(state_action_value))
        """

        # Im trying the DDPG concatonate in the middle technique
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))

        action_value = F.relu(self.action_value(action))

        state_action_value = T.cat((state_value, action_value), dim=1)
        state_action_value = F.relu(self.fc3(state_action_value))
        # Output layer for Q-value estimation
        q_output = self.output(state_action_value)
        return q_output

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


# This network is a representation of our policy, It outputs actions values given a state, The TD3 paper argues its still capable of overestiamtion bias
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name):
        """
        Initializes the Actor Network.

        Parameters:
        - alpha (float): Learning rate for the optimizer.
        - input_dims (tuple): Dimensions of the input state.
        - fc1_dims (int): Number of neurons in the first fully connected layer.
        - fc2_dims (int): Number of neurons in the second fully connected layer.
        - n_actions (int): Number of actions in the action space.
        - name (str): Name of the network used for checkpointing.
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.chkpt_dir = (
            "./src/active_slam_learning/active_slam_learning/learning/td3/models"
        )
        self.checkpoint_file = os.path.join(self.chkpt_dir, name + "_td3")

        # Define fully connected layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Output layer for action probabilities
        self.mu = nn.Linear(
            self.fc2_dims, self.n_actions
        )  # n_actions because of continuous actions space

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

        # Optimizer for updating network parameters
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Device selection (GPU if available, otherwise CPU)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # Send model to device
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the actor network.

        Parameters:
        - state (torch.Tensor): Input state tensor.

        Returns:
        - actions (torch.Tensor): Output tensor representing actions.
        """
        # Pass through first fully connected layer with ReLU activation
        actions = F.relu(self.fc1(state))

        # Pass through second fully connected layer with ReLU activation
        actions = F.relu(self.fc2(actions))

        # Output layer with tanh activation to bound action probabilities between -1 and 1
        actions = T.tanh(self.mu(actions))

        return actions

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))
