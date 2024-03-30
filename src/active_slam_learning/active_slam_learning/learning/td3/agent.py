import numpy as np
from numpy.core.multiarray import zeros
import torch.nn.functional as F
import torch as T
from active_slam_learning.learning.td3.noise import OUActionNoise, OUNoise
from active_slam_learning.learning.td3.replay_memory import ReplayBuffer
from active_slam_learning.learning.td3.networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        max_action_values,
        logger,
        gamma=0.99,
        update_actor_interval=2,
        warmup=1000,
        n_actions=2,
        max_size=1000000,
        layer1_size=400,
        layer2_size=300,
        batch_size=100,
        noise=0.1,
    ):
        """
        Initialize the Agent object.

        Parameters:
        - alpha (float): Actor network learning rate.
        - beta (float): Critic network learning rate.
        - input_dims (tuple): Input dimensions of the environment.
        - tau (float): Soft update parameter.
        - max_action_values (array): Array of maximum values for each continuous action (symmertrical about 0)
        - gamma (float): Discount factor for future rewards.
        - update_actor_interval (int): Interval for updating the actor network.
        - warmup (int): Number of steps before using the learned policy.
        - n_actions (int): Number of actions in the action space.
        - max_size (int): Maximum size of the replay buffer.
        - layer1_size (int): Size of the first hidden layer in networks.
        - layer2_size (int): Size of the second hidden layer in networks.
        - batch_size (int): Batch size for training networks.
        - noise (float): Noise level for exploration.
        """
        # Initialize parameters
        self.gamma = gamma
        self.tau = tau
        self.memory_size = max_size
        self.max_action_values = max_action_values
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        # OU noise has correlation, reset action_noise every episode let learning noise run free
        self.action_noise = OUActionNoise(mu=np.zeros(n_actions))
        self.action_noise2 = OUNoise(
            action_space=n_actions, max_sigma=0.1, min_sigma=0.1, decay_period=8000000
        )
        self.noise_step = 0

        # Initialize actor networks
        self.actor = ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions,
            name="actor",
        )
        self.target_actor = ActorNetwork(
            alpha, input_dims, layer1_size, layer2_size, n_actions, name="target_actor"
        )

        # Initialize critic networks
        self.critic_1 = CriticNetwork(
            beta, input_dims, layer1_size, layer2_size, n_actions, name="critic_1"
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, layer1_size, layer2_size, n_actions, name="critic_2"
        )
        self.target_critic_1 = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions,
            name="target_critic_1",
        )
        self.target_critic_2 = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions,
            name="target_critic_2",
        )

        # Initialize networks' target parameters
        self.update_network_parameters(tau=1)

    # We pick randomly until warmup is finished
    def choose_action(self, observation):
        """
        Choose an action using the policy.

        During the initial exploration phase (warmup), actions are chosen randomly.
        Once warmup is finished, actions are selected using the actor network's policy.

        Parameters:
        - observation (array): Current state observation.

        Returns:
        - action (array): Action selected by the policy.
        """
        # Exploration during the initial phase (warmup)
        if self.time_step < self.warmup:
            # Randomly sample action values
            mu = T.tensor(
                np.random.normal(scale=0.2, size=(self.n_actions,)),
                device=self.actor.device,
            )
        else:
            # Use the actor network to select actions based on the current state
            state = T.tensor(observation, dtype=T.float, device=self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        # Add noise to the selected action for exploration
        # mu_prime = mu + T.tensor(
        #    np.random.normal(scale=self.noise), dtype=T.float, device=self.actor.device
        # )  # This is naive normal noise, we like OU or pink noise

        # The noise should not be larger than half the max and min actions
        noise = T.tensor(
            self.action_noise2.get_noise(self.time_step),
            dtype=T.float,
            device=self.actor.device,
        )
        noise = T.tensor(self.action_noise(), dtype=T.float, device=self.actor.device)
        mu_prime = T.clamp(T.add(mu, noise), -1.0, 1.0)

        # Clip the action values to ensure they fall within the action space bounds
        # Clip both continuous actions seperately

        # Normalize for env output,  then clamp to max values
        # mu_prime[0] = (mu_prime[0] * self.max_action_values[0]).clamp(
        #    -self.max_action_values[0], self.max_action_values[0]
        # )
        # mu_prime[1] = (mu_prime[1] * self.max_action_values[1]).clamp(
        #    -self.max_action_values[1], self.max_action_values[1]
        # )

        # Increment the time step counter
        self.time_step += 1
        self.noise_step += 1

        # Convert the action tensor to numpy array and return
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, terminal):
        """
        Store a transition tuple in memory buffer.

        Parameters:
        - state (array): Current state.
        - action (array): Action taken.
        - reward (float): Reward received.
        - next_state (array): Next state after taking action.
        - terminal (bool): Flag indicating terminal state.
        """

        # Store experience in memory buffer
        self.memory.store_transition(state, action, reward, next_state, terminal)

    def learn(self):
        """
        Update actor and critic networks based on experiences sampled from the replay buffer.

        This function implements the learning process outlined in the original Twin Delayed DDPG (TD3) paper.
        It samples experiences from the replay buffer, computes target Q-values using the target critic networks,
        and updates the critic networks to minimize the Mean Squared Error (MSE) between the predicted Q-values
        and the target Q-values. The actor network is updated to maximize the Q-value estimated by the critic.
        Target networks are updated using soft target network updates for stability.
        """

        # Check if enough samples in memory buffer
        if self.memory.mem_counter < self.memory_size / 3:
            return

        # Sample experiences from memory buffer
        states, actions, rewards, next_states, terminals = self.memory.sample_memory(
            self.batch_size
        )

        # Convert the sampled experiences into tensors
        states = T.tensor(states, dtype=T.float, device=self.critic_1.device)
        actions = T.tensor(actions, dtype=T.float, device=self.critic_1.device)
        rewards = T.tensor(rewards, dtype=T.float, device=self.critic_1.device)
        next_states = T.tensor(next_states, dtype=T.float, device=self.critic_1.device)
        terminals = T.tensor(terminals, device=self.critic_1.device)

        with T.no_grad():
            # Create noise for exploration
            noise = (T.randn_like(actions) * 0.2).clamp(-0.5, 0.5)
            # Sample actions from target actor network with added noise
            next_actions = (self.target_actor.forward(next_states) + noise).clamp(
                -1.0, 1.0
            )

            # Will comment this, sorry it looks messy
            # Max and min action[0] is currently 0.22, so scale then clip
            #

            # Below is the crux of the double dueling technique first established in DDQN

            # Compute target Q-values using target critic networks
            target_Q1 = self.target_critic_1.forward(next_states, next_actions)
            target_Q2 = self.target_critic_2.forward(next_states, next_actions)
            target_Q1[terminals] = 0.0
            target_Q2[terminals] = 0.0

            # Take the minimum of the two Q-values to reduce overestimation bias
            target_Q = T.min(target_Q1, target_Q2)

            # Compute target Q-values
            target_Q = rewards + self.gamma * target_Q.view(-1)
            target_Q = target_Q.view(self.batch_size, 1)

        # Compute current Q-values using critic networks
        current_Q1 = self.critic_1.forward(states, actions)
        current_Q2 = self.critic_2.forward(states, actions)

        # Compute critic loss
        critic_loss_Q1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_Q2 = F.mse_loss(current_Q2, target_Q)
        critic_loss = critic_loss_Q1 + critic_loss_Q2

        # Zero the gradients of the critic networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        # Backpropagate and update critic networks
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Increment the learning step counter
        self.learn_step_counter += 1

        # Perform delayed updates for actor network
        if self.learn_step_counter % self.update_actor_iter != 0:
            return

        # Compute actor loss
        actor_q1_loss = self.critic_1.forward(states, self.actor.forward(states))
        actor_loss = -T.mean(actor_q1_loss)

        # Zero the gradients of the actor network
        self.actor.optimizer.zero_grad()

        # Backpropagate and update actor network
        actor_loss.backward()
        self.actor.optimizer.step()

        # Perform delayed updates for target networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """
        Update target actor and critic networks by interpolating their parameters
        towards the parameters of the main actor and critic networks.

        Parameters:
        - tau (float): Interpolation parameter. If None, uses the default value.
        """

        # Set default value for tau if not provided
        if tau is None:
            tau = self.tau

        # Get named parameters for actor and critic networks
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()

        # Get named parameters for target actor and critic networks
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        # Create state dictionaries for actor and critic parameters
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)

        # Create state dictionaries for target actor and critic parameters
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        # Interpolate target critic 1 parameters
        for name in critic_1_state_dict:
            critic_1_state_dict[name] = (
                tau * critic_1_state_dict[name].clone()
                + (1 - tau) * target_critic_1_state_dict[name].clone()
            )

        # Interpolate target critic 2 parameters
        for name in critic_2_state_dict:
            critic_2_state_dict[name] = (
                tau * critic_2_state_dict[name].clone()
                + (1 - tau) * target_critic_2_state_dict[name].clone()
            )

        # Interpolate target actor parameters
        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        # Load interpolated parameters into target networks
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

    def reset_noise(self):
        self.action_noise2.reset()
        self.action_noise.reset()
        self.noise_step = 0
