import copy
import numpy as np
import torch.nn.functional as F
import torch as T
from active_slam_learning.learning.ddpg.replay_memory import ReplayBuffer
from active_slam_learning.learning.ddpg.networks import ActorNetwork, CriticNetwork
from active_slam_learning.learning.ddpg.noise import (
    DifferentialDriveOUNoise,
    DifferentialDrivePinkNoise,
)


class Agent:
    """
    Agent class for DDPG, encapsulating the actor and critic networks,
    along with methods for training and action selection.
    """

    def __init__(
        self,
        actor_dims: int,
        critic_dims: int,
        n_actions: int,
        max_actions: np.ndarray,
        min_actions: np.ndarray,
        alpha: float,
        beta: float,
        tau: float,
        gamma: float = 0.99,
        actor_fc1: int = 256,
        actor_fc2: int = 256,
        critic_fc1: int = 256,
        critic_fc2: int = 256,
        batch_size: int = 64,
    ):
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_actions = max_actions
        self.gamma = gamma

        self.actor = ActorNetwork(
            input_dims=actor_dims,
            learning_rate=alpha,
            fc1=actor_fc1,
            fc2=actor_fc2,
            n_actions=n_actions,
            max_actions=max_actions,
        )

        self.critic = CriticNetwork(
            input_dims=critic_dims,
            learning_rate=beta,
            fc1=critic_fc1,
            fc2=critic_fc2,
        )

        self.target_actor = ActorNetwork(
            input_dims=actor_dims,
            learning_rate=alpha,
            fc1=actor_fc1,
            fc2=actor_fc2,
            n_actions=n_actions,
            max_actions=max_actions,
        )

        self.target_critic = CriticNetwork(
            input_dims=critic_dims,
            learning_rate=beta,
            fc1=critic_fc1,
            fc2=critic_fc2,
        )

        self.ou_noise = DifferentialDriveOUNoise(
            mean=0,
            theta=0.15,
            sigma=0.2,
            dt=0.01,
            max_angular=max_actions[1],
            min_angular=min_actions[1],
            max_linear=max_actions[0],
            min_linear=min_actions[0],
        )

        self.pink_noise = DifferentialDrivePinkNoise(
            alpha=1,
            steps=5000,
            max_angular=max_actions[1],
            min_angular=min_actions[1],
            max_linear=max_actions[0],
            min_linear=min_actions[0],
        )

        # Set network and target network weights to equal each other
        self.update_network_parameters(tau=1)

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
        with T.no_grad():
            # Convert observation to tensor
            state = T.tensor(observation[np.newaxis, :], dtype=T.float).to(
                self.actor.device
            )
            # Generate actions using the actor network
            mu = self.actor.forward(state).to(self.actor.device).cpu().numpy()[0]
            # Add pink noise
            # noise = self.pink_noise() * 0.1
            noise = np.random.normal(0, self.max_actions * 0.1, size=self.n_actions)
            # Clip
            return (mu + noise).clip(-self.max_actions, self.max_actions)

    def choose_random_action(self) -> np.ndarray:
        # Generate action using Ornstein–Uhlenbeck noise
        return self.ou_noise()

    def learn(self, memory: ReplayBuffer):
        # Check if enough memory is in the buffer before sampling
        if not memory.is_ready(self.batch_size):
            return

        # Sample a batch of memories
        (
            states,
            actions,
            rewards,
            next_states,
            terminals,
        ) = memory.sample_memory(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.actor.device)
        terminals = T.tensor(terminals).to(self.actor.device)

        # ------------------- Update critic network -------------------- #

        with T.no_grad():
            next_actions = self.target_actor(next_states)
            Q_critic_next = self.target_critic(next_states, next_actions)
            Q_critic_next[terminals] = 0.0
            Q_critic_next = Q_critic_next.view(-1)

            Q_target = rewards + self.gamma * Q_critic_next
            Q_target = Q_target.view(self.batch_size, 1)

        Q_critic = self.critic.forward(states, actions)

        # Loss calculation
        critic_loss = F.mse_loss(Q_critic, Q_target)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # -------------- Update actor network -------------------------#

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # -------------------- Update target networks ---------------#

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filepath) -> None:
        T.save(self.actor.state_dict(), filepath / "ddpg_actor")
        T.save(self.actor.optimizer.state_dict(), filepath / "ddpg_actor_optimiser")

        T.save(self.critic.state_dict(), filepath / "ddpg_critic")
        T.save(self.critic.optimizer.state_dict(), filepath / "ddpg_critic_optimiser")

        print("... saving checkpoint ...")

    def load(self, filepath) -> None:
        self.actor.load_state_dict((T.load(filepath / "ddpg_actor")))
        self.actor.optimizer.load_state_dict(T.load(filepath / "ddpg_actor_optimiser"))
        self.target_actor = copy.deepcopy(self.actor)

        self.critic.load_state_dict((T.load(filepath / "ddpg_critic")))
        self.critic.optimizer.load_state_dict(
            T.load(filepath / "ddpg_critic_optimiser")
        )
        self.target_critic = copy.deepcopy(self.critic)

        print("... loading checkpoint ...")
