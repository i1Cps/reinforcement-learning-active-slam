import numpy as np
from torch._C import dtype
import torch.nn.functional as F
import torch as T
import torch.nn as nn
from active_slam_learning.learning.ddpg.noise import OUNoise_1, OUNoise_2
from active_slam_learning.learning.ddpg.replay_memory import ReplayBuffer
from active_slam_learning.learning.ddpg.networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        alpha,
        beta,
        tau,
        input_dims,
        n_actions,
        max_action,
        logger,
        gamma=0.99,
        max_size=1000000,
        fc1_dims=400,
        fc2_dims=300,
        batch_size=64,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.logger = logger
        self.time_step = 0
        self.n_actions = n_actions
        self.max_action = max_action

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = ActorNetwork(
            input_dims, fc1_dims, fc2_dims, n_actions, max_action, alpha, name="actor"
        )

        self.critic = CriticNetwork(
            input_dims, fc1_dims, fc2_dims, n_actions, beta, name="critic"
        )

        self.target_actor = ActorNetwork(
            input_dims,
            fc1_dims,
            fc2_dims,
            n_actions,
            max_action,
            alpha,
            name="target_actor",
        )

        self.target_critic = CriticNetwork(
            input_dims, fc1_dims, fc2_dims, n_actions, beta, name="target_critic"
        )
        self.update_network_parameters(tau=1)

    # Use the actor network to generate an action given a state,  this network is a representation of our policy
    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation[np.newaxis, :], dtype=T.float).to(
                self.actor.device
            )
            mu = self.actor.forward(state).to(self.actor.device).cpu().numpy()[0]
            noise = np.random.normal(0, self.max_action * 0.1, size=self.n_actions)
            return (mu + noise).clip(-self.max_action, self.max_action)

    def choose_random_action(self):
        return np.random.normal(0, self.max_action * 0.1, size=self.n_actions).clip(
            -self.max_action, self.max_action
        )

    def store_transition(self, state, action, reward, new_state, terminal):
        self.memory.store_transition(state, action, reward, new_state, terminal)

    def learn(self):
        if not self.memory.is_ready(self.batch_size):
            return

        (
            states,
            actions,
            rewards,
            next_states,
            terminals,
        ) = self.memory.sample_memory(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.actor.device)
        terminals = T.tensor(terminals).to(self.actor.device)

        # ------------------- Update critic -------------------- #

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

        # -------------- Update actor -------------------------#

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

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_state_dict[name].clone()
            )

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
