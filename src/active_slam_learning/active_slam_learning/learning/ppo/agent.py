import torch as T
from torch.cuda import memory
from active_slam_learning.learning.ppo.memory import PPOMemory
from active_slam_learning.learning.ppo.networks import (
    ContinuousActorNetwork,
    ContinuousCriticNetwork,
)
import numpy as np


class Agent:
    def __init__(
        self,
        input_dims,
        n_actions,
        fc1_dims=256,
        fc2_dims=256,
        gamma=0.99,
        alpha=3e-4,
        entrophy_coefficient=1e-3,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entrophy_coefficient = entrophy_coefficient
        self.actor = ContinuousActorNetwork(
            input_dims=input_dims,
            n_actions=n_actions,
            fc1=fc1_dims,
            fc2=fc2_dims,
            alpha=alpha,
        )
        self.critic = ContinuousCriticNetwork(
            input_dims=input_dims, fc1=fc1_dims, fc2=fc2_dims, alpha=alpha
        )
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, reward, next_state, terminated, probs):
        self.memory.store_memory(state, action, reward, next_state, terminated, probs)

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(
                observation[np.newaxis, :], dtype=T.float, device=self.actor.device
            )

            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)

        return action.cpu().numpy().flatten(), probs.cpu().numpy().flatten()

    def calc_adv_and_returns(self, memories):
        states, next_states, rewards, terminated = memories
        with T.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            deltas = rewards + self.gamma * next_values - values
            deltas = deltas.cpu().flatten().numpy()
            terminated = terminated.cpu().numpy()
            adv = [0]
            for dlt, mask in zip(deltas[::-1], terminated[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]

            adv = T.tensor(adv).float().unsqueeze(1).to(self.actor.device)
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)
        return adv, returns

    def learn(self):
        (
            state_array,
            action_array,
            reward_array,
            next_state_array,
            terminated_array,
            old_prob_array,
        ) = self.memory.recall()
        state_array = T.tensor(state_array, dtype=T.float, device=self.actor.device)
        action_array = T.tensor(action_array, dtype=T.float, device=self.actor.device)

        # Pree the unsqueeze
        reward_array = T.tensor(
            reward_array, dtype=T.float, device=self.actor.device
        ).unsqueeze(1)

        next_state_array = T.tensor(
            next_state_array, dtype=T.float, device=self.actor.device
        )
        terminated_array = T.tensor(
            terminated_array, dtype=T.float, device=self.actor.device
        )
        old_prob_array = T.tensor(
            old_prob_array, dtype=T.float, device=self.actor.device
        )

        adv, returns = self.calc_adv_and_returns(
            (state_array, next_state_array, reward_array, terminated_array)
        )

        for epoch in range(self.n_epochs):
            batches = self.memory.generate_batches()
            for batch in batches:
                states = state_array[batch]
                old_probs = old_prob_array[batch]
                actions = action_array[batch]

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(
                    new_probs.sum(1, keepdim=True) - old_probs.sum(1, keepdim=True)
                )
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * adv[batch]
                )

                entropy = dist.entropy().sum(1, keepdim=True)
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)
                actor_loss -= self.entrophy_coefficient * entropy
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()

                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def save_models(self):
        self.critic.save_checkpoint()
        self.actor.save_checkpoint()

    def load_models(self):
        self.critic.load_checkpoint()
        self.actor.load_checkpoint()
