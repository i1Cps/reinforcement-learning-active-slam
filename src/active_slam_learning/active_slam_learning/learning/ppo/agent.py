import torch as T
from active_slam_learning.learning.ppo.memory import PPOMemory
from active_slam_learning.learning.ppo.networks import ActorNetwork, CriticNetwork
import numpy as np


class Agent:
    """
    Agent class for PPO, encapsulating the actor and critic networks,
    along with methods for training and action selection.
    """

    def __init__(
        self,
        actor_dims: int,
        critic_dims: int,
        n_actions: int,
        alpha: float,
        beta: float,
        entropy_coefficient: float,
        gae_lambda: float,
        policy_clip: float,
        n_epochs: int,
        gamma: float = 0.99,
        actor_fc1: int = 256,
        actor_fc2: int = 256,
        critic_fc1: int = 256,
        critic_fc2: int = 256,
    ):
        self.gamma = gamma  # Dicount factor for rewards
        self.gae_lambda = gae_lambda  # Lambda for GAE (Generalised Advantage Estimate)
        self.policy_clip = policy_clip  # Clipping parameter for PPO
        self.n_epochs = n_epochs  # Number of training epochs per update
        self.entropy_coefficient = (
            entropy_coefficient  # Coefficient for entropy regularisation
        )

        # Initialise the actor network
        self.actor = ActorNetwork(
            input_dims=actor_dims,
            learning_rate=alpha,
            n_actions=n_actions,
            fc1=actor_fc1,
            fc2=actor_fc2,
        )

        # Initialise the critic network
        self.critic = CriticNetwork(
            input_dims=critic_dims,
            learning_rate=beta,
            fc1=critic_fc1,
            fc2=critic_fc2,
        )

    def choose_action(self, observation: np.ndarray) -> tuple:
        with T.no_grad():
            # Convert observation to tensor and add batch dimension
            state = T.tensor(
                observation[np.newaxis, :], dtype=T.float, device=self.actor.device
            )
            # Get action distribution from actor and sample an action
            dist = self.actor(state)
            action = dist.sample()
            # Get log probability of sampled action
            probs = dist.log_prob(action)
        return action.cpu().numpy().flatten(), probs.cpu().numpy().flatten()

    def calc_adv_and_returns(self, memories: tuple) -> tuple:
        states, next_states, rewards, terminated = memories
        with T.no_grad():
            # Get the state values from critic network
            values = self.critic(states)
            next_values = self.critic(next_states)

            # Calculate deltas for GAE
            deltas = rewards + self.gamma * next_values - values
            deltas = deltas.cpu().flatten().numpy()
            terminated = terminated.cpu().numpy()

            # Initialise advantage array and GAE variable
            adv = np.zeros_like(deltas)
            gae = 0

            # Calculate advantages using GAE ~ start from last time step
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.gae_lambda * gae * (
                    1 - terminated[t]
                )
                adv[t] = gae

            adv = T.tensor(adv).float().unsqueeze(1).to(self.actor.device)
            returns = adv + values
            # Normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)
        return adv, returns

    def learn(self, memory):
        # Always check if memory is ready before learning
        if not memory.is_ready():
            return
        (
            state_array,
            action_array,
            reward_array,
            next_state_array,
            terminated_array,
            old_prob_array,
        ) = memory.recall()

        # Convert memory to tensors
        device = self.critic.device
        state_array = T.tensor(state_array, dtype=T.float, device=device)
        action_array = T.tensor(action_array, dtype=T.float, device=device)
        reward_array = T.tensor(reward_array, dtype=T.float, device=device).unsqueeze(1)
        next_state_array = T.tensor(next_state_array, dtype=T.float, device=device)
        terminated_array = T.tensor(terminated_array, dtype=T.float, device=device)
        old_prob_array = T.tensor(old_prob_array, dtype=T.float, device=device)

        # Calculate advantages and returns
        adv, returns = self.calc_adv_and_returns(
            (state_array, next_state_array, reward_array, terminated_array)
        )

        # Learning loop over epochs
        for epoch in range(self.n_epochs):
            mini_batches = memory.generate_batches()
            # Loop over mini batches
            for mini_batch in mini_batches:
                states = state_array[mini_batch]
                old_probs = old_prob_array[mini_batch]
                actions = action_array[mini_batch]

                # Get new action distributions from states
                dist = self.actor(states)
                # Get new log probabilities of selected actions
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(
                    new_probs.sum(1, keepdim=True) - old_probs.sum(1, keepdim=True)
                )

                # Weighted probabilities
                weighted_probs = adv[mini_batch] * prob_ratio
                # Weighted cipped probablities
                weighted_clipped_probs = adv[mini_batch] * (
                    T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                )

                # In PPO we maintain entropy to make sure our actor network dosnt become too deterministic,
                # Entropy ~ Uncertainty in distribution
                # Entropy regularisation -> exploration
                entropy = dist.entropy().sum(1, keepdim=True)

                # Loss calculation
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)
                # Take away entropy value from loss to encourage a certain level of randomness in the action distribution
                actor_loss -= self.entropy_coefficient * entropy
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()

                # Gradient clipping
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                # Loss calculation
                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[mini_batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

        # Clear memory after learning
        memory.clear_memory()

    def save(self, filepath) -> None:
        T.save(self.actor.state_dict(), filepath / "ppo_actor")
        T.save(self.actor.optimizer.state_dict(), filepath / "ppo_actor_optimiser")

        T.save(self.critic.state_dict(), filepath / "ppo_critic")
        T.save(self.critic.optimizer.state_dict(), filepath / "ppo_critic_optimiser")

        print("... saving checkpoint ...")

    def load(self, filepath) -> None:
        self.actor.load_state_dict((T.load(filepath / "ppo_actor")))
        self.actor.optimizer.load_state_dict(T.load(filepath / "ppo_actor_optimiser"))

        self.critic.load_state_dict((T.load(filepath / "ppo_critic")))
        self.critic.optimizer.load_state_dict(T.load(filepath / "ppo_critic_optimiser"))

        print("... loading checkpoint ...")
