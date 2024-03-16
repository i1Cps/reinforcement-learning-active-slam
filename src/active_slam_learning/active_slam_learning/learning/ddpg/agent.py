import numpy as np
import torch.nn.functional as F
import torch as T
from active_slam_learning.learning.ddpg.noise import OUActionNoise
from active_slam_learning.learning.ddpg.replay_memory import ReplayBuffer
from active_slam_learning.learning.ddpg.networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        n_actions,
        logger,
        gamma=0.99,
        max_size=1000000,
        fc1_dims=400,
        fc2_dims=300,
        batch_size=64,
        episodes_until_learning=50,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        self.logger = logger
        self.episodes_until_learning = episodes_until_learning

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(
            input_dims, fc1_dims, fc2_dims, n_actions, alpha, name="actor"
        )

        self.critic = CriticNetwork(
            input_dims, fc1_dims, fc2_dims, n_actions, beta, name="critic"
        )

        self.target_actor = ActorNetwork(
            input_dims, fc1_dims, fc2_dims, n_actions, alpha, name="target_actor"
        )

        self.target_critic = CriticNetwork(
            input_dims, fc1_dims, fc2_dims, n_actions, beta, name="target_critic"
        )
        self.update_network_parameters(tau=1)

    # Use the actor network to generate an action given a state,  this network is a representation of our policy
    def choose_action(self, observation):
        # Required for batch normalization and drop out stuff
        self.actor.eval()
        # state = T.from_numpy(observation)
        state = T.tensor(observation[np.newaxis, :], dtype=T.float).to(
            self.actor.device
        )

        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()

        # AKA The chosen actions
        return mu_prime.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, new_state, terminal):
        self.memory.store_transition(state, action, reward, new_state, terminal)

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

    # Skip learning if episode number is less than X
    def learn(self, current_episode):
        if current_episode < self.episodes_until_learning:
            return

        states, actions, rewards, new_states, terminals = self.memory.sample_memory(
            self.batch_size
        )
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor.device)
        terminals = T.tensor(terminals).to(self.actor.device)

        # Below is based on OG DDPG continuous control paper
        # It uses typical Actor critic from policy gradient theorem but with DQN style stability using target network

        # critic_value_next is the value from the next state action pair
        target_actions = self.target_actor.forward(new_states)
        critic_value_next = self.target_critic.forward(new_states, target_actions)

        # critic_value_cur is the value for the current state action pair that we sampled
        critic_value_cur = self.critic.forward(states, actions)

        critic_value_next[terminals] = 0.0
        critic_value_next = critic_value_next.view(-1)

        target = rewards + self.gamma * critic_value_next
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        # critic_loss = F.mse_loss(target, critic_value_cur)
        # Order is irrelevant for this loss function
        critic_loss = F.mse_loss(critic_value_cur, target)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    # Instead of a hard copy, we use, tau, a weight/ratio of how much we want to copy over,
    # 0.5 50%, 0.9 90% of original critic etc
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

        # unrecommened version for batch normalization instead of layer
        # self.target_critic.load_state_dict(critic_state_dict, strict =False)
        # self.target_actor.laod_state_dict(actor_state_dict, strick = False)
        #

    def reset_noise(self):
        self.noise.reset()
