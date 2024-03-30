# This file contains all my noises, currently we just have OUNoise, I plan to add pink noise and brown noise maybe
import numpy as np


class OUActionNoise:
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        """
        Ornstein-Uhlenbeck process for generating noise in continuous control tasks.

        Parameters:
        - mu (array): Mean of the noise.
        - sigma (float): Standard deviation of the noise.
        - theta (float): Rate of mean reversion.
        - dt (float): Time step.
        - x0 (array, optional): Initial value of the noise process. Defaults to None.

        Initializes the parameters of the Ornstein-Uhlenbeck process.
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Generate noise sample using Ornstein-Uhlenbeck process.

        Returns:
        - x (array): Noise sample.

        Generates a noise sample by applying the Ornstein-Uhlenbeck process equation
        to the previous noise value.
        """
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        """
        Reset the noise process.

        Resets the noise process to its initial value (x0) or to zeros if x0 is not provided.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class OUNoise(object):
    def __init__(
        self,
        action_space,
        mu=0.0,
        theta=0.15,
        max_sigma=0.99,
        min_sigma=0.01,
        decay_period=600000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self, t=0):
        ou_state = self.evolve_state()
        decaying = float(float(t) / self.decay_period)
        self.sigma = max(
            self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying),
            self.min_sigma,
        )
        return ou_state
