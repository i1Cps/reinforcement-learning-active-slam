import numpy as np


class OUNoise:
    def __init__(self, size, mean=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.size = size
        self.mean = mean
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mean

    def __call__(self):
        x = self.state
        dx = self.theta * (self.mean - x) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.random.normal(size=self.size)
        self.state = x + dx
        return self.state


class PinkNoise:
    def __init__(self, size, alpha=1):
        self.size = size
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.state = np.zeros(self.size)

    def __call__(self):
        white = np.random.normal(size=self.size)
        pink = np.fft.irfft(
            np.fft.rfft(white)
            * (np.arange(self.size // 2 + 1) + 1) ** (-self.alpha / 2.0)
        )
        self.state = pink[: self.size]
        return self.state


class DifferentialDriveOUNoise:
    def __init__(
        self,
        action_size=2,
        mean=0.0,
        theta=0.15,
        sigma=0.2,
        dt=1e-2,
        max_angular=2.2,
        min_angular=-2.2,
        max_linear=0.2,
        min_linear=-0.2,
    ):
        self.angular_noise = OUNoise(1, mean, theta, sigma, dt)
        self.linear_noise = OUNoise(1, mean, theta, sigma, dt)
        self.max_angular = max_angular
        self.min_angular = min_angular
        self.max_linear = max_linear
        self.min_linear = min_linear

    def reset(self):
        self.angular_noise.reset()
        self.linear_noise.reset()

    def __call__(self):
        angular_noise = self.angular_noise()
        linear_noise = self.linear_noise()
        angular_noise = np.clip(angular_noise, self.min_angular, self.max_angular)
        linear_noise = np.clip(linear_noise, self.min_linear, self.max_linear)
        return np.array([linear_noise[0], angular_noise[0]])


class DifferentialDrivePinkNoise:
    def __init__(
        self,
        action_size=2,
        alpha=1,
        steps=2000,
        max_angular=2.2,
        min_angular=-2.2,
        max_linear=0.2,
        min_linear=-0.2,
    ):
        self.steps = steps
        self.angular_noise = PinkNoise(steps, alpha)
        self.linear_noise = PinkNoise(steps, alpha)
        self.max_angular = max_angular
        self.min_angular = min_angular
        self.max_linear = max_linear
        self.min_linear = min_linear

    def reset(self):
        self.angular_noise.reset()
        self.linear_noise.reset()

    def __call__(self, step):
        angular_noise = self.angular_noise()[step]
        linear_noise = self.linear_noise()[step]
        angular_noise = np.clip(angular_noise, self.min_angular, self.max_angular)
        linear_noise = np.clip(linear_noise, self.min_linear, self.max_linear)
        return np.array([linear_noise, angular_noise])


noise = DifferentialDriveOUNoise()
action = noise()
print(action)
steps = 1000
pink_noise = DifferentialDrivePinkNoise(steps=steps)

actions_pink = np.zeros((steps, 2))

for i in range(steps):
    actions_pink[i] = pink_noise(i)
