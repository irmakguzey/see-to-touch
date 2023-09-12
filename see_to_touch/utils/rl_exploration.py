import numpy as np
import math
import matplotlib.pyplot as plt

# Taken by https://ai.stackexchange.com/questions/39896/choosing-and-designing-decay-types-for-epsilon-greedy-exploration-in-reinforceme
def exponential_epsilon_decay(step_idx, epsilon_start=1, epsilon_end=0.01, epsilon_decay=100_000):
    """
    Calculates the value of epsilon for a given step index using exponential decay and the specified parameters.

    Parameters:
    step_idx (int): The index of the current step.
    epsilon_start (float): The starting value of epsilon.
    epsilon_end (float): The minimum value of epsilon.
    epsilon_decay (float): The rate at which epsilon decays.

    Returns:
    float: The value of epsilon for the given step index.
    """
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * step_idx / epsilon_decay)

def linear_epsilon_decay(step_idx, epsilon_start=1, epsilon_end=0.01, epsilon_decay=100_000):
    """
    Calculates the value of epsilon for a given step index using linear decay and the specified parameters.

    Parameters:
    step_idx (int): The index of the current step.
    epsilon_start (float): The starting value of epsilon.
    epsilon_end (float): The minimum value of epsilon.
    epsilon_decay (float): The total number of steps over which epsilon will decay from epsilon_start to epsilon_end.

    Returns:
    float: The value of epsilon for the given step index.
    """
    return epsilon_end + (epsilon_start - epsilon_end) * max((1 - step_idx / epsilon_decay), 0)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)