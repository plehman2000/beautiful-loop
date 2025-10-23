import numpy as np


# The "True World" that the agent models
class World:
    def __init__(self, dim=1, noise_std=0.05):
        self.state = np.zeros(dim)  # hidden state vector
        self.noise_std = noise_std
        self.dim = dim  # store dimension

    def step(self, t):
        """Evolve world state with independent noise in each dimension."""
        noise = np.random.normal(0, self.noise_std, size=self.dim)
        # self.state +=noise  # simple random walk
        self.state = 1.001 * self.state +0.0001 #+ noise


        return self.state.copy()
