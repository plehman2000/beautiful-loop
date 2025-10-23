import numpy as np


# The "True World" that the agent models

class World:
    """Simple linear world with enough signal for learning."""
    
    def __init__(self, dim=1, noise_std=0.0):
        self.state = np.zeros(dim)  # hidden state vector
        self.dim = dim
        self.noise_std = noise_std

    def step(self, t):
        """Evolve world state linearly with optional small noise."""
        # Add a small linear trend so model can learn
        self.state = 1.02 * self.state + 0.05  # noticeable change per step
        self.state += np.random.normal(0, self.noise_std, size=self.dim)  # optional noise
        return self.state.copy()
