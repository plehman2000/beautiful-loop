import numpy as np

class BaseModel:
    """Abstract base class for internal predictive models."""
    def predict(self, state):
        raise NotImplementedError("Subclasses must implement predict()")


class LinearModel:
    """Learnable linear transition x_{t+1} = W x_t + b"""
    
    def __init__(self, state_dim, lr=0.01):
        self.state_dim = state_dim
        self.lr = lr
        self.W = np.eye(state_dim)       # linear weights
        self.b = np.zeros(state_dim)     # bias term to handle small states

    def __call__(self, state):
        """Predict next state."""
        return self.W @ state + self.b

    def update(self, prev_state, next_state):
        """Update weights using prediction error."""
        pred = self(prev_state)
        error = next_state - pred
        self.W += self.lr * np.outer(error, prev_state)
        self.b += self.lr * error          # bias update ensures learning even if prev_state ~ 0
        # calmp to avoid divergence if w gets too big
        self.W = np.clip(self.W, -5, 5)
        self.b = np.clip(self.b, -5, 5)


# import torch
# import torch.nn as nn


# class RNNModel(nn.Module):
#     """
#     A minimal RNN world model that learns temporal dynamics.
#     Input: current internal state
#     Output: predicted next state
#     """

#     def __init__(self, state_dim, hidden_dim=32, lr=1e-3):
#         super().__init__()
#         self.state_dim = state_dim
#         self.hidden_dim = hidden_dim
#         self.rnn = nn.RNN(input_size=state_dim, hidden_size=hidden_dim, batch_first=True)
#         self.output = nn.Linear(hidden_dim, state_dim)

#         self.hidden_state = torch.zeros(1, 1, hidden_dim)  # (num_layers, batch, hidden)
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         self.loss_fn = nn.MSELoss()

#     def forward(self, x):
#         """Forward pass: predict next state."""
#         x = x.unsqueeze(0).unsqueeze(0)  # (batch=1, seq=1, features)
#         out, self.hidden_state = self.rnn(x, self.hidden_state.detach())
#         return self.output(out.squeeze(0).squeeze(0))

#     def update(self, prev_state, next_state):
#         """Train step using prediction error."""
#         self.optimizer.zero_grad()

#         # Convert to tensors
#         prev_state = torch.tensor(prev_state, dtype=torch.float32)
#         next_state = torch.tensor(next_state, dtype=torch.float32)

#         prediction = self.forward(prev_state)
#         loss = self.loss_fn(prediction, next_state)
#         loss.backward()
#         self.optimizer.step()

#         return loss.item()

