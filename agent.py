import numpy as np
from model import LinearModel


class Agent:
    def __init__(
        self,
        state_dim=1,
        sensory_noise_std=0.5,
        inference_lr=0.05,
        precision_prediction_lr=0.1,
        hyperprecision_prediction_lr=0.05,
        log_precision_bounds=(-2.0, 2.0),
        model=None,
    ):
        self.state_dim = state_dim
        self.internal_state = np.zeros(state_dim)
        self.prev_internal_state = np.zeros(state_dim)   # store past belief
        self.sensory_noise_std = sensory_noise_std
        self.inference_lr = inference_lr
        self.precision_prediction_lr = precision_prediction_lr
        self.hyperprecision_prediction_lr = hyperprecision_prediction_lr
        self.log_precision_bounds = log_precision_bounds
        self.log_precision = np.zeros(state_dim)
        self.prev_error = np.zeros(state_dim)

        # Plug in model — if none given, default to a linear one
        self.model = model if model is not None else LinearModel(state_dim)

    def sense(self, true_state):
        """Noisy sensory input."""
        return true_state + np.random.normal(0, self.sensory_noise_std, size=self.state_dim)

    def predict_sensation(self):
        """Predict the next sensory input using the model."""
        return self.model(self.internal_state)

    def update_belief(self, sensation):
        """Update internal state and let model learn dynamics."""
        predicted = self.predict_sensation()
        error = sensation - predicted
        precision = np.exp(self.log_precision)

        # Standard predictive coding update
        self.prev_internal_state = self.internal_state.copy()
        self.internal_state += self.inference_lr * precision * error

        # Model learns how the internal state evolves
        self.model.update(self.prev_internal_state, self.internal_state)

        return error

    def predict_log_precision(self, error):
        """Predict future precision (confidence)."""
        return self.log_precision + self.precision_prediction_lr * (0.5 - np.abs(error))

    def update_log_precision(self, log_precision_pred, error):
        """Hyper-self-correction using relative error change."""

        # Compute relative change in prediction error magnitude
        prev_abs = np.abs(self.prev_error)
        curr_abs = np.abs(error)

        # Avoid division by zero — normalize relative change
        rel_change = (prev_abs - curr_abs) / (prev_abs + 1e-6)

        # Update hyperprecision proportionally to improvement or worsening
        self.log_precision += self.hyperprecision_prediction_lr * rel_change

        # Blend predicted precision and learned update (keeps it stable)
        self.log_precision = 0.5 * self.log_precision + 0.5 * log_precision_pred

        # Optional: apply mild decay so it doesn’t drift upward indefinitely
        self.log_precision *= 0.995

        # Clamp to avoid runaway precision
        self.log_precision = np.clip(self.log_precision, *self.log_precision_bounds)

        # Store for next step
        self.prev_error = error.copy()

