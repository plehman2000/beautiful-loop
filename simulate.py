from world import World
from agent import Agent
import numpy as np


def run_simulation(
    T=300, state_dim=3, world_noise_std=0.01, sensory_noise_std=0.1, alpha=0.05
):
    world = World(dim=state_dim, noise_std=world_noise_std)
    agent = Agent(
        state_dim=state_dim, sensory_noise_std=sensory_noise_std, inference_lr=alpha
    )

    x_true = np.zeros((T, state_dim))
    state_estimates = np.zeros((T, state_dim))
    prediction_errors = np.zeros((T, state_dim))
    precisions = np.zeros((T, state_dim))  # actual precision
    predicted_precisions = np.zeros((T, state_dim))  # predicted precision

    for t in range(T):
        true_state = world.step(t)
        x_true[t] = true_state

        observation = agent.sense(true_state)

        pred_error = agent.update_belief(observation)
        log_precision_pred = agent.predict_log_precision(pred_error)
        agent.update_log_precision(log_precision_pred, pred_error)

        state_estimates[t] = agent.internal_state
        prediction_errors[t] = pred_error
        precisions[t] = np.exp(agent.log_precision)  # actual precision
        predicted_precisions[t] = np.exp(
            log_precision_pred
        )  # store predicted precision

    return x_true, state_estimates, prediction_errors, precisions, predicted_precisions
