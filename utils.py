import matplotlib.pyplot as plt
import numpy as np


def plot_simulation(
    x_true, state_estimates, prediction_errors, precisions, predicted_precisions=None
):
    """
    Stack plots vertically: for each dimension, show state, prediction error, precision.
    Optionally plot predicted precision alongside actual precision.
    """
    T, dim = x_true.shape
    plt.figure(figsize=(12, 3 * dim * 3))  # height scales with total number of rows

    row = 1
    for d in range(dim):
        # True vs estimated state
        plt.subplot(dim * 3, 1, row)
        plt.plot(x_true[:, d], label=f"True State {d}", color="black")
        plt.plot(state_estimates[:, d], label=f"Estimate {d}", color="blue")
        plt.ylabel("State")
        plt.legend()
        row += 1

        # Prediction error
        plt.subplot(dim * 3, 1, row)
        plt.plot(prediction_errors[:, d], label=f"Prediction Error {d}", color="red")
        plt.axhline(0, color="gray", linestyle="--", lw=0.8)
        plt.ylabel("Error")
        plt.legend()
        row += 1

        # Precision
        plt.subplot(dim * 3, 1, row)
        plt.plot(precisions[:, d], label=f"Actual Precision Φ {d}", color="orange")
        if predicted_precisions is not None:
            plt.plot(
                predicted_precisions[:, d],
                label=f"Predicted Precision Φ {d}",
                color="green",
                linestyle="--",
            )
        plt.ylabel("Precision")
        plt.legend()
        row += 1

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()
