import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from agent import Agent
    from world import World
    from simulate import run_simulation
    from utils import plot_simulation
    return plot_simulation, run_simulation


@app.cell
def _(plot_simulation, run_simulation):
    x_true, state_estimates, prediction_errors, precisions, predicted_precisions = (
        run_simulation(T=500, state_dim=1)
    )
    plot_simulation(
        x_true, state_estimates, prediction_errors, precisions, predicted_precisions
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
