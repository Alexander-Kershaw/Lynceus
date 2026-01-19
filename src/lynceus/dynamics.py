from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CVModel:
    """
    -------------------------------------------------------------------
    Constant Velocity (CV) model in 2 dimensions.

    State: x = [px, py, vx, vy]^T. (state transition matrix)

    This matrix is the discrete time equivalent of the continuous time
    (i.e , position advances with velocity * dt for continuous time).

    -------------------------------------------------------------------
    Discrete-time update (no process noise yet):
        px_{k+1} = px_k + vx_k * dt
        py_{k+1} = py_k + vy_k * dt
        vx_{k+1} = vx_k
        vy_{k+1} = vy_k
    -------------------------------------------------------------------

    Note: Initial implementation does not include any randomness or process noise.
    This will be a deterministic baseline model initially

    -------------------------------------------------------------------
    """
    dt: float

    def F(self) -> np.ndarray:
        # State transition matrix method for the CV model
        dt = self.dt
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def step(self, x: np.ndarray) -> np.ndarray:
        # Advance one timestep
        return self.F() @ x


def simulate_truth_cv(
    x0: np.ndarray,
    dt: float,
    steps: int,
) -> np.ndarray:
    """
    -------------------------------------------------------------------
    Simulates a single target under the CV model

    Returns
    -------------------------------------------------------------------
    X : (steps+1, 4) array
        X[k] is the state at timestep k, starting at k=0 with x0.
    -------------------------------------------------------------------
    """
    if x0.shape != (4,):
        raise ValueError(f"x0 must have shape (4,), got {x0.shape}")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    model = CVModel(dt=dt)
    X = np.zeros((steps + 1, 4), dtype=float)
    X[0] = x0

    for k in range(steps):
        X[k + 1] = model.step(X[k])

    return X


def simulate_truth_cv_multi(
    X0: np.ndarray,
    dt: float,
    steps: int,
) -> np.ndarray:
    """
    -------------------------------------------------------------------

    Simulate N targets under the CV model

    Process:

    - Reuse the same transition matrix F

    - Apply the matrix F to each target independently for each timestep

    - Output: time x target x state

    Parameters
    -------------------------------------------------------------------
    X0 : (N,4) array
        Initial states for all targets.
    dt : float
    steps : int

    Returns
    -------------------------------------------------------------------
    X : (steps+1, N, 4) array
        X[k, i] is state of target i at timestep k.

    -------------------------------------------------------------------    
    """
    if X0.ndim != 2 or X0.shape[1] != 4:
        raise ValueError(f"X0 must have shape (N,4), got {X0.shape}")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    model = CVModel(dt=dt)
    F = model.F()

    n = X0.shape[0]
    X = np.zeros((steps + 1, n, 4), dtype=float)
    X[0] = X0

    for k in range(steps):
        # applying each state to each target
        X[k + 1] = (F @ X[k].T).T

    return X
