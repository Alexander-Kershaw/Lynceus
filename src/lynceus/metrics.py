from __future__ import annotations

import numpy as np

"""
-------------------------------------------------------------------

Notes:

- rmse() computes 2D Euclidean distance for each timestep, which is averaged and squared rooted

- Measurement RMSE only can use timesteps where a measurement exists, cannot operate using a null value,

-------------------------------------------------------------------

"""


def rmse(a: np.ndarray, b: np.ndarray) -> float:
   
    # Root Mean Square Error between two arrays of equal shape

    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


def rmse_measurements_vs_truth(Z: list[np.ndarray | None], X: np.ndarray) -> float:
    
    # RMSE of measurement positions vs truth positions, using only timesteps where a measurement exists
    
    truth_pos = X[:, 0:2]
    meas_pos = []
    truth_used = []

    for k, z in enumerate(Z):
        if z is None:
            continue
        meas_pos.append(z)
        truth_used.append(truth_pos[k])

    if len(meas_pos) == 0:
        raise ValueError("no measurements available to score")

    meas_pos_arr = np.vstack(meas_pos)
    truth_used_arr = np.vstack(truth_used)
    return rmse(meas_pos_arr, truth_used_arr)
