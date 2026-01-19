from __future__ import annotations

import numpy as np


def crossing_scenario() -> np.ndarray:
    """
    ------------------------------------------------------------------
    This script defines a scenario where two-target crossing in 2D with 
    constant velocity (heading towards each other on the same line
    trajectory, and overlap in the middle)

    This is to test the worst case scenario for multi target distinction
    for radar systems

    Returns X0: (N,4) initial states [px, py, vx, vy]

    ------------------------------------------------------------------
    """

    # Diagonal crossing scenario 
    X0 = np.array(
        [
            [-25.0, -12.0, 1.2, 0.6],   # target 0 moves right
            [25.0, -12.0, -1.2, 0.6],   # target 1 moves left (crossing paths)
        ],
        dtype=float,
    )
    return X0
