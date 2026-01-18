from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianSensor:
    """
    -------------------------------------------------------------------
    Simple Cartesian sensor measuring position

    Measurement model:
        z = H x + v
    where:
        x = [px, py, vx, vy]
        z = [px, py]
        H picks out position components
        v ~ N(0, R), R = sigma^2 I
    
    -------------------------------------------------------------------
    """
    sigma: float
    p_miss: float
    rng: np.random.Generator

    def H(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def R(self) -> np.ndarray:
        s2 = float(self.sigma) ** 2
        return np.array([[s2, 0.0], [0.0, s2]], dtype=float)

    def measure(self, x: np.ndarray) -> np.ndarray | None:
        
        # Returns a noisy measurement z = [x, y] or None if missed detection occurs
        
        if x.shape != (4,):
            raise ValueError(f"x must have shape (4,), got {x.shape}")

        # Missed detection
        if self.rng.random() < self.p_miss:
            return None

        H = self.H()
        z_true = H @ x
        noise = self.rng.normal(loc=0.0, scale=self.sigma, size=(2,))
        return z_true + noise


def simulate_measurements(X: np.ndarray, sensor: CartesianSensor) -> list[np.ndarray | None]:
    
    # Produces a singular measurement per timestep (or None if missed).
    
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"X must have shape (T,4), got {X.shape}")

    Z: list[np.ndarray | None] = []
    for k in range(X.shape[0]):
        Z.append(sensor.measure(X[k]))
    return Z
