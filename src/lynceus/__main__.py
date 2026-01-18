from __future__ import annotations

from lynceus.config import SimConfig
from lynceus.dynamics import simulate_truth_cv
from lynceus.sensors import CartesianSensor, simulate_measurements

from termcolor import colored

import numpy as np


# Entry point for the LYNCEUS simulation framework

def main() -> None:
    cfg = SimConfig()
    print(colored("LYNCEUS boot successful", "green"))
    print(colored(f"dt={cfg.dt}, steps={cfg.steps}, seed={cfg.seed}", "light_yellow"))

    # Truth 
    x0 = np.array([0.0, 0.0, 1.2, -0.4], dtype=float)

    X = simulate_truth_cv(x0=x0, dt=cfg.dt, steps=cfg.steps)

    print(colored("Truth (first 5 states):", "yellow"))
    for k in range(min(5, X.shape[0])):
        px, py, vx, vy = X[k]
        print(f"k={k:02d}  pos=({px:7.2f},{py:7.2f})  vel=({vx:5.2f},{vy:5.2f})")


    # Sensor
    rng = np.random.default_rng(cfg.seed)
    sensor = CartesianSensor(sigma=cfg.meas_sigma, p_miss=cfg.p_miss, rng=rng)
    Z = simulate_measurements(X, sensor)

    print(colored("Measurements (first 10):", "yellow"))
    for k in range(min(10, len(Z))):
        z = Z[k]
        if z is None:
            print(f"k={k:02d}  z=None (miss)")
        else:
            print(f"k={k:02d}  z=({z[0]:7.2f},{z[1]:7.2f})")

if __name__ == "__main__":
    main()
