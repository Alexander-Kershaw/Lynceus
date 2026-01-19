from __future__ import annotations

from lynceus.config import SimConfig
from lynceus.dynamics import CVModel, simulate_truth_cv
from lynceus.sensors import CartesianSensor, simulate_measurements
from lynceus.filters import Q_cv, kf_predict, kf_update
from lynceus.metrics import rmse, rmse_measurements_vs_truth
from lynceus.plotting import plot_truth_meas_kf

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


    # Kalman Filter
    model = CVModel(dt=cfg.dt)
    F = model.F()
    Q = Q_cv(dt=cfg.dt, accel_sigma=cfg.accel_sigma)
    H = sensor.H()
    R = sensor.R()

    # Uses first measurement if available otherwise falls back to truth position
    x_est = np.zeros(4, dtype=float)
    if Z[0] is not None:
        x_est[0:2] = Z[0]
    else:
        x_est[0:2] = X[0, 0:2]

    # High initial uncertainty
    P_est = np.diag([25.0, 25.0, 10.0, 10.0]).astype(float)

    est = np.zeros_like(X)
    est[0] = x_est

    print(colored("KF track (first 10):", "magenta"))
    for k in range(1, X.shape[0]):
        # Predict every step
        x_pred, P_pred = kf_predict(x_est, P_est, F, Q)

        # Update only if measurement exists
        if Z[k] is not None:
            x_est, P_est = kf_update(x_pred, P_pred, Z[k], H, R)
            tag = "upd"
        else:
            x_est, P_est = x_pred, P_pred
            tag = "prd"

        est[k] = x_est

        if k < 10:
            px, py, vx, vy = x_est
            print(f"k={k:02d} [{tag}]  pos=({px:7.2f},{py:7.2f})  vel=({vx:5.2f},{vy:5.2f})")

    
    # Metrics 
    truth_pos = X[:, 0:2]
    kf_pos = est[:, 0:2]

    kf_rmse = rmse(kf_pos, truth_pos)
    meas_rmse = rmse_measurements_vs_truth(Z, X)

    print(colored("Metrics:", "green"))
    print(f"RMSE position | KF vs truth: {kf_rmse:.3f}")
    print(f"RMSE position | meas vs truth: {meas_rmse:.3f}")

    # Plot trajectories (truth), measurements, Kalman filter
    plot_truth_meas_kf(X, Z, est)


if __name__ == "__main__":
    main()
