from __future__ import annotations

from lynceus.config import SimConfig
from lynceus.dynamics import CVModel, simulate_truth_cv, simulate_truth_cv_multi
from lynceus.sensors import CartesianSensor, simulate_measurements, simulate_measurements_multi
from lynceus.filters import Q_cv, kf_predict, kf_update
from lynceus.metrics import rmse, rmse_measurements_vs_truth
from lynceus.plotting import plot_truth_meas_kf, plot_radar_screen, plot_truth_and_detections_multi
from lynceus.scenarios import crossing_scenario
from lynceus.tracking import MultiTargetTracker


from termcolor import colored

import numpy as np



# Entry point for the LYNCEUS simulation framework

def main() -> None:
    cfg = SimConfig()
    print(colored("LYNCEUS boot successful", "green"))
    print(colored(f"dt={cfg.dt}, steps={cfg.steps}, seed={cfg.seed}", "light_yellow"))

    # Multi-target truth 
    X0 = crossing_scenario()

    X = simulate_truth_cv_multi(X0=X0, dt=cfg.dt, steps=cfg.steps)

    print(colored("Truth (k=0 positions):", "cyan"))
    for i in range(X.shape[1]):
        px, py = X[0, i, 0], X[0, i, 1]
        print(f"tgt={i}  pos=({px:7.2f},{py:7.2f})")

    # Multi-target sensor
    rng = np.random.default_rng(cfg.seed)
    sensor = CartesianSensor(sigma=cfg.meas_sigma, p_miss=cfg.p_miss, rng=rng)
    Z = simulate_measurements_multi(X, sensor)

    # Multi-target tracking
    tracker = MultiTargetTracker(
        dt=cfg.dt,
        accel_sigma=cfg.accel_sigma,
        H=sensor.H(),
        R=sensor.R(),
        gate_radius=cfg.gate_radius,
        confirm_hits=cfg.confirm_hits,
        kill_misses=cfg.kill_misses,
    )

    # Run tracking over time
    track_log: list[list[tuple[int, float, float, bool]]] = []
    for k in range(len(Z)):
        tracks = tracker.step(k, Z[k])
        snap = []
        for t in tracks:
            snap.append((t.track_id, float(t.x[0]), float(t.x[1]), t.confirmed, t.misses)) # Including misses as well as confirmed to verify tracking
        track_log.append(snap)

    print(colored("Tracks at timesteps 0..9:", "magenta"))
    for k in range(10):
        items = ", ".join([f"id={tid} ({x:.1f},{y:.1f}) m={m}{'*' if conf else ''}" for tid, x, y, conf, m in track_log[k]])

        print(f"k={k:02d}  {items}")


    print(colored("Detections per timestep (first 10):", "yellow"))
    for k in range(min(10, len(Z))):
        print(f"k={k:02d}  detections={len(Z[k])}")


    return



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

    # Radar style plot
    plot_radar_screen(X, Z, est, trail=25)


if __name__ == "__main__":
    main()
