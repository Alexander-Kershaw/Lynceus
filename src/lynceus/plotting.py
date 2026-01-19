from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_truth_meas_kf(
    X: np.ndarray,
    Z: list[np.ndarray | None],
    est: np.ndarray,
    title: str = "LYNCEUS: true trajectory vs measurement vs Kalman Filter",
) -> None:
    truth = X[:, 0:2]
    kf = est[:, 0:2]

    # Skip missing points
    meas_pts = []
    for z in Z:
        if z is not None:
            meas_pts.append(z)
    meas = np.vstack(meas_pts) if len(meas_pts) else np.empty((0, 2))

    plt.figure()
    plt.plot(truth[:, 0], truth[:, 1], label="truth")
    if meas.shape[0] > 0:
        plt.scatter(meas[:, 0], meas[:, 1], label="measurements", marker="x")
    plt.plot(kf[:, 0], kf[:, 1], label="Kalman filter track")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_radar_screen(
    X: np.ndarray,
    Z: list[np.ndarray | None],
    est: np.ndarray,
    trail: int = 25,
    title: str = "LYNCEUS radar screen (single track)",
) -> None:
    truth = X[:, 0:2]
    kf = est[:, 0:2]

    # Measurements (missed measurements get skipped)
    meas_pts = []
    for z in Z:
        if z is not None:
            meas_pts.append(z)
    meas = np.vstack(meas_pts) if len(meas_pts) else np.empty((0, 2))

    # Trail window
    n = truth.shape[0]
    start = max(0, n - trail)

    plt.figure()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    # Trails
    plt.plot(truth[start:, 0], truth[start:, 1], label="truth trail")
    plt.plot(kf[start:, 0], kf[start:, 1], label="KF trail")

    # Measurement hits
    if meas.shape[0] > 0:
        plt.scatter(meas[:, 0], meas[:, 1], label="hits", marker="x")

    # Current KF (Kalman filter) state
    px, py, vx, vy = est[-1]
    plt.scatter([px], [py], label="track now", marker="o")

    # Velocity vector arrow 
    speed = float(np.hypot(vx, vy))
    scale = 1.0 if speed == 0.0 else 3.0 / speed 
    plt.arrow(px, py, vx * scale, vy * scale, length_includes_head=True)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()
