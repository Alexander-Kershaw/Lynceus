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
