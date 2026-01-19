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



def plot_truth_and_detections_multi(
    X: np.ndarray,
    Z: list[list[np.ndarray]],
    title: str = "LYNCEUS: target crossing scenario",
) -> None:
    """
    ------------------------------------------------------------------------
    For the plot of multi-target truth trajectories and unlabelled detections.

    note: Detections are flattened into one cloud or raw detection hits,
    This is because it should look messy near the crossing as targets become
    more indistinguishable. 

    X: (T, N, 4)
    Z: list length T, each element is list of (2,) detections

    ------------------------------------------------------------------------
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if X.ndim != 3 or X.shape[2] != 4:
        raise ValueError(f"X must have shape (T,N,4), got {X.shape}")

    T, N, _ = X.shape
    if N < 2:
        raise ValueError("At least 2 targets required to show a crossing event")
    
    # Compute closest approach between target 0 and 1 (by truth positions)
    p0 = X[:, 0, 0:2]
    p1 = X[:, 1, 0:2]
    d = np.linalg.norm(p0 - p1, axis=1)
    k_star = int(np.argmin(d))  # timestep where they're closest

    plt.figure()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    # Truth trajectories per target
    for i in range(N):
        truth = X[:, i, 0:2]
        plt.plot(truth[:, 0], truth[:, 1], label=f"truth tgt {i}")

        # Marking target positions at their closest approach
        plt.scatter([truth[k_star, 0]], [truth[k_star, 1]], marker="o")

    # Flatten detections into a raw detection hit cloud
    dets = []
    for k in range(T):
        for z in Z[k]:
            dets.append(z)
    if len(dets) > 0:
        D = np.vstack(dets)
        plt.scatter(D[:, 0], D[:, 1], marker="x", label="detections")

    # Crossing point indicator (detection hits cloud should be here)
    mid = 0.5 * (p0[k_star] + p1[k_star])
    plt.scatter([mid[0]], [mid[1]], marker="*", label=f"closest approach k={k_star}")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f"[crossing] closest approach at k={k_star}, distance={d[k_star]:.3f}, midpoint=({mid[0]:.2f},{mid[1]:.2f})")
