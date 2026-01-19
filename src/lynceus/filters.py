from __future__ import annotations

import numpy as np


def Q_cv(dt: float, accel_sigma: float) -> np.ndarray:
    """
    -------------------------------------------------------------------

    Process noise covariance Q for a 2D constant-velocity model driven by
    white-noise acceleration.

    -------------------------------------------------------------------

    State: [px, py, vx, vy]
    accel_sigma: std-dev of acceleration noise (same in x and y)

    -------------------------------------------------------------------
    """
    q = float(accel_sigma) ** 2

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    # 1D block for [p, v] under white-noise acceleration:
    # [[dt^4/4, dt^3/2],
    #  [dt^3/2, dt^2  ]] * q
    Q1 = np.array([[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]], dtype=float) * q

    Q = np.zeros((4, 4), dtype=float)
    # x-dimension (px, vx)
    Q[0, 0] = Q1[0, 0]
    Q[0, 2] = Q1[0, 1]
    Q[2, 0] = Q1[1, 0]
    Q[2, 2] = Q1[1, 1]
    # y-dimension (py, vy)
    Q[1, 1] = Q1[0, 0]
    Q[1, 3] = Q1[0, 1]
    Q[3, 1] = Q1[1, 0]
    Q[3, 3] = Q1[1, 1]
    return Q


def kf_predict(
    x: np.ndarray,
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    -------------------------------------------------------------------
    
    Prediction:

      x_pred = F x
      P_pred = F P F^T + Q

    -------------------------------------------------------------------
    """
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def kf_update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    -------------------------------------------------------------------

    Update:
      y = z - H x_pred
      S = H P_pred H^T + R
      K = P_pred H^T S^{-1}
      x_upd = x_pred + K y
      P_upd = (I - K H) P_pred

    -------------------------------------------------------------------

    """
    y = z - (H @ x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ y
    I = np.eye(P_pred.shape[0], dtype=float)
    P_upd = (I - K @ H) @ P_pred
    return x_upd, P_upd



"""
Some notes:

- x is the best guess of the state [x, y, vx, vy]

- P is how uncertain this estimate is

- the predict method extraoplates this estimate forward by some temporal
  steps via physics (F) and adds uncertainty on this approximation (Q)

- update serves to blend the prediction with the measurement using the 
  Kalman gain K

- Observations missed by radar dropout/occulsion skips the update method,
  while keeping the prediction

"""