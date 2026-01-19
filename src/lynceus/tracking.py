from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from lynceus.dynamics import CVModel
from lynceus.filters import Q_cv, kf_predict, kf_update

from scipy.optimize import linear_sum_assignment

"""
------------------------------------------------------------------

Using Kalman Filter per track embedded in a tracker method

Features:

- Uses last predicted position (not last measurement), so should
be more resilient to crosses and misses

- Track confirmation protocol provents every random detection hit
being tracked, should help with building resilience to cluttered signals


------------------------------------------------------------------
"""


@dataclass
class Track:
    track_id: int
    x: np.ndarray       
    P: np.ndarray    
    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False

    # Track Initiation
    last_z: np.ndarray | None = None
    last_k: int | None = None


# Defining euclidean distance
def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(d[0] * d[0] + d[1] * d[1]))


# Initiation gate (to avoid duplication spawning of targets / target explosion)
def within_radius(p: np.ndarray, q: np.ndarray, r: float) -> bool:
    return _euclid(p, q) <= r


def hungarian_assign(
    track_pos: list[np.ndarray],
    detections: list[np.ndarray],
    gate_radius: float,
) -> tuple[dict[int, int], set[int], set[int]]:
    """
    --------------------------------------------------------------------------------------------------

    Hungarian assignment with gating

    Hungarian assignment serves to find the global minimum cost matching. Gating is
    enforced by making out-of-gate pairs impossible by the assignment of a large cost.
    After assignment, true distance is still verified amd invalid pairs are disgarded.

    This should resoult in a clean one-to-one mapping and avoid duplications leading to
    tracker explosion.

    --------------------------------------------------------------------------------------------------
    """
    nT = len(track_pos)
    nD = len(detections)

    assignments: dict[int, int] = {}
    unassigned_tracks = set(range(nT))
    unassigned_dets = set(range(nD))

    if nT == 0 or nD == 0:
        return assignments, unassigned_tracks, unassigned_dets

    # Cost matrix
    C = np.zeros((nT, nD), dtype=float)
    for ti in range(nT):
        for di in range(nD):
            C[ti, di] = _euclid(track_pos[ti], detections[di])

    big = gate_radius * 1e6
    C_gated = C.copy()
    C_gated[C_gated > gate_radius] = big

    row_ind, col_ind = linear_sum_assignment(C_gated)

    for ti, di in zip(row_ind.tolist(), col_ind.tolist()):
        if C[ti, di] <= gate_radius:
            assignments[ti] = di

    used_tracks = set(assignments.keys())
    used_dets = set(assignments.values())
    unassigned_tracks = set(range(nT)) - used_tracks
    unassigned_dets = set(range(nD)) - used_dets
    return assignments, unassigned_tracks, unassigned_dets



class MultiTargetTracker:
    def __init__(
        self,
        dt: float,
        accel_sigma: float,
        H: np.ndarray,
        R: np.ndarray,
        gate_radius: float,
        confirm_hits: int,
        kill_misses: int,
    ) -> None:
        self.model = CVModel(dt=dt)
        self.F = self.model.F()
        self.Q = Q_cv(dt=dt, accel_sigma=accel_sigma)
        self.H = H
        self.R = R

        self.gate_radius = gate_radius
        self.init_radius = gate_radius
        self.confirm_hits = confirm_hits
        self.kill_misses = kill_misses

        self._next_id = 1
        self.tracks: list[Track] = []

    def _init_track(self, z: np.ndarray, k: int) -> Track:
        x = np.array([z[0], z[1], 0.0, 0.0], dtype=float)
        P = np.diag([25.0, 25.0, 100.0, 100.0]).astype(float)
        t = Track(
            track_id=self._next_id,
            x=x,
            P=P,
            age=0,
            hits=1,
            misses=0,
            confirmed=False,
            last_z=z.copy(),
            last_k=k,
        )
        self._next_id += 1
        return t


    def step(self, k: int, detections: list[np.ndarray]) -> list[Track]:
        """
        ------------------------------------------------------------
        
        Advance tracker by one timestep given a set of detections
        Returns the current track list after update

        -------------------------------------------------------------
        """
        # Predict all tracks
        pred_x: list[np.ndarray] = []
        pred_P: list[np.ndarray] = []
        pred_pos: list[np.ndarray] = []

        for trk in self.tracks:
            x_pred, P_pred = kf_predict(trk.x, trk.P, self.F, self.Q)
            pred_x.append(x_pred)
            pred_P.append(P_pred)
            pred_pos.append(x_pred[0:2])

        # Associate detections to predicted track positions
        assignments, unassigned_tracks, unassigned_dets = hungarian_assign(
            track_pos=pred_pos,
            detections=detections,
            gate_radius=self.gate_radius,
        )

        # Update assigned tracks
        for ti, di in assignments.items():
            z = detections[di]
            x_upd, P_upd = kf_update(pred_x[ti], pred_P[ti], z, self.H, self.R)

            trk = self.tracks[ti]

            # If track is not confirmed and use previous measurement, estimate velocity
            if (not trk.confirmed) and (trk.last_z is not None) and (trk.last_k is not None):
                dt_steps = k - trk.last_k
                if dt_steps > 0:
                    dt_eff = dt_steps * (self.F[0, 2]) # dt_steps * Velocity
                    v_est = (z - trk.last_z) / dt_eff
                    # Injecting velocity guess into predicted state before KF update
                    pred_x[ti] = pred_x[ti].copy()
                    pred_x[ti][2:4] = v_est

            trk.x = x_upd
            trk.P = P_upd
            trk.age += 1
            trk.hits += 1
            trk.misses = 0
            trk.last_z = z.copy()
            trk.last_k = k
            
            if (not trk.confirmed) and trk.hits >= self.confirm_hits:
                trk.confirmed = True

        # Handle unassigned tracks
        for ti in sorted(unassigned_tracks):
            trk = self.tracks[ti]
            trk.x = pred_x[ti]
            trk.P = pred_P[ti]
            trk.age += 1
            trk.misses += 1

        # Create new tracks from unassigned detections
        for di in sorted(unassigned_dets):
            z = detections[di]

            # If detection is close to any predicted track position
            # do not spawn a new track as its likely a duplicate track causing tracking explosion
            too_close = False
            for p in pred_pos:
                if within_radius(p, z, self.init_radius):
                    too_close = True
                    break

            if not too_close:
                self.tracks.append(self._init_track(z, k))

        # Delete tracks with too many misses
        self.tracks = [t for t in self.tracks if t.misses < self.kill_misses]

        return self.tracks
