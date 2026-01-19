from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    dt: float = 1.0
    steps: int = 50
    seed: int = 7

    # Sensor parameters 
    meas_sigma: float = 2.0 # standard deviation of cartesian position measurements noise (controls radar resolution)
    p_miss: float = 0.20 # probability a detection is missed (simulates radar droppouts and occlusions)

    # Motion / process noise
    accel_sigma: float = 0.5 # constant to indicate to Kalman filter module how much unmodelled acceleration to expect

    # Tracking / association
    gate_radius: float = 4.0     # max distance to accept detection-to-track
    confirm_hits: int = 3        # hits required to confirm a track
    kill_misses: int = 5         # consecutive misses before deletion


