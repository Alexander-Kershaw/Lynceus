from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    dt: float = 1.0
    steps: int = 50
    seed: int = 7

    # Sensor parameters 
    meas_sigma: float = 1.0 # standard deviation of cartesian position measurements noise (controls radar resolution)
    p_miss: float = 0.10 # probability a detection is missed (simulates radar droppouts and occlusions)

