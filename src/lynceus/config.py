from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    dt: float = 1.0
    steps: int = 50
    seed: int = 7
