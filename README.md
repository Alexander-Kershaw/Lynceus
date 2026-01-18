***
# LYNCEUS
***
**Radar Tracking and Sensor Fusion System (Multi-Target Tracking)**

Lynceus is a radar-style tracking system that turns noisy, unreliable sensor detections into a stable tactical picture of moving targets.

The core problem is **estimation under uncertainty**:
- measurements are noisy
- detections can be missed (dropouts and radar occlusions)
- later: clutter (false positives), crossings, and multi-target identity tracking

---

## Current Status
- Repository scaffold + runnable module  
- 2D constant-velocity ground truth simulator (single target)  
- Cartesian sensor model: Gaussian measurement noise, radar dropout implementation


Next milestone adds a full **Kalman Filter** (predict/update) using the simulated sensor stream.

---

## Current Implementations

### 1) Ground Truth Simulator (CV Model)
State vector:
\[
x = [p_x,\ p_y,\ v_x,\ v_y]^T
\]

Discrete-time update:
\[
p_{x,k+1} = p_{x,k} + v_{x,k}\Delta t,\quad
p_{y,k+1} = p_{y,k} + v_{y,k}\Delta t
\]
\[
v_{x,k+1} = v_{x,k},\quad
v_{y,k+1} = v_{y,k}
\]

Outputs an array of truth states over time.

### 2) Sensor Model (Cartesian Radar-like)
Measurement vector:
\[
z = [p_x,\ p_y]^T
\]

Measurement model:
\[
z = Hx + v,\quad v \sim \mathcal{N}(0,\ R)
\]

Supports:
- `sigma` measurement noise
- `p_miss` probability of missed detection (`None` returned)

---

## Repository Layout

```text
lynceus/
pyproject.toml
environment.yml
README.md
.gitignore
src/
lynceus/
init.py
main.py
config.py
dynamics.py
sensors.py
```

---

## Setup

### Conda Environment 

```bash
conda env create -f environment.yml
conda activate lynceus
```

### Editable Install

```bash
python -m pip install -e .
```

---

## Run

```bash
python -m lynceus
```

---

## Current Roadmap

- Kalman Filter (CV + Cartesian) with missed-detection handling
- Multi-target simulation (crossings + maneuvers)
- Data association: nearest-neighbour + gating. Hungarian assignment
- Track management: tentative/confirmed tracks, deletion on consecutive misses
- Metrics: RMSE (position error), ID switches / track purity, continuity (fragmentation)
- Visual deliverables:, animated truth vs measurements vs tracks, “radar screen” tactical view with track IDs + velocity vectors

---