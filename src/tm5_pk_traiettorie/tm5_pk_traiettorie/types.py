from dataclasses import dataclass
import numpy as np

@dataclass
class CartesianTrajectoryPoint:
    t: float
    X: np.ndarray      # shape (3,) ora, (6,) in futuro
    Xdot: np.ndarray   # shape (3,)
    Xddot: np.ndarray  # shape (3,)

@dataclass
class Joint:
    t: float
    q: np.ndarray      # shape (6,)
    qdot: np.ndarray   # shape (6,)
