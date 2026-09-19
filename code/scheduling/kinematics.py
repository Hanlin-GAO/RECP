import numpy as np


def move_toward(p, target, step):
    p = np.asarray(p, dtype=float)
    target = np.asarray(target, dtype=float)
    d = target - p
    dist = float(np.linalg.norm(d))
    if dist < 1e-12:
        return p.copy(), 0.0
    alpha = min(step / dist, 1.0)
    newp = p + alpha * d
    moved = float(np.linalg.norm(newp - p))
    return newp, moved


def dist(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))
