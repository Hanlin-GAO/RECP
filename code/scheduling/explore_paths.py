import numpy as np


def _sample_polyline(vertices, n_points):
    """Sample a closed or open polyline at equal arc-length intervals."""
    verts = np.asarray(vertices, dtype=float)
    segs = verts[1:] - verts[:-1]
    seg_lens = np.linalg.norm(segs, axis=1)
    total = float(np.sum(seg_lens))
    if total <= 1e-12:
        return [(float(verts[0, 0]), float(verts[0, 1]))] * n_points

    # Target distance along the polyline
    s_targets = np.linspace(0.0, total, n_points, endpoint=False)

    pts = []
    acc = 0.0
    j = 0
    for s in s_targets:
        while j < len(seg_lens) - 1 and acc + seg_lens[j] < s:
            acc += seg_lens[j]
            j += 1
        if seg_lens[j] <= 1e-12:
            p = verts[j].copy()
        else:
            r = (s - acc) / seg_lens[j]
            p = verts[j] + r * (verts[j + 1] - verts[j])
        pts.append((float(p[0]), float(p[1])))
    return pts


def build_circle_route(center, radius=220.0, n_points=160):
    cx, cy = float(center[0]), float(center[1])
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def build_triangle_route(center, radius=260.0, n_points=180, rotate_deg=0.0):
    """Construct an equilateral triangle from its circumradius and sample its edges."""
    cx, cy = float(center[0]), float(center[1])
    rot = np.deg2rad(rotate_deg)
    angles = np.array([90.0, 210.0, 330.0]) * np.pi / 180.0 + rot
    verts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    # Close the route
    verts = np.vstack([verts, verts[0:1]])
    return _sample_polyline(verts, n_points)


def build_square_route(center, half_side=220.0, n_points=200, rotate_deg=0.0):
    """Construct a square centered at the given position with side length 2 * half_side."""
    cx, cy = float(center[0]), float(center[1])
    # Construct axis-aligned vertices
    verts = np.array([
        [-half_side, -half_side],
        [ half_side, -half_side],
        [ half_side,  half_side],
        [-half_side,  half_side],
        [-half_side, -half_side],
    ], dtype=float)

    # Rotate the vertices
    rot = np.deg2rad(rotate_deg)
    R = np.array([[np.cos(rot), -np.sin(rot)],
                  [np.sin(rot),  np.cos(rot)]], dtype=float)
    verts = (verts @ R.T)
    verts[:, 0] += cx
    verts[:, 1] += cy
    return _sample_polyline(verts, n_points)


def get_waypoint(waypoints, idx):
    """Return a waypoint on a loop using a wrapped index."""
    if len(waypoints) == 0:
        return (0.0, 0.0)
    j = int(idx) % len(waypoints)
    return waypoints[j]
