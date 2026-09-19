import json
import os

import numpy as np

from drone_profiles import load_drone_profiles
from scheduling_paths import SCENE_DATA

# Soccer field (meters): x in [0,105], y in [0,68]
FIELD_X_MIN = 0.0
FIELD_Y_MIN = 0.0
FIELD_X_MAX = 105.0
FIELD_Y_MAX = 68.0


def _make_circle_path(center=(52.5, 34.0), r=18.0, n=18):
    cx, cy = float(center[0]), float(center[1])
    ang = np.linspace(0, 2 * np.pi, int(n), endpoint=False)
    pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
    return pts.tolist()


def _make_triangle_path(p1=(30.0, 14.0), p2=(75.0, 14.0), p3=(52.5, 54.0), n_edge=8):
    pts = []

    def interp(a, b):
        a = np.array(a, float)
        b = np.array(b, float)
        for i in range(int(n_edge)):
            alpha = i / float(n_edge)
            pts.append((a * (1 - alpha) + b * alpha).tolist())

    interp(p1, p2)
    interp(p2, p3)
    interp(p3, p1)
    return pts


def _make_square_path(x0=34.0, y0=18.0, w=30.0, n_edge=7):
    p1 = (x0, y0)
    p2 = (x0 + w, y0)
    p3 = (x0 + w, y0 + w)
    p4 = (x0, y0 + w)

    pts = []

    def interp(a, b):
        a = np.array(a, float)
        b = np.array(b, float)
        for i in range(int(n_edge)):
            alpha = i / float(n_edge)
            pts.append((a * (1 - alpha) + b * alpha).tolist())

    interp(p1, p2)
    interp(p2, p3)
    interp(p3, p4)
    interp(p4, p1)
    return pts


def _polygon_area(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _resample_closed_path(points, n_samples):
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=float)
    if len(pts) == 1:
        return np.repeat(pts, int(n_samples), axis=0)
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]

    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(seg))
    if total < 1e-9:
        return np.repeat(closed[:1], int(n_samples), axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, int(n_samples), endpoint=False)
    out = []
    idx = 0
    for dist in targets:
        while idx + 1 < len(cum) - 1 and cum[idx + 1] < dist:
            idx += 1
        span = seg[idx]
        alpha = 0.0 if span < 1e-9 else (dist - cum[idx]) / span
        out.append((1.0 - alpha) * closed[idx] + alpha * closed[idx + 1])
    return np.asarray(out, dtype=float)


def _convex_hull(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 3:
        return pts.copy()

    pts = np.unique(np.round(pts, decimals=8), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0.0:
            lower.pop()
        lower.append(pt)

    upper = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0.0:
            upper.pop()
        upper.append(pt)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    return hull if len(hull) >= 3 else pts.copy()


def _ray_polygon_distance(polygon, center, angle):
    poly = np.asarray(polygon, dtype=float)
    center = np.asarray(center, dtype=float)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
    best = None

    for idx in range(len(poly)):
        a = poly[idx]
        b = poly[(idx + 1) % len(poly)]
        edge = b - a
        mat = np.array([[direction[0], -edge[0]], [direction[1], -edge[1]]], dtype=float)
        det = float(np.linalg.det(mat))
        if abs(det) < 1e-9:
            continue
        t, u = np.linalg.solve(mat, a - center)
        if t >= 0.0 and 0.0 <= u <= 1.0:
            if best is None or t < best:
                best = float(t)
    return best


def _build_radial_clearance_loop(center, obstacle_boundary, terrain_boundary, clearance_fraction, *, n_samples=144,
                                 min_clearance=0.20):
    center = np.asarray(center, dtype=float)
    pts = []
    for angle in np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=False):
        r_in = _ray_polygon_distance(obstacle_boundary, center, angle)
        r_out = _ray_polygon_distance(terrain_boundary, center, angle)
        if r_in is None or r_out is None:
            continue
        gap = max(float(r_out) - float(r_in), float(min_clearance))
        radius = float(r_in) + float(clearance_fraction) * gap
        pts.append(center + radius * np.array([np.cos(angle), np.sin(angle)], dtype=float))
    return np.asarray(pts, dtype=float)


def _point_in_polygon(point, polygon):
    x, y = float(point[0]), float(point[1])
    poly = np.asarray(polygon, dtype=float)
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y))
        if intersects:
            x_cross = (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _point_in_any_polygon(point, polygons):
    return any(_point_in_polygon(point, poly) for poly in polygons)


def _normalize(vec):
    vec = np.asarray(vec, dtype=float)
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return vec / nrm


def _push_point_to_passable(anchor, center, terrain_boundary, obstacle_regions, *, step_m=0.05, max_steps=240,
                            clearance_m=0.08):
    anchor = np.asarray(anchor, dtype=float)
    center = np.asarray(center, dtype=float)
    if _point_in_polygon(anchor, terrain_boundary) and not _point_in_any_polygon(anchor, obstacle_regions):
        return anchor.copy()

    radial = _normalize(anchor - center)
    tangent = np.array([-radial[1], radial[0]], dtype=float)
    directions = [
        radial,
        -radial,
        tangent,
        -tangent,
        _normalize(radial + tangent),
        _normalize(radial - tangent),
        _normalize(-radial + tangent),
        _normalize(-radial - tangent),
    ]

    for direction in directions:
        for step in range(1, int(max_steps) + 1):
            cand = anchor + direction * (float(step) * float(step_m))
            if _point_in_polygon(cand, terrain_boundary) and not _point_in_any_polygon(cand, obstacle_regions):
                cleared = cand + direction * float(clearance_m)
                if _point_in_polygon(cleared, terrain_boundary) and not _point_in_any_polygon(cleared, obstacle_regions):
                    return cleared
                return cand

    return anchor.copy()


def _enforce_passable_loop(loop, center, terrain_boundary, obstacle_regions, *, passes=3):
    loop = np.asarray(loop, dtype=float)
    if len(loop) == 0:
        return loop
    out = loop.copy()
    for _ in range(int(max(passes, 1))):
        out = np.asarray([
            _push_point_to_passable(pt, center, terrain_boundary, obstacle_regions)
            for pt in out
        ], dtype=float)
    return out


def _point_is_passable(point, terrain_boundary, obstacle_regions):
    return _point_in_polygon(point, terrain_boundary) and not _point_in_any_polygon(point, obstacle_regions)


def _segment_is_passable(start, goal, terrain_boundary, obstacle_regions, *, n_samples=25):
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    for alpha in np.linspace(0.0, 1.0, int(max(n_samples, 2))):
        pt = start * (1.0 - float(alpha)) + goal * float(alpha)
        if not _point_is_passable(pt, terrain_boundary, obstacle_regions):
            return False
    return True


def _densify_passable_loop(loop, center, terrain_boundary, obstacle_regions, *, max_passes=8, max_points=4096):
    out = np.asarray(loop, dtype=float)
    if len(out) < 2:
        return out

    for _ in range(int(max(max_passes, 1))):
        changed = False
        expanded = []
        n_pts = len(out)
        for idx in range(n_pts):
            start = out[idx]
            goal = out[(idx + 1) % n_pts]
            expanded.append(start.copy())
            if _segment_is_passable(start, goal, terrain_boundary, obstacle_regions):
                continue

            anchors = [
                start * (2.0 / 3.0) + goal * (1.0 / 3.0),
                0.5 * (start + goal),
                start * (1.0 / 3.0) + goal * (2.0 / 3.0),
            ]
            inserted = 0
            for anchor in anchors:
                candidate = _push_point_to_passable(anchor, center, terrain_boundary, obstacle_regions)
                if not _point_is_passable(candidate, terrain_boundary, obstacle_regions):
                    continue
                if np.linalg.norm(candidate - start) <= 1e-6 or np.linalg.norm(candidate - goal) <= 1e-6:
                    continue
                if inserted > 0 and np.linalg.norm(candidate - expanded[-1]) <= 1e-6:
                    continue
                expanded.append(candidate)
                inserted += 1

            changed = changed or (inserted > 0)

        out = np.asarray(expanded, dtype=float)
        out = _enforce_passable_loop(out, center, terrain_boundary, obstacle_regions, passes=1)
        if (not changed) or len(out) >= int(max_points):
            break

    return out


def _safe_task_point(anchor, center, terrain_boundary, obstacle_regions):
    point = _push_point_to_passable(anchor, center, terrain_boundary, obstacle_regions)
    return point if _point_is_passable(point, terrain_boundary, obstacle_regions) else np.asarray(anchor, dtype=float)


def _build_terrain_inset_loop(center, terrain_boundary, obstacle_regions, inset_m, *, n_samples=180, passes=2):
    center = np.asarray(center, dtype=float)
    boundary = _resample_closed_path(terrain_boundary, n_samples)
    if len(boundary) == 0:
        return np.zeros((0, 2), dtype=float)

    loop = []
    for pt in boundary:
        direction = _normalize(center - pt)
        candidate = np.asarray(pt, dtype=float) + direction * float(inset_m)
        candidate = _push_point_to_passable(candidate, center, terrain_boundary, obstacle_regions)
        loop.append(candidate)

    loop = np.asarray(loop, dtype=float)
    return _enforce_passable_loop(loop, center, terrain_boundary, obstacle_regions, passes=passes)


def _project_to_passable(anchor, direction, desired_offset, terrain_boundary, obstacle_boundary, min_offset=0.15):
    direction = _normalize(direction)
    desired_offset = float(max(desired_offset, min_offset))
    offsets = np.linspace(desired_offset, min_offset, max(int(np.ceil((desired_offset - min_offset) / 0.05)) + 1, 2))

    anchor = np.asarray(anchor, dtype=float)
    if _point_in_polygon(anchor, terrain_boundary) and not _point_in_polygon(anchor, obstacle_boundary):
        best_fallback = anchor.copy()
    else:
        best_fallback = anchor.copy()

    for offset in offsets:
        cand = anchor + direction * float(offset)
        if _point_in_polygon(cand, terrain_boundary) and not _point_in_polygon(cand, obstacle_boundary):
            return cand

    return best_fallback


def _build_offset_loop(boundary, terrain_boundary, obstacle_boundary, offset_m, *, n_samples=72, phase=0.0):
    anchors = _resample_closed_path(boundary, n_samples)
    winding = np.sign(_polygon_area(anchors))
    if abs(winding) < 1e-9:
        winding = 1.0
    loop = []
    for idx, pt in enumerate(anchors):
        prev_pt = anchors[(idx - 1) % len(anchors)]
        next_pt = anchors[(idx + 1) % len(anchors)]
        tangent = next_pt - prev_pt
        if winding > 0.0:
            normal = np.array([tangent[1], -tangent[0]], dtype=float)
        else:
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
        loop.append(_project_to_passable(pt, normal, offset_m, terrain_boundary, obstacle_boundary))
    loop = np.asarray(loop, dtype=float)
    if len(loop) > 0 and abs(float(phase)) > 1e-12:
        shift = int(round(float(phase) * len(loop))) % len(loop)
        loop = np.roll(loop, -shift, axis=0)
    return loop


def _sample_loop_points(loop, fractions):
    arr = np.asarray(loop, dtype=float)
    if len(arr) == 0:
        return []
    picks = []
    for frac in fractions:
        idx = int(round(float(frac) * len(arr))) % len(arr)
        picks.append(arr[idx].copy())
    return picks


def _regions_match(regions_a, regions_b):
    if len(regions_a) != len(regions_b):
        return False
    for reg_a, reg_b in zip(regions_a, regions_b):
        arr_a = np.asarray(reg_a, dtype=float)
        arr_b = np.asarray(reg_b, dtype=float)
        if arr_a.shape != arr_b.shape:
            return False
        if np.max(np.abs(arr_a - arr_b)) > 1e-6:
            return False
    return True


def _default_terrain_json_path():
    for base_dir in _annotation_dir_candidates():
        candidate = os.path.join(base_dir, "地形边界.json")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_annotation_dir_candidates()[0], "地形边界.json")


def _default_manual_routes_path():
    for base_dir in _annotation_dir_candidates():
        candidate = os.path.join(base_dir, "manual_routes_annotations.json")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_annotation_dir_candidates()[0], "manual_routes_annotations.json")


def _annotation_dir_candidates():
    return [str(SCENE_DATA)]


def _manual_drone_zone_overlays(manual):
    overlays = []
    for item in manual.get("transport_drones", []):
        name = str(item.get("name", f"Drone-{len(overlays) + 1}"))
        for zone_key, label_suffix in (("start_zone_m", "S"), ("drop_zone_m", "D")):
            points = np.asarray(item.get(zone_key, []), dtype=float)
            if len(points) < 3:
                continue
            overlays.append({
                "name": name,
                "kind": zone_key,
                "label": f"{name}-{label_suffix}",
                "points": points.tolist(),
            })
    return overlays


def _load_real_terrain_geometry(terrain_json_path=None):
    terrain_json_path = terrain_json_path or _default_terrain_json_path()
    with open(terrain_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    terrain_boundary = np.asarray(data["terrain_boundary_cm"], dtype=float) / 100.0
    no_go_regions = [np.asarray(region, dtype=float) / 100.0 for region in data.get("no_go_regions_cm", []) if len(region) >= 3]
    green_regions = [np.asarray(region, dtype=float) / 100.0 for region in data.get("green_regions_cm", []) if len(region) >= 3]
    if len(no_go_regions) == 0 and len(green_regions) == 0:
        raise ValueError("No no-go or green geometry found in terrain boundary file.")

    display_green_regions = green_regions
    if len(no_go_regions) > 0 and len(green_regions) > 0 and _regions_match(green_regions, no_go_regions):
        display_green_regions = []

    obstacle_regions = no_go_regions if len(no_go_regions) > 0 else green_regions
    main_no_go = max(obstacle_regions, key=lambda region: abs(_polygon_area(region)))

    charging_station_defs = []
    for idx, station in enumerate(data.get("charging_stations_cm", [])):
        point_cm = station.get("point_cm")
        if point_cm is None or len(point_cm) < 2:
            continue
        charging_station_defs.append({
            "id": int(station.get("id", idx)),
            "name": str(station.get("name", f"S{idx + 1}")),
            "pos": np.asarray(point_cm[:2], dtype=float) / 100.0,
        })
    charging_station_defs.sort(key=lambda item: item["id"])

    box = data["bounding_box_cm"]
    field_bounds = {
        "x_min": float(box["x_min"]) / 100.0,
        "x_max": float(box["x_max"]) / 100.0,
        "y_min": float(box["y_min"]) / 100.0,
        "y_max": float(box["y_max"]) / 100.0,
    }

    return {
        "terrain_json_path": terrain_json_path,
        "terrain_boundary": terrain_boundary,
        "green_regions": display_green_regions,
        "no_go_regions": no_go_regions,
        "main_no_go": main_no_go,
        "charging_stations": charging_station_defs,
        "field_bounds": field_bounds,
    }


def _first_passable_idx(loop, all_no_go_regions, start_hint=0):
    """Return the index (from start_hint) of the first loop point not inside any no-go region."""
    n = len(loop)
    for offset in range(n):
        idx = (start_hint + offset) % n
        if not any(_point_in_polygon(loop[idx], ng) for ng in all_no_go_regions):
            return idx
    return int(start_hint) % max(n, 1)  # fallback


def build_real_mixed_scenario(terrain_json_path=None, flight_profiles_dir=None):
    """Build a scenario from the local terrain annotation. Convert centimeters to meters, construct three ground patrol loops, load three station positions, and assign processed flight power profiles to three drones."""
    geom = _load_real_terrain_geometry(terrain_json_path)
    terrain_boundary = geom["terrain_boundary"]
    green_regions = geom["green_regions"]
    no_go_regions = geom["no_go_regions"]
    main_no_go = geom["main_no_go"]
    safe_no_go_hull = _convex_hull(main_no_go)
    center = np.mean(safe_no_go_hull, axis=0)
    field_bounds = geom["field_bounds"]
    flight_profiles = load_drone_profiles(processed_dir=flight_profiles_dir)

    # Three smooth inset loops derived from the terrain boundary.
    # This keeps adjacent segments inside the ellipse more reliably than direct radial chords.
    explore_loops = [
        _build_terrain_inset_loop(center, terrain_boundary, no_go_regions, 0.9, n_samples=180),
        _build_terrain_inset_loop(center, terrain_boundary, no_go_regions, 2.1, n_samples=180),
        _build_terrain_inset_loop(center, terrain_boundary, no_go_regions, 3.3, n_samples=180),
    ]
    # Spread rovers 120° apart; each start is validated to be outside any no-go region
    n_pts = [len(lp) for lp in explore_loops]
    phase_hints = [0, n_pts[1] // 3, 2 * n_pts[2] // 3]
    for li in range(3):
        loop = explore_loops[li]
        if len(loop) == 0:
            continue
        start = _first_passable_idx(loop, no_go_regions, start_hint=phase_hints[li])
        explore_loops[li] = np.roll(loop, -start, axis=0)

    service_ring = _build_terrain_inset_loop(center, terrain_boundary, no_go_regions, 2.7, n_samples=180)
    dropoff_pts = [
        _safe_task_point(pt, center, terrain_boundary, no_go_regions)
        for pt in _sample_loop_points(service_ring, [0.08, 0.39, 0.71])
    ]

    station_defs = geom["charging_stations"]
    if len(station_defs) >= 3:
        station_pos = [np.asarray(st["pos"], dtype=float) for st in station_defs[:3]]
    else:
        station_ring = _build_terrain_inset_loop(center, terrain_boundary, no_go_regions, 1.8, n_samples=180)
        station_pos = _sample_loop_points(station_ring, [0.08, 0.43, 0.77])

    station_access_pos = [
        _safe_task_point(pt, center, terrain_boundary, no_go_regions)
        for pt in station_pos
    ]

    # Station battery: 6.5 Ah × 14.8 V (4S LiPo) ≈ 96 Wh = 0.096 kWh; PV panel: 200 W
    _ST_E_kWh = 0.096
    stations = [
        {"id": 0, "pos": tuple(station_pos[0]),
         "E_station_kWh": _ST_E_kWh, "soc0_station": 0.60, "soc_min_station": 0.15, "soc_max_station": 0.95,
         "approach_radius_m": 0.45,
         "pv_peak_kW": 0.20, "P_aux_kW": 0.005,
         "P_st_ch_max_kW": 0.22, "P_st_dis_max_kW": 0.22,
         "P_tx_max_kW": 0.20, "P_tx_ramp_kW_per_step": 0.02},

        {"id": 1, "pos": tuple(station_pos[1]),
         "E_station_kWh": _ST_E_kWh, "soc0_station": 0.55, "soc_min_station": 0.15, "soc_max_station": 0.95,
         "approach_radius_m": 0.45,
         "pv_peak_kW": 0.20, "P_aux_kW": 0.005,
         "P_st_ch_max_kW": 0.22, "P_st_dis_max_kW": 0.22,
         "P_tx_max_kW": 0.20, "P_tx_ramp_kW_per_step": 0.02},

        {"id": 2, "pos": tuple(station_pos[2]),
         "E_station_kWh": _ST_E_kWh, "soc0_station": 0.65, "soc_min_station": 0.15, "soc_max_station": 0.95,
         "approach_radius_m": 0.45,
         "pv_peak_kW": 0.20, "P_aux_kW": 0.005,
         "P_st_ch_max_kW": 0.22, "P_st_dis_max_kW": 0.22,
         "P_tx_max_kW": 0.20, "P_tx_ramp_kW_per_step": 0.02},
    ]

    explore_rovers = [
        {"name": "Explore-Inner",
         "start_pos": tuple(explore_loops[0][0]),
         "soc0": 0.28,   # starts just below soc_stop → charges immediately (staggered)
         "pv_peak_kW": 0.0,
         "E_bat_kWh": 0.24,
         "P_ch_max_kW": 0.18,
         "P_dis_max_kW": 0.60,
         "path_points": explore_loops[0].tolist(),
         "path_idx": 0,
         "P_drive_explore_kW": 0.18, "P_aux_kW": 0.03,
         "P_arm_base_kW": 0.07, "P_arm_amp_kW": 0.03, "arm_phase": 0.4},

        {"name": "Explore-Mid",
         "start_pos": tuple(explore_loops[1][0]),
         "soc0": 0.36,   # charges after ~5 min (staggered)
         "pv_peak_kW": 0.0,
         "E_bat_kWh": 0.22,
         "P_ch_max_kW": 0.18,
         "P_dis_max_kW": 0.60,
         "path_points": explore_loops[1].tolist(),
         "path_idx": 0,
         "P_drive_explore_kW": 0.19, "P_aux_kW": 0.03,
         "P_arm_base_kW": 0.08, "P_arm_amp_kW": 0.03, "arm_phase": 1.4},

        {"name": "Explore-Outer",
         "start_pos": tuple(explore_loops[2][0]),
         "soc0": 0.44,   # charges after ~10 min (staggered)
         "pv_peak_kW": 0.0,
         "E_bat_kWh": 0.23,
         "P_ch_max_kW": 0.18,
         "P_dis_max_kW": 0.60,
         "path_points": explore_loops[2].tolist(),
         "path_idx": 0,
         "P_drive_explore_kW": 0.20, "P_aux_kW": 0.03,
         "P_arm_base_kW": 0.08, "P_arm_amp_kW": 0.04, "arm_phase": 2.3},
    ]

    cargo_pairs = [
        {"cargo_id": 0, "pickup": tuple(station_access_pos[0]), "dropoff": tuple(dropoff_pts[0])},
        {"cargo_id": 1, "pickup": tuple(station_access_pos[1]), "dropoff": tuple(dropoff_pts[1])},
        {"cargo_id": 2, "pickup": tuple(station_access_pos[2]), "dropoff": tuple(dropoff_pts[2])},
    ]

    drone_soc0 = [0.68, 0.50, 0.36]   # staggered so each drone also charges during sim
    drone_charge_cap = [0.11, 0.12, 0.12]
    transport_rovers = []
    for idx, profile in enumerate(flight_profiles[:3]):
        pair = cargo_pairs[idx]
        pickup = np.asarray(pair["pickup"], dtype=float)
        dropoff = np.asarray(pair["dropoff"], dtype=float)
        route_dist = float(np.linalg.norm(dropoff - pickup))
        leg_steps = max(len(profile["power_profile_kW"]), 1)
        air_step = route_dist / float(leg_steps)
        transport_rovers.append({
            "name": f"Drone-{profile['label']}",
            "start_pos": tuple(pair["pickup"]),
            "soc0": drone_soc0[idx],
            "pv_peak_kW": 0.0,
            "E_bat_kWh": profile["battery_capacity_kWh"],
            "P_ch_max_kW": drone_charge_cap[idx],
            "P_dis_max_kW": 0.18,
            "cargo_pairs": [pair],
            "first_cargo_id": pair["cargo_id"],
            "fixed_cargo_id": pair["cargo_id"],
            "first_stage": "TO_DROPOFF",
            "mobility": "air",
            "target_height_m": profile["target_height_m"],
            "transport_power_profile_kW": profile["power_profile_kW"],
            "transport_avg_power_kW": profile["average_power_kW"],
            "transport_sample_interval_s": profile["sample_interval_s"],
            "transport_leg_duration_s": profile["duration_s"],
            "v_air_per_step": air_step,
            "v_to_station_per_step": max(air_step * 1.15, 1e-6),
            "go_station_power_kW": profile["average_power_kW"],
            "home_station_id": int(stations[idx]["id"]),  # always return to own base
        })

    scene = {
        "name": "real_terrain_manual_no_go",
        "field_bounds": field_bounds,
        "terrain_boundary": terrain_boundary.tolist(),
        "terrain_is_green": False,         # only detected green_regions are grass; ellipse is the field boundary
        "green_regions": [region.tolist() for region in green_regions],
        "no_go_regions": [region.tolist() for region in no_go_regions],
        "main_no_go": main_no_go.tolist(),
        "routing_ring": service_ring.tolist(),
        "charging_stations": [
            {
                "id": int(st["id"]),
                "name": st["name"],
                "pos": np.asarray(st["pos"], dtype=float).tolist(),
                "approach_radius_m": float(stations[idx].get("approach_radius_m", 0.45)),
            }
            for idx, st in enumerate(geom["charging_stations"][:len(stations)])
        ],
        "source_file": geom["terrain_json_path"],
    }
    return stations, explore_rovers, transport_rovers, scene


def build_manual_mixed_scenario(manual_routes_path=None, terrain_json_path=None, flight_profiles_dir=None):
    """Build a scenario from manually annotated local routes. Ground vehicles follow the supplied path points; drones travel between their waypoints. Station locations and terrain come from the supplied scene JSON."""
    manual_routes_path = manual_routes_path or _default_manual_routes_path()
    if not os.path.exists(manual_routes_path):
        raise FileNotFoundError(f"Manual routes file not found: {manual_routes_path}")

    with open(manual_routes_path, "r", encoding="utf-8") as f:
        manual = json.load(f)

    geom = _load_real_terrain_geometry(terrain_json_path)
    terrain_boundary = geom["terrain_boundary"]
    green_regions = geom["green_regions"]
    no_go_regions = geom["no_go_regions"]
    main_no_go = geom["main_no_go"]
    center = np.mean(_convex_hull(main_no_go), axis=0)
    field_bounds = geom["field_bounds"]
    flight_profiles = load_drone_profiles(processed_dir=flight_profiles_dir)
    drone_zone_overlays = _manual_drone_zone_overlays(manual)
    drone_zone_regions = [
        np.asarray(zone["points"], dtype=float)
        for zone in drone_zone_overlays
        if len(zone.get("points", [])) >= 3
    ]
    ground_no_go_regions = [np.asarray(region, dtype=float).copy() for region in no_go_regions]
    ground_no_go_regions.extend(np.asarray(region, dtype=float).copy() for region in green_regions)
    ground_no_go_regions.extend(np.asarray(region, dtype=float).copy() for region in drone_zone_regions)

    station_defs = geom["charging_stations"]
    if len(station_defs) < 3:
        raise ValueError("Manual route mode requires at least 3 charging stations in 地形边界.json")
    station_pos = [np.asarray(st["pos"], dtype=float) for st in station_defs[:3]]

    # Only one routing loop is needed in manual-route mode; keep it coarse so scene build stays fast.
    service_ring = _build_terrain_inset_loop(center, terrain_boundary, ground_no_go_regions, 2.7, n_samples=48)
    station_access_pos = [
        service_ring[int(np.argmin(np.linalg.norm(service_ring - pos.reshape(1, 2), axis=1)))]
        for pos in station_pos
    ]

    _ST_E_kWh = 0.096
    stations = [
        {"id": 0, "pos": tuple(station_pos[0]),
         "E_station_kWh": _ST_E_kWh, "soc0_station": 0.60, "soc_min_station": 0.15, "soc_max_station": 0.95,
         "ground_access_pos": tuple(station_access_pos[0]),
         "approach_radius_m": 0.45,
         "pv_peak_kW": 0.20, "P_aux_kW": 0.005,
         "P_st_ch_max_kW": 0.22, "P_st_dis_max_kW": 0.22,
         "P_tx_max_kW": 0.20, "P_tx_ramp_kW_per_step": 0.02},
        {"id": 1, "pos": tuple(station_pos[1]),
         "E_station_kWh": _ST_E_kWh, "soc0_station": 0.55, "soc_min_station": 0.15, "soc_max_station": 0.95,
         "ground_access_pos": tuple(station_access_pos[1]),
         "approach_radius_m": 0.45,
         "pv_peak_kW": 0.20, "P_aux_kW": 0.005,
         "P_st_ch_max_kW": 0.22, "P_st_dis_max_kW": 0.22,
         "P_tx_max_kW": 0.20, "P_tx_ramp_kW_per_step": 0.02},
        {"id": 2, "pos": tuple(station_pos[2]),
         "E_station_kWh": _ST_E_kWh, "soc0_station": 0.65, "soc_min_station": 0.15, "soc_max_station": 0.95,
         "ground_access_pos": tuple(station_access_pos[2]),
         "approach_radius_m": 0.45,
         "pv_peak_kW": 0.20, "P_aux_kW": 0.005,
         "P_st_ch_max_kW": 0.22, "P_st_dis_max_kW": 0.22,
         "P_tx_max_kW": 0.20, "P_tx_ramp_kW_per_step": 0.02},
    ]

    rover_defaults = {
        "Explore-Inner": {"soc0": 0.42, "E_bat_kWh": 0.24, "P_drive_explore_kW": 0.18, "P_arm_base_kW": 0.07, "P_arm_amp_kW": 0.03, "arm_phase": 0.4},
        "Explore-Mid": {"soc0": 0.325, "soc_stop": 0.325, "soc_resume": 0.40, "force_initial_charge": True, "E_bat_kWh": 0.22, "P_drive_explore_kW": 0.19, "P_arm_base_kW": 0.08, "P_arm_amp_kW": 0.03, "arm_phase": 1.4},
        "Explore-Outer": {"soc0": 0.54, "E_bat_kWh": 0.23, "P_drive_explore_kW": 0.20, "P_arm_base_kW": 0.08, "P_arm_amp_kW": 0.04, "arm_phase": 2.3},
    }

    explore_rovers = []
    for item in manual.get("explore_rovers", []):
        name = str(item.get("name", f"Explore-{len(explore_rovers) + 1}"))
        path_points = np.asarray(item.get("path_points_m", []), dtype=float)
        if len(path_points) < 2:
            raise ValueError(f"Explore rover {name} must provide at least 2 path_points_m entries")
        defaults = rover_defaults.get(name, rover_defaults["Explore-Mid"])
        start_pos = np.asarray(item.get("start_pos_m") or path_points[0], dtype=float)
        if np.linalg.norm(start_pos - path_points[0]) > 1e-6:
            path_points = np.vstack([start_pos.reshape(1, 2), path_points])

        can_force_initial_charge = bool(defaults.get("force_initial_charge", False)) and _point_is_passable(
            start_pos,
            terrain_boundary,
            ground_no_go_regions,
        )

        initial_charge_goal = None
        initial_charge_station_id = None
        if can_force_initial_charge:
            visible_ring_nodes = [
                node for node in service_ring
                if _segment_is_passable(start_pos, node, terrain_boundary, ground_no_go_regions)
            ]
            if len(visible_ring_nodes) > 0:
                visible_ring_nodes = np.asarray(visible_ring_nodes, dtype=float)
                anchor_station_idx = 1 if len(stations) > 1 else 0
                anchor_access = np.asarray(stations[anchor_station_idx]["ground_access_pos"], dtype=float)
                chosen_goal = visible_ring_nodes[int(np.argmin(np.linalg.norm(visible_ring_nodes - anchor_access.reshape(1, 2), axis=1)))]
                initial_charge_goal = tuple(chosen_goal)
                initial_charge_station_id = int(stations[anchor_station_idx]["id"])

        explore_rovers.append({
            "name": name,
            "start_pos": tuple(start_pos),
            "soc0": float(defaults["soc0"]),
            "soc_stop": float(defaults.get("soc_stop", 0.28)),
            "soc_resume": float(defaults.get("soc_resume", 0.40)),
            "force_initial_charge": bool(can_force_initial_charge),
            "initial_charge_goal": initial_charge_goal,
            "initial_charge_station_id": initial_charge_station_id,
            "pv_peak_kW": 0.0,
            "E_bat_kWh": float(defaults["E_bat_kWh"]),
            "P_ch_max_kW": 0.18,
            "P_dis_max_kW": 0.60,
            "path_points": path_points.tolist(),
            "path_idx": 0,
            "loop": bool(item.get("loop", True)),
            "manual_path": True,
            "P_drive_explore_kW": float(defaults["P_drive_explore_kW"]),
            "P_aux_kW": 0.03,
            "P_arm_base_kW": float(defaults["P_arm_base_kW"]),
            "P_arm_amp_kW": float(defaults["P_arm_amp_kW"]),
            "arm_phase": float(defaults["arm_phase"]),
        })

    profile_by_name = {f"Drone-{profile['label']}": profile for profile in flight_profiles}
    drone_soc0 = {"Drone-3m": 0.68, "Drone-6m": 0.50, "Drone-9m": 0.36}
    drone_charge_cap = {"Drone-3m": 0.11, "Drone-6m": 0.12, "Drone-9m": 0.12}

    def nearest_station_id(point):
        point = np.asarray(point, dtype=float)
        best_sid = 0
        best_dist = float("inf")
        for st in stations:
            dist = float(np.linalg.norm(point - np.asarray(st["pos"], dtype=float)))
            if dist < best_dist:
                best_dist = dist
                best_sid = int(st["id"])
        return best_sid

    transport_rovers = []
    for item in manual.get("transport_drones", []):
        name = str(item.get("name", f"Drone-{len(transport_rovers) + 1}"))
        profile = profile_by_name.get(name)
        if profile is None:
            raise ValueError(f"No real drone power profile found for {name}")
        waypoints = np.asarray(item.get("waypoints_m", []), dtype=float)
        if len(waypoints) < 2:
            raise ValueError(f"Transport drone {name} must provide at least 2 waypoints_m entries")

        start_pos = np.asarray(item.get("start_pos_m") or waypoints[0], dtype=float)
        if np.linalg.norm(start_pos - waypoints[0]) > 1e-6:
            waypoints = np.vstack([start_pos.reshape(1, 2), waypoints])

        cargo_pairs = []
        for idx in range(len(waypoints) - 1):
            cargo_pairs.append({
                "cargo_id": idx,
                "pickup": tuple(waypoints[idx]),
                "dropoff": tuple(waypoints[idx + 1]),
            })

        segment_lengths = [float(np.linalg.norm(np.asarray(cp["dropoff"]) - np.asarray(cp["pickup"]))) for cp in cargo_pairs]
        mean_segment_dist = float(np.mean(segment_lengths)) if len(segment_lengths) > 0 else 1e-6
        leg_steps = max(len(profile["power_profile_kW"]), 1)
        air_step = max(mean_segment_dist / float(leg_steps), 1e-6)

        transport_rovers.append({
            "name": name,
            "start_pos": tuple(start_pos),
            "soc0": float(drone_soc0.get(name, 0.50)),
            "pv_peak_kW": 0.0,
            "E_bat_kWh": profile["battery_capacity_kWh"],
            "P_ch_max_kW": float(drone_charge_cap.get(name, 0.12)),
            "P_dis_max_kW": 0.18,
            "cargo_pairs": cargo_pairs,
            "first_cargo_id": int(cargo_pairs[0]["cargo_id"]),
            "fixed_cargo_id": int(cargo_pairs[0]["cargo_id"]),
            "route_mode": "sequence",
            "route_pair_idx": 0,
            "mobility": "air",
            "target_height_m": profile["target_height_m"],
            "transport_power_profile_kW": profile["power_profile_kW"],
            "transport_avg_power_kW": profile["average_power_kW"],
            "transport_sample_interval_s": profile["sample_interval_s"],
            "transport_leg_duration_s": profile["duration_s"],
            "v_air_per_step": air_step,
            "v_to_station_per_step": max(air_step * 1.15, 1e-6),
            "go_station_power_kW": profile["average_power_kW"],
            "home_station_id": nearest_station_id(start_pos),
            "manual_waypoints": waypoints.tolist(),
        })

    scene = {
        "name": "real_terrain_manual_routes",
        "field_bounds": field_bounds,
        "terrain_boundary": terrain_boundary.tolist(),
        "terrain_is_green": False,
        "green_regions": [region.tolist() for region in green_regions],
        "no_go_regions": [region.tolist() for region in no_go_regions],
        "ground_no_go_regions": [region.tolist() for region in ground_no_go_regions],
        "drone_takeoff_landing_zones": drone_zone_overlays,
        "main_no_go": main_no_go.tolist(),
        "routing_ring": service_ring.tolist(),
        "ground_clearance_m": 0.55,
        "charging_stations": [
            {
                "id": int(st["id"]),
                "name": st["name"],
                "pos": np.asarray(st["pos"], dtype=float).tolist(),
                "approach_radius_m": float(stations[idx].get("approach_radius_m", 0.45)),
            }
            for idx, st in enumerate(geom["charging_stations"][:len(stations)])
        ],
        "source_file": geom["terrain_json_path"],
        "manual_routes_file": manual_routes_path,
    }
    return stations, explore_rovers, transport_rovers, scene


def build_default_mixed_scenario():
    """
    Soccer-field scenario (meters):
    - Field: 105 m × 68 m, origin at lower-left (0,0)
    - 3 stations
    - 6 rovers: 3 explore (triangle/circle/square) + 3 transport
    - 3 cargo pairs
    """

    # 3 charging stations
    stations = [
        {"id": 0, "pos": (10.0, 10.0),
         "E_station_kWh": 180.0, "soc0_station": 0.80, "soc_min_station": 0.20, "soc_max_station": 0.95,
         "approach_radius_m": 1.0,
         "pv_peak_kW": 55.0, "P_aux_kW": 0.40,
         "P_st_ch_max_kW": 30.0, "P_st_dis_max_kW": 45.0,
         "P_tx_max_kW": 40.0, "P_tx_ramp_kW_per_step": 6.0},

        {"id": 1, "pos": (95.0, 10.0),
         "E_station_kWh": 160.0, "soc0_station": 0.75, "soc_min_station": 0.20, "soc_max_station": 0.95,
         "approach_radius_m": 1.0,
         "pv_peak_kW": 50.0, "P_aux_kW": 0.40,
         "P_st_ch_max_kW": 28.0, "P_st_dis_max_kW": 42.0,
         "P_tx_max_kW": 38.0, "P_tx_ramp_kW_per_step": 6.0},

        {"id": 2, "pos": (52.5, 58.0),
         "E_station_kWh": 170.0, "soc0_station": 0.78, "soc_min_station": 0.20, "soc_max_station": 0.95,
         "approach_radius_m": 1.0,
         "pv_peak_kW": 52.0, "P_aux_kW": 0.40,
         "P_st_ch_max_kW": 28.0, "P_st_dis_max_kW": 44.0,
         "P_tx_max_kW": 40.0, "P_tx_ramp_kW_per_step": 6.0},
    ]

    # 3 explore rovers: triangle / circle / square
    explore_rovers = [
        {"name": "Explore-Triangle",
         "start_pos": (52.5, 12.0),
         "soc0": 0.68,
         "pv_peak_kW": 1.2,
         "path_points": _make_triangle_path(p1=(30.0, 14.0), p2=(75.0, 14.0), p3=(52.5, 54.0), n_edge=8),
         "path_idx": 0,
         "P_drive_explore_kW": 0.70, "P_aux_kW": 0.24,
         "P_arm_base_kW": 0.50, "P_arm_amp_kW": 0.40, "arm_phase": 1.0},

        {"name": "Explore-Circle",
         "start_pos": (18.0, 18.0),
         "soc0": 0.70,
         "pv_peak_kW": 1.4,
         "path_points": _make_circle_path(center=(52.5, 34.0), r=18.0, n=18),
         "path_idx": 0,
         "P_drive_explore_kW": 0.75, "P_aux_kW": 0.22,
         "P_arm_base_kW": 0.55, "P_arm_amp_kW": 0.36, "arm_phase": 0.2},

        {"name": "Explore-Square",
         "start_pos": (18.0, 56.0),
         "soc0": 0.72,
         "pv_peak_kW": 1.3,
         "path_points": _make_square_path(x0=34.0, y0=18.0, w=30.0, n_edge=7),
         "path_idx": 0,
         "P_drive_explore_kW": 0.78, "P_aux_kW": 0.20,
         "P_arm_base_kW": 0.60, "P_arm_amp_kW": 0.32, "arm_phase": 2.1},
    ]

    # 3 cargo pairs
    cargo_pairs = [
        {"cargo_id": 0, "pickup": (24.0, 20.0), "dropoff": (86.0, 52.0)},
        {"cargo_id": 1, "pickup": (22.0, 54.0), "dropoff": (90.0, 18.0)},
        {"cargo_id": 2, "pickup": (52.5, 34.0), "dropoff": (96.0, 34.0)},
    ]

    # 3 transport rovers
    transport_rovers = [
        {"name": "Transport-A",
         "start_pos": (8.0, 8.0),
         "soc0": 0.78,
         "pv_peak_kW": 1.0,
         "cargo_pairs": cargo_pairs,
         "first_cargo_id": 0,
         "P_drive_kW": 1.25, "P_drive_cargo_kW": 1.85,
         "P_aux_kW": 0.22, "P_handle_base_kW": 0.10, "P_handle_amp_kW": 0.18, "handle_phase": 0.4},

        {"name": "Transport-B",
         "start_pos": (98.0, 8.0),
         "soc0": 0.64,
         "pv_peak_kW": 0.9,
         "cargo_pairs": cargo_pairs,
         "first_cargo_id": 1,
         "P_drive_kW": 1.20, "P_drive_cargo_kW": 1.80,
         "P_aux_kW": 0.24, "P_handle_base_kW": 0.11, "P_handle_amp_kW": 0.17, "handle_phase": 1.3},

        {"name": "Transport-C",
         "start_pos": (52.5, 60.0),
         "soc0": 0.70,
         "pv_peak_kW": 1.1,
         "cargo_pairs": cargo_pairs,
         "first_cargo_id": 2,
         "P_drive_kW": 1.30, "P_drive_cargo_kW": 1.95,
         "P_aux_kW": 0.20, "P_handle_base_kW": 0.09, "P_handle_amp_kW": 0.19, "handle_phase": 2.2},
    ]

    return stations, explore_rovers, transport_rovers
