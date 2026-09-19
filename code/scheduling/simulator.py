import numpy as np
from kinematics import move_toward
from planning import estimate_soc_after_travel, estimate_charge_steps, sort_stations_by_distance
from station_manager import StationManager


def simulate_mixed_system(
    pv_env,
    stations,
    explore_rovers,
    transport_rovers,
    scene=None,
    N_steps=288,
    dt_hours=0.25,

    E_bat_kWh=60.0,
    soc_min=0.20,
    soc_max=0.90,
    eta_ch=0.95,
    eta_dis=0.95,
    P_ch_max_kW=12.0,
    P_dis_max_kW=15.0,

    eta_wireless=0.92,

    soc_stop=0.32,
    soc_resume=0.72,
    safety_soc=0.03,

    # NEW: transport stricter resume to avoid micro-charge chatter
    soc_resume_transport=0.80,
    min_charge_steps=3,
    min_charge_steps_transport=8,

    P_idle_kW=0.20,
    P_charge_aux_kW=0.10,

    P_return_base_kW=0.35,
    k_return_kW_per_unit=0.0010,

    v_explore_per_step=25.0,
    v_transport_per_step=6.0,
    v_to_station_per_step=18.0,
    arrival_eps=10.0,

    extra_leave_margin=0.06,
    enable_wait=True,

    # NEW: drones stop after this many deliveries (round trips); rovers keep working.
    round_trip_count=None,
    # NEW: in the final `return_tail_steps` steps, explore rovers return to the nearest
    # station and park there, so their trajectories end at a charging station.
    return_tail_steps=0,
):
    pv_env = np.asarray(pv_env, dtype=float).reshape(-1)
    N = min(int(N_steps), len(pv_env))

    rovers = []
    for rv in explore_rovers:
        r = dict(rv); r["type"] = "explore"; rovers.append(r)
    for rv in transport_rovers:
        r = dict(rv); r["type"] = "transport"; rovers.append(r)

    K = len(rovers)
    S = len(stations)

    sm = StationManager(stations, dt_hours=dt_hours)

    pos = np.zeros((K, 2), dtype=float)
    soc = np.zeros(K, dtype=float)
    E = np.zeros(K, dtype=float)
    E_cap = np.zeros(K, dtype=float)
    soc_floor = np.zeros(K, dtype=float)
    soc_ceiling = np.zeros(K, dtype=float)
    P_ch_max_arr = np.zeros(K, dtype=float)
    P_dis_max_arr = np.zeros(K, dtype=float)

    mode = np.array(["WORK"] * K, dtype=object)  # WORK / GO_STATION / WAIT_STATION / CHARGE / RETURN_RESUME / WAIT_PV
    target_station = np.full(K, -1, dtype=int)

    resume_pos = np.zeros((K, 2), dtype=float)

    transport_stage = np.array(["TO_PICKUP"] * K, dtype=object)
    transport_goal = np.zeros((K, 2), dtype=float)
    transport_job_id = np.zeros(K, dtype=int)
    transport_cargo_id = np.full(K, -1, dtype=int)
    transport_leg_origin = np.zeros((K, 2), dtype=float)
    transport_leg_step = np.zeros(K, dtype=int)
    transport_leg_total = np.ones(K, dtype=int)

    op_timer = np.zeros(K, dtype=int)
    charge_step_cnt = np.zeros(K, dtype=int)
    explore_completion_count = np.zeros(K, dtype=int)
    transport_delivery_count = np.zeros(K, dtype=int)
    transport_done = np.zeros(K, dtype=bool)
    end_game = np.zeros(K, dtype=bool)

    delivery_cap = None if round_trip_count is None else int(round_trip_count)
    return_trigger_step = (N - int(return_tail_steps)) if int(return_tail_steps) > 0 else None

    nav_goal = np.zeros((K, 2), dtype=float)
    nav_mode = np.array([""] * K, dtype=object)
    nav_queue = [[] for _ in range(K)]
    charge_goal = np.zeros((K, 2), dtype=float)
    charge_goal_valid = np.zeros(K, dtype=bool)

    for k, rv in enumerate(rovers):
        pos[k] = np.asarray(rv.get("start_pos", (0.0, 0.0)), dtype=float)
        E_cap[k] = float(rv.get("E_bat_kWh", E_bat_kWh))
        soc_floor[k] = float(rv.get("soc_min", soc_min))
        soc_ceiling[k] = float(rv.get("soc_max", soc_max))
        P_ch_max_arr[k] = float(rv.get("P_ch_max_kW", P_ch_max_kW))
        P_dis_max_arr[k] = float(rv.get("P_dis_max_kW", P_dis_max_kW))

        soc0 = float(np.clip(rv.get("soc0", 0.7), soc_floor[k], soc_ceiling[k]))
        soc[k] = soc0
        E[k] = soc0 * E_cap[k]

        if rv["type"] == "explore":
            resume_pos[k] = pos[k].copy()
        else:
            _init_transport_first_job(k, rovers, transport_stage, transport_goal, transport_job_id, transport_cargo_id)

    E_min = soc_floor * E_cap
    E_max = soc_ceiling * E_cap

    # correlated noise
    noise = np.zeros((N, K), dtype=float)
    for k, rv in enumerate(rovers):
        seed = (abs(hash(rv.get("name", f"R{k}"))) % (2**32))
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(N) * 0.12
        w = 9
        ker = np.ones(w) / w
        noise[:, k] = np.convolve(raw, ker, mode="same")

    # records
    soc_hist = np.zeros((N, K))
    x_hist = np.zeros((N, K))
    y_hist = np.zeros((N, K))
    mode_hist = np.zeros((N, K), dtype=object)

    P_load_hist = np.zeros((N, K))
    P_pv_hist = np.zeros((N, K))
    P_station_used_hist = np.zeros((N, K))
    P_bat_hist = np.zeros((N, K))
    unmet_hist = np.zeros((N, K))

    charge_station_id_hist = np.full((N, K), -1, dtype=int)

    station_soc_hist = np.zeros((N, S))
    station_queue_hist = np.zeros((N, S), dtype=int)

    transport_job_hist = np.full((N, K), -1, dtype=int)
    transport_cargo_hist = np.full((N, K), -1, dtype=int)

    def arrived(p, goal):
        return float(np.linalg.norm(p - goal)) <= float(arrival_eps)

    def is_air_transport(k):
        return rovers[k]["type"] == "transport" and str(rovers[k].get("mobility", "ground")).lower() == "air"

    def work_speed(k):
        rv = rovers[k]
        if rv["type"] == "explore":
            return float(rv.get("v_explore_per_step", v_explore_per_step))
        if is_air_transport(k):
            return float(rv.get("v_air_per_step", v_transport_per_step))
        return float(rv.get("v_transport_per_step", v_transport_per_step))

    def station_speed(k):
        rv = rovers[k]
        if is_air_transport(k):
            return float(rv.get("v_to_station_per_step", rv.get("v_air_per_step", v_to_station_per_step)))
        return float(rv.get("v_to_station_per_step", v_to_station_per_step))

    def scene_obstacle_polygons():
        if scene is None:
            return []
        polys = []
        for region in scene.get("ground_no_go_regions", []):
            poly = np.asarray(region, dtype=float)
            if len(poly) >= 3:
                polys.append(poly)
        if len(polys) > 0:
            return polys
        for region in scene.get("no_go_regions", []):
            poly = np.asarray(region, dtype=float)
            if len(poly) >= 3:
                polys.append(poly)
        if len(polys) > 0:
            return polys
        main_no_go = scene.get("main_no_go", None)
        if main_no_go is not None:
            poly = np.asarray(main_no_go, dtype=float)
            if len(poly) >= 3:
                return [poly]
        for region in scene.get("green_regions", []):
            poly = np.asarray(region, dtype=float)
            if len(poly) >= 3:
                polys.append(poly)
        return polys

    obstacle_polys = scene_obstacle_polygons()
    terrain_poly = None if scene is None else np.asarray(scene.get("terrain_boundary", []), dtype=float)
    if terrain_poly is not None and len(terrain_poly) < 3:
        terrain_poly = None
    routing_ring = None if scene is None else np.asarray(scene.get("routing_ring", []), dtype=float)
    if routing_ring is not None and len(routing_ring) < 3:
        routing_ring = None
    ground_clearance_m = float(scene.get("ground_clearance_m", 0.55)) if scene is not None else 0.55

    def segment_cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, p):
        return (
            min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
        )

    def segments_intersect(a, b, c, d):
        o1 = segment_cross(a, b, c)
        o2 = segment_cross(a, b, d)
        o3 = segment_cross(c, d, a)
        o4 = segment_cross(c, d, b)

        if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
            return True
        if abs(o1) < 1e-9 and on_segment(a, b, c):
            return True
        if abs(o2) < 1e-9 and on_segment(a, b, d):
            return True
        if abs(o3) < 1e-9 and on_segment(c, d, a):
            return True
        if abs(o4) < 1e-9 and on_segment(c, d, b):
            return True
        return False

    def point_in_polygon(point, polygon):
        if polygon is None:
            return False
        x, y = float(point[0]), float(point[1])
        poly = np.asarray(polygon, dtype=float)
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            crosses = ((yi > y) != (yj > y))
            if crosses:
                x_cross = (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def point_in_any_obstacle(point):
        for poly in obstacle_polys:
            if point_in_polygon(point, poly):
                return True
        return False

    def point_in_terrain(point):
        return True if terrain_poly is None else point_in_polygon(point, terrain_poly)

    def segment_hits_obstacle(p0, p1):
        a = np.asarray(p0, dtype=float)
        b = np.asarray(p1, dtype=float)
        if not point_in_terrain(a) or not point_in_terrain(b):
            return True
        if terrain_poly is not None:
            for alpha in np.linspace(0.0, 1.0, 33):
                pt = a * (1.0 - float(alpha)) + b * float(alpha)
                if not point_in_terrain(pt):
                    return True
        if len(obstacle_polys) == 0:
            return False
        for obstacle_poly in obstacle_polys:
            if point_in_polygon(a, obstacle_poly) or point_in_polygon(b, obstacle_poly):
                return True
            if point_in_polygon((a + b) * 0.5, obstacle_poly):
                return True
            for idx in range(len(obstacle_poly)):
                c = obstacle_poly[idx]
                d = obstacle_poly[(idx + 1) % len(obstacle_poly)]
                if segments_intersect(a, b, c, d):
                    return True
        return False

    def route_is_clear(start, route):
        prev = np.asarray(start, dtype=float)
        for waypoint in route:
            wp = np.asarray(waypoint, dtype=float)
            if segment_hits_obstacle(prev, wp):
                return False
            prev = wp
        return True

    def station_target_point(origin, station):
        center = np.asarray(station["pos"], dtype=float)
        if is_air_transport(k_ref[0]):
            return center.copy()
        access = station.get("ground_access_pos", None)
        if access is not None:
            target = np.asarray(access, dtype=float)
            if point_in_terrain(target) and not point_in_any_obstacle(target):
                return target
        radius = float(station.get("approach_radius_m", 0.45))
        direction = np.asarray(origin, dtype=float) - center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([1.0, 0.0], dtype=float)
        else:
            direction = direction / norm
        target = center + direction * radius
        return target if not point_in_any_obstacle(target) else center.copy()

    k_ref = [0]

    def route_arc(i0, i1, step_dir):
        n_nodes = len(routing_ring)
        idx = int(i0)
        pts = [routing_ring[idx].copy()]
        total = 0.0
        while idx != int(i1):
            nxt = (idx + step_dir) % n_nodes
            total += float(np.linalg.norm(routing_ring[nxt] - routing_ring[idx]))
            pts.append(routing_ring[nxt].copy())
            idx = nxt
        return total, pts

    def plan_path(start, goal):
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if not segment_hits_obstacle(start, goal):
            return [goal.copy()]
        if routing_ring is None:
            return []

        visible_start = []
        visible_goal = []
        for idx, node in enumerate(routing_ring):
            if not segment_hits_obstacle(start, node):
                visible_start.append((float(np.linalg.norm(node - start)), idx))
            if not segment_hits_obstacle(goal, node):
                visible_goal.append((float(np.linalg.norm(node - goal)), idx))

        if len(visible_start) == 0 or len(visible_goal) == 0:
            return []

        visible_start.sort(key=lambda x: x[0])
        visible_goal.sort(key=lambda x: x[0])

        best_route = None
        best_len = None
        start_candidates = visible_start if len(visible_start) <= 24 else visible_start[:24]
        goal_candidates = visible_goal if len(visible_goal) <= 24 else visible_goal[:24]

        for d0, i0 in start_candidates:
            for d1, i1 in goal_candidates:
                cw_len, cw_pts = route_arc(i0, i1, +1)
                ccw_len, ccw_pts = route_arc(i0, i1, -1)
                for arc_len, arc_pts in [(cw_len, cw_pts), (ccw_len, ccw_pts)]:
                    total = d0 + arc_len + d1
                    candidate = [pt.copy() for pt in arc_pts] + [goal.copy()]
                    if best_len is None or total < best_len:
                        best_len = total
                        best_route = candidate

        return best_route if best_route is not None else []

    def planned_distance(start, goal):
        route = plan_path(start, goal)
        if len(route) == 0:
            return float("inf")
        total = 0.0
        prev = np.asarray(start, dtype=float)
        for waypoint in route:
            wp = np.asarray(waypoint, dtype=float)
            total += float(np.linalg.norm(wp - prev))
            prev = wp
        return total

    def estimate_soc_after_distance(soc_now, travel_dist, pv_peak_kW, t0, v_per_step, P_load_move_kW, E_bat_kWh_cap):
        steps = int(np.ceil(float(travel_dist) / max(float(v_per_step), 1e-6)))
        if steps <= 0:
            return float(soc_now)

        t1 = min(int(t0 + steps), len(pv_env))
        if t0 >= len(pv_env):
            pv_mean = 0.0
        else:
            pv_mean = float(np.mean(np.clip(pv_env[t0:t1], 0.0, 1.0))) * float(pv_peak_kW)

        deficit_mean = max(float(P_load_move_kW) - pv_mean, 0.0)
        E_need_kWh = (deficit_mean / max(float(eta_dis), 1e-9)) * steps * float(dt_hours)
        return float(soc_now) - E_need_kWh / max(float(E_bat_kWh_cap), 1e-9)

    def estimate_soc_after_profile(soc_now, profile_kW, start_step, pv_peak_kW, t0, E_bat_kWh_cap):
        profile = np.asarray(profile_kW, dtype=float)
        start_step = int(np.clip(start_step, 0, len(profile)))
        if start_step >= len(profile):
            return float(soc_now)

        span = len(profile) - start_step
        t1 = min(int(t0 + span), len(pv_env))
        if t0 >= len(pv_env):
            pv_mean = 0.0
        else:
            pv_mean = float(np.mean(np.clip(pv_env[t0:t1], 0.0, 1.0))) * float(pv_peak_kW)

        P_move_mean = float(np.mean(profile[start_step:]))
        deficit_mean = max(P_move_mean - pv_mean, 0.0)
        E_need_kWh = (deficit_mean / max(float(eta_dis), 1e-9)) * span * float(dt_hours)
        return float(soc_now) - E_need_kWh / max(float(E_bat_kWh_cap), 1e-9)

    def ensure_route(k, route_mode, final_goal):
        goal = np.asarray(final_goal, dtype=float)
        need_replan = (nav_mode[k] != route_mode) or (len(nav_queue[k]) == 0) or (float(np.linalg.norm(nav_goal[k] - goal)) > max(arrival_eps, 1e-6))
        if need_replan:
            if is_air_transport(k):
                nav_queue[k] = [goal.copy()]
            elif route_mode == "EXPLORE_WORK" and bool(rovers[k].get("manual_path", False)):
                nav_queue[k] = [goal.copy()]
            else:
                route = plan_path(pos[k], goal)
                nav_queue[k] = [np.asarray(pt, dtype=float) for pt in route] if len(route) > 0 else [pos[k].copy()]
            nav_goal[k] = goal.copy()
            nav_mode[k] = route_mode

    def clear_route(k):
        nav_queue[k] = []
        nav_mode[k] = ""

    def apply_ground_clearance(k, prev_pos, proposed_pos):
        if is_air_transport(k):
            return np.asarray(proposed_pos, dtype=float)

        prev = np.asarray(prev_pos, dtype=float)
        cand = np.asarray(proposed_pos, dtype=float)
        move_vec = cand - prev
        move_len = float(np.linalg.norm(move_vec))
        if move_len < 1e-9:
            return prev.copy()

        unit = move_vec / move_len
        allowed_len = move_len
        for other in range(K):
            if other == k or is_air_transport(other):
                continue

            other_pos = pos[other].copy()
            start_gap = float(np.linalg.norm(prev - other_pos))
            if start_gap < ground_clearance_m - 1e-9:
                return prev.copy()

            rel = other_pos - prev
            along = float(np.dot(rel, unit))
            along_clip = float(np.clip(along, 0.0, allowed_len))
            closest = prev + unit * along_clip
            lateral = float(np.linalg.norm(other_pos - closest))
            if lateral >= ground_clearance_m:
                continue

            backoff = np.sqrt(max(ground_clearance_m ** 2 - lateral ** 2, 0.0))
            allowed_len = min(allowed_len, max(along_clip - backoff, 0.0))

        safe = prev + unit * allowed_len
        return safe if point_in_terrain(safe) and not point_in_any_obstacle(safe) else prev.copy()

    def move_with_route(k, route_mode, final_goal, step_size):
        ensure_route(k, route_mode, final_goal)
        while len(nav_queue[k]) > 0 and arrived(pos[k], nav_queue[k][0]):
            nav_queue[k].pop(0)
        target = np.asarray(final_goal, dtype=float) if len(nav_queue[k]) == 0 else nav_queue[k][0]
        prev = pos[k].copy()
        proposed, _ = move_toward(pos[k], target, step_size)
        if route_mode == "EXPLORE_WORK" and bool(rovers[k].get("manual_path", False)):
            pos[k] = np.asarray(proposed, dtype=float)
        else:
            pos[k] = apply_ground_clearance(k, prev, proposed)
        moved = float(np.linalg.norm(pos[k] - prev))
        while len(nav_queue[k]) > 0 and arrived(pos[k], nav_queue[k][0]):
            nav_queue[k].pop(0)
        return moved

    def rover_pv_kW(k, t):
        pv_peak = float(rovers[k].get("pv_peak_kW", 1.2))
        return float(np.clip(pv_env[t], 0.0, 1.0)) * pv_peak

    def explore_next_point(k):
        rv = rovers[k]
        pts = np.asarray(rv["path_points"], dtype=float)
        if bool(rv.get("loop", True)):
            idx = int(rv.get("path_idx", 0)) % len(pts)
        else:
            idx = int(np.clip(int(rv.get("path_idx", 0)), 0, len(pts) - 1))
        return pts[idx], idx

    def explore_advance(k):
        rv = rovers[k]
        pts = np.asarray(rv["path_points"], dtype=float)
        idx = int(rv.get("path_idx", 0))
        if bool(rv.get("loop", True)):
            rv["path_idx"] = (idx + 1) % len(pts)
        else:
            rv["path_idx"] = min(idx + 1, len(pts) - 1)

    def explore_load_kW(k, t):
        rv = rovers[k]
        P_drive = float(rv.get("P_drive_explore_kW", 0.18))
        P_aux = float(rv.get("P_aux_kW", 0.03))
        base = float(rv.get("P_arm_base_kW", 0.07))
        amp = float(rv.get("P_arm_amp_kW", 0.03))
        phase = float(rv.get("arm_phase", 0.0))
        # Smooth slow sinusoid only (period ~200 steps; at 0.1 s/step → ~20 s period)
        arm = base + amp * np.sin(2.0 * np.pi * (t / 200.0) + phase)
        return P_idle_kW + P_drive + P_aux + max(arm, 0.0)

    def work_trace_load_kW(k, t):
        # Optional absolute-time measured output-power trace (kW) sampled by wall-clock
        # time. When present it overrides the WORK-phase load for this entity. Drone leg
        # timing still follows transport_power_profile_kW; only the load value is replaced.
        rv = rovers[k]
        prof = rv.get("work_power_profile_kW", None)
        if prof is None:
            return None
        prof = np.asarray(prof, dtype=float).reshape(-1)
        if prof.size == 0:
            return None
        interval_s = float(rv.get("work_sample_interval_s", dt_hours * 3600.0))
        t_now_s = float(t) * float(dt_hours) * 3600.0
        fidx = t_now_s / max(interval_s, 1e-9)
        i0 = int(np.floor(fidx))
        if i0 >= prof.size - 1:
            return float(prof[-1])
        frac = fidx - float(i0)
        return float(prof[i0] * (1.0 - frac) + prof[i0 + 1] * frac)

    def transport_load_kW(k, t, carrying):
        rv = rovers[k]
        if is_air_transport(k):
            profile = np.asarray(rv.get("transport_power_profile_kW", []), dtype=float)
            if len(profile) == 0:
                return float(rv.get("transport_avg_power_kW", P_idle_kW))
            idx = int(np.clip(transport_leg_step[k], 0, len(profile) - 1))
            return float(profile[idx])

        P_drive = float(rv.get("P_drive_cargo_kW", 1.85)) if carrying else float(rv.get("P_drive_kW", 1.25))
        P_aux = float(rv.get("P_aux_kW", 0.22))

        base = float(rv.get("P_handle_base_kW", 0.10))
        amp = float(rv.get("P_handle_amp_kW", 0.18))
        phase = float(rv.get("handle_phase", 0.0))

        low = base + amp * (0.6 + 0.4 * np.sin(2*np.pi*(t/32.0) + phase))
        high = 0.12 * (0.6 + 0.4 * np.sin(2*np.pi*(t/5.0) + phase*1.3))
        jitter = 0.14 * max(noise[t, k], -0.6)
        handle = max(low + high + jitter, 0.02)

        burst = (1.1 if carrying else 0.9) if op_timer[k] > 0 else 0.0
        return P_idle_kW + P_drive + P_aux + handle + burst

    def go_station_load_kW(k, moved_dist):
        if is_air_transport(k):
            rv = rovers[k]
            return float(rv.get("go_station_power_kW", rv.get("transport_avg_power_kW", P_idle_kW)))
        P_move = P_return_base_kW + k_return_kW_per_unit * float(moved_dist)
        return P_idle_kW + P_move

    def station_wait_load_kW():
        return P_idle_kW + P_charge_aux_kW

    def reset_transport_leg(k):
        transport_leg_origin[k] = pos[k].copy()
        if is_air_transport(k):
            profile = np.asarray(rovers[k].get("transport_power_profile_kW", []), dtype=float)
            transport_leg_total[k] = max(len(profile), 1)
        else:
            transport_leg_total[k] = 0
        transport_leg_step[k] = 0

    def move_transport_work(k, t):
        goal = np.asarray(transport_goal[k], dtype=float)
        carrying = (transport_stage[k] == "TO_DROPOFF")
        if not is_air_transport(k):
            moved = move_with_route(k, "TRANSPORT_WORK", goal, work_speed(k))
            load = transport_load_kW(k, t, carrying)
            return moved, load, arrived(pos[k], goal)

        rv = rovers[k]
        profile = np.asarray(rv.get("transport_power_profile_kW", []), dtype=float)
        total_steps = max(int(transport_leg_total[k]), 1)
        step_idx = int(np.clip(transport_leg_step[k], 0, total_steps - 1))
        load = float(profile[min(step_idx, len(profile) - 1)]) if len(profile) > 0 else float(rv.get("transport_avg_power_kW", P_idle_kW))

        prev = pos[k].copy()
        progress = float(step_idx + 1) / float(total_steps)
        pos[k] = transport_leg_origin[k] * (1.0 - progress) + goal * progress
        moved = float(np.linalg.norm(pos[k] - prev))
        transport_leg_step[k] = step_idx + 1

        if transport_leg_step[k] >= total_steps or arrived(pos[k], goal):
            pos[k] = goal.copy()
            return moved, load, True

        return moved, load, False

    def decide_charge_station(k, t):
        k_ref[0] = k
        pos_now = pos[k]
        st_sorted = sort_stations_by_distance(pos_now, stations)
        if len(st_sorted) == 0:
            return None, None

        # Air transports always prefer their home base station
        home_sid = rovers[k].get("home_station_id", None)
        if home_sid is not None and is_air_transport(k):
            home_st = next((s for s in stations if int(s["id"]) == int(home_sid)), None)
            if home_st is not None:
                st_sorted = [home_st] + [s for s in st_sorted if int(s["id"]) != int(home_sid)]

        rv = rovers[k]
        pv_peak = float(rovers[k].get("pv_peak_kW", 1.2))
        P_move_est = go_station_load_kW(k, 0.0)
        target_soc = float(rv.get("soc_resume", soc_resume_transport if rv["type"] == "transport" else soc_resume))

        def candidate_info(st):
            sid = int(st["id"])
            st_pos = station_target_point(pos_now, st)
            if is_air_transport(k):
                d = float(np.linalg.norm(st_pos - pos_now))
            else:
                d = planned_distance(pos_now, st_pos)
            if not np.isfinite(d):
                return None
            steps = int(np.ceil(d / max(station_speed(k), 1e-6)))
            arrival_step = int(t + steps)

            soc_arr = estimate_soc_after_distance(
                soc_now=soc[k],
                travel_dist=d,
                pv_peak_kW=pv_peak,
                t0=t,
                v_per_step=station_speed(k),
                P_load_move_kW=P_move_est,
                E_bat_kWh_cap=E_cap[k],
            )

            if (not is_air_transport(k)) and (soc_arr < (soc_floor[k] + safety_soc)):
                return None

            return {
                "sid": sid,
                "goal": st_pos,
                "distance": d,
                "arrival_step": arrival_step,
                "soc_arr": soc_arr,
            }

        def reserve(candidate, require_idle):
            sid = int(candidate["sid"])
            arrival_step = int(candidate["arrival_step"])

            ch_steps = estimate_charge_steps(
                soc_arrive_est=float(candidate["soc_arr"]),
                soc_target=target_soc,
                E_bat_kWh=E_cap[k],
                dt_hours=dt_hours,
                P_ch_max_kW=P_ch_max_arr[k],
                eta_ch_rover=eta_ch,
                eta_wireless=eta_wireless
            )

            ok = sm.request_charge(sid, k, t, arrival_step, ch_steps, require_idle_at_arrival=require_idle)
            return ok

        candidates = []
        for st in st_sorted:
            info = candidate_info(st)
            if info is not None:
                candidates.append(info)

        if len(candidates) == 0:
            return None, None

        candidates.sort(key=lambda item: item["distance"])

        for candidate in candidates:
            if reserve(candidate, require_idle=True):
                return int(candidate["sid"]), candidate["goal"]

        best = candidates[0]
        sm.request_charge(int(best["sid"]), k, t, int(best["arrival_step"]), 1, require_idle_at_arrival=False)
        return int(best["sid"]), best["goal"]

    for k, rv in enumerate(rovers):
        if rv["type"] == "transport":
            reset_transport_leg(k)

    for k, rv in enumerate(rovers):
        if rv["type"] != "explore" or not bool(rv.get("force_initial_charge", False)):
            continue
        resume_pos[k] = pos[k].copy()
        sid = None
        goal = None
        if rv.get("initial_charge_goal") is not None and rv.get("initial_charge_station_id") is not None:
            sid = int(rv.get("initial_charge_station_id"))
            goal = np.asarray(rv.get("initial_charge_goal"), dtype=float)
            travel_dist = float(np.linalg.norm(goal - pos[k]))
            arrival_step = int(np.ceil(travel_dist / max(station_speed(k), 1e-6)))
            target_soc = float(rv.get("soc_resume", soc_resume))
            soc_arr = estimate_soc_after_distance(
                soc_now=soc[k],
                travel_dist=travel_dist,
                pv_peak_kW=float(rv.get("pv_peak_kW", 0.0)),
                t0=0,
                v_per_step=station_speed(k),
                P_load_move_kW=go_station_load_kW(k, 0.0),
                E_bat_kWh_cap=E_cap[k],
            )
            ch_steps = estimate_charge_steps(
                soc_arrive_est=soc_arr,
                soc_target=target_soc,
                E_bat_kWh=E_cap[k],
                dt_hours=dt_hours,
                P_ch_max_kW=P_ch_max_arr[k],
                eta_ch_rover=eta_ch,
                eta_wireless=eta_wireless,
            )
            if not sm.request_charge(sid, k, 0, arrival_step, ch_steps, require_idle_at_arrival=False):
                sid = None
                goal = None

        if sid is None or goal is None:
            sid, goal = decide_charge_station(k, 0)
        if sid is not None and goal is not None:
            target_station[k] = int(sid)
            charge_goal[k] = np.asarray(goal, dtype=float)
            charge_goal_valid[k] = True
            mode[k] = "GO_STATION"
        elif enable_wait:
            mode[k] = "WAIT_PV"

    for t in range(N):
        for si, st in enumerate(stations):
            sid = int(st["id"])
            station_soc_hist[t, si] = sm.station_soc(sid)
            station_queue_hist[t, si] = sm.queue_length(sid)

        for k in range(K):
            if op_timer[k] > 0:
                op_timer[k] -= 1

        # End-game: send explore rovers back to a charging station so their trajectories
        # finish at a charger. Reuse the normal charge routing (reachable approach points)
        # to keep pathfinding cheap, and mark them so they stay parked once arrived.
        if return_trigger_step is not None and t == return_trigger_step:
            for k in range(K):
                if rovers[k]["type"] != "explore":
                    continue
                end_game[k] = True
                if mode[k] in ("GO_STATION", "WAIT_STATION", "CHARGE"):
                    continue
                sid, goal = decide_charge_station(k, t)
                if sid is not None and goal is not None:
                    resume_pos[k] = pos[k].copy()
                    target_station[k] = int(sid)
                    charge_goal[k] = np.asarray(goal, dtype=float)
                    charge_goal_valid[k] = True
                    clear_route(k)
                    mode[k] = "GO_STATION"

        P_load_tmp = np.zeros(K, dtype=float)
        P_pv_tmp = np.zeros(K, dtype=float)
        P_station_used_tmp = np.zeros(K, dtype=float)
        charge_station_id_tmp = np.full(K, -1, dtype=int)
        moved_tmp = np.zeros(K, dtype=float)

        # motion & mode
        for k in range(K):
            P_pv_tmp[k] = rover_pv_kW(k, t)

            if mode[k] == "RETURN_RESUME":
                goal = resume_pos[k]
                moved = move_with_route(k, "RETURN_RESUME", goal, station_speed(k))
                moved_tmp[k] = moved
                P_load_tmp[k] = go_station_load_kW(k, moved)
                if arrived(pos[k], goal):
                    clear_route(k)
                    mode[k] = "WORK"

            elif mode[k] == "DONE":
                # Drone has finished all round trips: land/park in place at idle.
                clear_route(k)
                P_load_tmp[k] = P_idle_kW

            elif mode[k] == "GO_STATION":
                sid = int(target_station[k])
                st_pos = None
                for st in stations:
                    if int(st["id"]) == sid:
                        st_pos = charge_goal[k].copy() if charge_goal_valid[k] else np.asarray(st["pos"], dtype=float)
                        break
                if st_pos is None:
                    clear_route(k)
                    mode[k] = "WORK"
                    charge_goal_valid[k] = False
                    if rovers[k]["type"] == "transport":
                        reset_transport_leg(k)
                else:
                    moved = move_with_route(k, "GO_STATION", st_pos, station_speed(k))
                    moved_tmp[k] = moved
                    P_load_tmp[k] = go_station_load_kW(k, moved)
                    if arrived(pos[k], st_pos):
                        clear_route(k)
                        sm.mark_arrived(sid, k)
                        mode[k] = "WAIT_STATION"
                        charge_goal_valid[k] = False
                        charge_step_cnt[k] = 0

            elif mode[k] in ["WAIT_STATION", "CHARGE"]:
                clear_route(k)
                P_load_tmp[k] = station_wait_load_kW()

            elif mode[k] == "WAIT_PV":
                clear_route(k)
                P_load_tmp[k] = P_idle_kW

            else:
                # WORK
                rv = rovers[k]
                if rv["type"] == "explore":
                    goal, _ = explore_next_point(k)
                    moved = move_with_route(k, "EXPLORE_WORK", goal, work_speed(k))
                    moved_tmp[k] = moved
                    _work_load = work_trace_load_kW(k, t)
                    P_load_tmp[k] = _work_load if _work_load is not None else explore_load_kW(k, t)
                    if arrived(pos[k], goal):
                        clear_route(k)
                        explore_advance(k)
                        explore_completion_count[k] += 1
                        op_timer[k] = 2
                else:
                    moved, load_kW, reached_goal = move_transport_work(k, t)
                    _work_load = work_trace_load_kW(k, t)
                    if _work_load is not None:
                        load_kW = _work_load
                    moved_tmp[k] = moved
                    P_load_tmp[k] = load_kW

                    if reached_goal:
                        clear_route(k)
                        op_timer[k] = 3
                        if transport_stage[k] == "TO_PICKUP":
                            transport_stage[k] = "TO_DROPOFF"
                            cp = _get_cargo_pair(rovers[k], transport_cargo_id[k])
                            transport_goal[k] = np.asarray(cp["dropoff"], dtype=float)
                            reset_transport_leg(k)
                        else:
                            transport_delivery_count[k] += 1
                            if delivery_cap is not None and transport_delivery_count[k] >= delivery_cap:
                                # Drone has completed all its round trips: stop and land.
                                transport_done[k] = True
                                clear_route(k)
                                mode[k] = "DONE"
                            else:
                                _choose_next_transport_job(k, rovers, pos, transport_stage, transport_goal,
                                                          transport_job_id, transport_cargo_id)
                                reset_transport_leg(k)

        # start charging if possible
        for st in stations:
            sm.start_charging_if_possible(int(st["id"]))

        # station power -> rover receive
        for st in stations:
            sid = int(st["id"])
            rid = sm.current[sid]

            if rid is None:
                sm.compute_station_power_once(sid, t, pv_env, P_tx_request_kW=0.0)
                continue

            P_rx_cap = min(P_ch_max_arr[rid], max((E_max[rid] - E[rid]) / max(dt_hours, 1e-9) / max(eta_ch, 1e-9), 0.0))
            P_tx_req = P_rx_cap / max(eta_wireless, 1e-9)

            P_tx_used, _ = sm.compute_station_power_once(sid, t, pv_env, P_tx_request_kW=P_tx_req)
            P_station_used_tmp[rid] = P_tx_used * eta_wireless
            if P_station_used_tmp[rid] > 1e-9:
                charge_station_id_tmp[rid] = sid

        # energy update & transitions
        for k in range(K):
            P_source = P_pv_tmp[k] + P_station_used_tmp[k]
            deficit = max(P_load_tmp[k] - P_source, 0.0)
            surplus = max(P_source - P_load_tmp[k], 0.0)

            P_dis = min(deficit, P_dis_max_arr[k])
            P_ch = min(surplus, P_ch_max_arr[k])

            if E[k] <= E_min[k] + 1e-12:
                P_dis = 0.0
            if E[k] >= E_max[k] - 1e-12:
                P_ch = 0.0

            dE = (eta_ch * P_ch - (P_dis / max(eta_dis, 1e-9))) * dt_hours
            if dE < 0 and (E[k] + dE < E_min[k]):
                allowable_dis_kWh = max(E[k] - E_min[k], 0.0)
                allowable_dis_kW = allowable_dis_kWh / max(dt_hours, 1e-9) * eta_dis
                P_dis = min(P_dis, allowable_dis_kW)
                dE = (eta_ch * P_ch - (P_dis / max(eta_dis, 1e-9))) * dt_hours

            E[k] = float(np.clip(E[k] + dE, E_min[k], E_max[k]))
            soc[k] = E[k] / max(E_cap[k], 1e-9)

            P_bat = P_dis - P_ch
            unmet = max(P_load_tmp[k] - (P_source + P_dis), 0.0)

            if mode[k] == "WAIT_STATION":
                sid = int(target_station[k])
                if sid >= 0 and sm.current[sid] == k:
                    mode[k] = "CHARGE"
                    charge_step_cnt[k] = 0

            if mode[k] == "CHARGE":
                charge_step_cnt[k] += 1

                # NEW: transport stricter leave condition to avoid micro-charging pulses
                if rovers[k]["type"] == "transport":
                    resume_target = float(rovers[k].get("soc_resume_transport", rovers[k].get("soc_resume", soc_resume_transport)))
                    can_leave = (soc[k] >= resume_target - 1e-12) or (E[k] >= E_max[k] - 1e-9)
                    if (charge_step_cnt[k] >= int(min_charge_steps_transport)) and can_leave:
                        sid = int(target_station[k])
                        sm.release(sid, k)
                        target_station[k] = -1
                        charge_step_cnt[k] = 0
                        clear_route(k)
                        if is_air_transport(k):
                            reset_transport_leg(k)
                        mode[k] = "WORK"
                else:
                    # Explore rover: HARD departure threshold — must reach soc_resume before leaving.
                    # No early-exit shortcut; prevents micro-charging chatter.
                    resume_target = float(rovers[k].get("soc_resume", soc_resume))
                    can_leave = (soc[k] >= resume_target - 1e-12) or (E[k] >= E_max[k] - 1e-9)

                    # End-game: keep the rover parked at the charging station to the horizon end.
                    if end_game[k]:
                        can_leave = False

                    if (charge_step_cnt[k] >= int(min_charge_steps)) and can_leave:
                        sid = int(target_station[k])
                        sm.release(sid, k)
                        target_station[k] = -1
                        charge_step_cnt[k] = 0
                        clear_route(k)
                        mode[k] = "RETURN_RESUME"

            if mode[k] == "WAIT_PV":
                stop_target = float(rovers[k].get("soc_stop", soc_stop))
                if soc[k] > stop_target + 1e-12 or P_pv_tmp[k] >= P_load_tmp[k]:
                    mode[k] = "WORK"
                    if is_air_transport(k):
                        reset_transport_leg(k)

            if mode[k] == "WORK":
                pv_peak = float(rovers[k].get("pv_peak_kW", 1.2))
                stop_target = float(rovers[k].get("soc_stop", soc_stop))

                if rovers[k]["type"] == "explore":
                    goal, _ = explore_next_point(k)
                    v_move = work_speed(k)
                    P_move_est = explore_load_kW(k, t)
                    soc_after = estimate_soc_after_travel(
                        soc_now=soc[k],
                        pos_now=pos[k],
                        pos_goal=goal,
                        pv_env=pv_env[:N],
                        pv_peak_kW=pv_peak,
                        t0=t,
                        dt_hours=dt_hours,
                        v_per_step=v_move,
                        P_load_move_kW=P_move_est,
                        E_bat_kWh=E_cap[k],
                        eta_dis=eta_dis
                    )
                    need_charge = (soc[k] <= stop_target + 1e-12) or (soc_after < (soc_floor[k] + safety_soc))
                else:
                    if is_air_transport(k):
                        leg_in_progress = 0 < transport_leg_step[k] < transport_leg_total[k]
                        profile = np.asarray(rovers[k].get("transport_power_profile_kW", []), dtype=float)
                        P_move_est = float(np.mean(profile)) if len(profile) > 0 else float(rovers[k].get("transport_avg_power_kW", P_idle_kW))
                        soc_after = estimate_soc_after_profile(
                            soc_now=soc[k],
                            profile_kW=profile,
                            start_step=transport_leg_step[k],
                            pv_peak_kW=pv_peak,
                            t0=t,
                            E_bat_kWh_cap=E_cap[k],
                        )
                        need_charge = (not leg_in_progress) and ((soc[k] <= stop_target + 1e-12) or (soc_after < (soc_floor[k] + safety_soc)))
                    else:
                        goal = transport_goal[k]
                        v_move = work_speed(k)
                        carrying = (transport_stage[k] == "TO_DROPOFF")
                        P_move_est = transport_load_kW(k, t, carrying)
                        travel_dist = planned_distance(pos[k], goal)
                        soc_after = estimate_soc_after_distance(
                            soc_now=soc[k],
                            travel_dist=travel_dist,
                            pv_peak_kW=pv_peak,
                            t0=t,
                            v_per_step=v_move,
                            P_load_move_kW=P_move_est,
                            E_bat_kWh_cap=E_cap[k],
                        )
                        need_charge = (soc[k] <= stop_target + 1e-12) or (soc_after < (soc_floor[k] + safety_soc))

                if need_charge:
                    if rovers[k]["type"] == "explore":
                        resume_pos[k] = pos[k].copy()
                    sid, goal = decide_charge_station(k, t)
                    if sid is not None and goal is not None:
                        target_station[k] = int(sid)
                        charge_goal[k] = np.asarray(goal, dtype=float)
                        charge_goal_valid[k] = True
                        mode[k] = "GO_STATION"
                    else:
                        if enable_wait:
                            mode[k] = "WAIT_PV"

            if enable_wait and (unmet > 1e-9) and (mode[k] not in ["CHARGE", "WAIT_STATION", "DONE"]):
                mode[k] = "WAIT_PV"

            soc_hist[t, k] = soc[k]
            x_hist[t, k] = pos[k, 0]
            y_hist[t, k] = pos[k, 1]
            mode_hist[t, k] = mode[k]

            P_load_hist[t, k] = P_load_tmp[k]
            P_pv_hist[t, k] = P_pv_tmp[k]
            P_station_used_hist[t, k] = P_station_used_tmp[k]
            P_bat_hist[t, k] = P_bat
            unmet_hist[t, k] = unmet
            charge_station_id_hist[t, k] = int(charge_station_id_tmp[k])

            if rovers[k]["type"] == "transport":
                transport_job_hist[t, k] = int(transport_job_id[k])
                transport_cargo_hist[t, k] = int(transport_cargo_id[k])

    explore_idx = [i for i, rv in enumerate(rovers) if rv["type"] == "explore"]
    transport_idx = [i for i, rv in enumerate(rovers) if rv["type"] == "transport"]

    return {
        "rovers": rovers,
        "stations": stations,
        "soc": soc_hist,
        "x": x_hist,
        "y": y_hist,
        "mode": mode_hist,
        "P_load": P_load_hist,
        "P_pv": P_pv_hist,
        "P_station_used": P_station_used_hist,
        "charge_station_id": charge_station_id_hist,
        "P_bat": P_bat_hist,
        "unmet": unmet_hist,
        "station_soc": station_soc_hist,
        "station_queue": station_queue_hist,
        "explore_idx": explore_idx,
        "transport_idx": transport_idx,
        "explore_completion_count": explore_completion_count,
        "transport_delivery_count": transport_delivery_count,
        "transport_job_hist": transport_job_hist,
        "transport_cargo_hist": transport_cargo_hist,
    }


def _get_cargo_pair(rv, cargo_id):
    cps = rv["cargo_pairs"]
    for cp in cps:
        if int(cp["cargo_id"]) == int(cargo_id):
            return cp
    return cps[0]


def _init_transport_first_job(k, rovers, transport_stage, transport_goal, transport_job_id, transport_cargo_id):
    rv = rovers[k]
    if rv["type"] != "transport":
        return
    cps = rv["cargo_pairs"]
    if str(rv.get("route_mode", "")).lower() == "sequence" and len(cps) > 0:
        seq_idx = int(rv.get("route_pair_idx", 0)) % len(cps)
        cp = cps[seq_idx]
        rv["route_pair_idx"] = seq_idx
        transport_job_id[k] = 0
        transport_cargo_id[k] = int(cp["cargo_id"])
        transport_stage[k] = "TO_PICKUP"
        transport_goal[k] = np.asarray(cp["pickup"], dtype=float)
        return
    cid = int(rv.get("first_cargo_id", cps[0]["cargo_id"]))
    cp = None
    for x in cps:
        if int(x["cargo_id"]) == cid:
            cp = x
            break
    if cp is None:
        cp = cps[0]
    transport_job_id[k] = 0
    transport_cargo_id[k] = int(cp["cargo_id"])
    transport_stage[k] = "TO_PICKUP"
    transport_goal[k] = np.asarray(cp["pickup"], dtype=float)


def _choose_next_transport_job(k, rovers, pos, transport_stage, transport_goal, transport_job_id, transport_cargo_id):
    rv = rovers[k]
    cps = rv["cargo_pairs"]
    if str(rv.get("route_mode", "")).lower() == "sequence" and len(cps) > 0:
        next_idx = (int(rv.get("route_pair_idx", 0)) + 1) % len(cps)
        rv["route_pair_idx"] = next_idx
        best = cps[next_idx]
        transport_job_id[k] = int(transport_job_id[k]) + 1
        transport_cargo_id[k] = int(best["cargo_id"])
        transport_stage[k] = "TO_PICKUP"
        transport_goal[k] = np.asarray(best["pickup"], dtype=float)
        return
    pnow = pos[k]

    best = None
    best_d = 1e18
    for cp in cps:
        p = np.asarray(cp["pickup"], dtype=float)
        d = float(np.linalg.norm(p - pnow))
        if d < best_d:
            best_d = d
            best = cp
    if best is None:
        best = cps[0]

    transport_job_id[k] = int(transport_job_id[k]) + 1
    transport_cargo_id[k] = int(best["cargo_id"])
    transport_stage[k] = "TO_PICKUP"
    transport_goal[k] = np.asarray(best["pickup"], dtype=float)
