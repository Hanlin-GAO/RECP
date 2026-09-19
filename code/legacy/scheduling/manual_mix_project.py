
raise RuntimeError('Inspection-only historical code. Execution is disabled; use scripts/reproduce.py train with the reviewed protocol.')

import csv
import os

import numpy as np

from plotting import plot_scene_boundary, plot_traj
from scenarios import build_manual_mixed_scenario
from simulator import simulate_mixed_system


def _write_positions_csv(base_dir, step_index, dt_hours, rovers, x_mat, y_mat):
    csv_path = os.path.join(base_dir, "mix_rover_positions.positions.csv")
    meta_path = os.path.join(base_dir, "mix_rover_positions.meta.csv")

    header = ["step", "time_hours"]
    for rv in rovers:
        header.extend([f"{rv['name']}_x", f"{rv['name']}_y"])

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, step in enumerate(step_index):
            row = [int(step), float(step) * float(dt_hours)]
            for rover_idx in range(len(rovers)):
                row.extend([float(x_mat[idx, rover_idx]), float(y_mat[idx, rover_idx])])
            writer.writerow(row)

    with open(meta_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["origin_lower_left_zero", False])
        writer.writerow(["x_offset_added", 0.0])
        writer.writerow(["y_offset_added", 0.0])
        writer.writerow(["note", "x_export = x_raw + x_offset_added; y_export = y_raw + y_offset_added"])

    return csv_path, meta_path


def _polyline_length(points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    diffs = pts[1:] - pts[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _estimate_manual_horizon_steps(stations, explore_rovers, transport_rovers, sample_interval_s, *,
                                   round_trip_count, v_explore_per_step, eta_wireless=0.92, eta_ch=0.95):
    station_tx_max = max(float(st.get("P_tx_max_kW", 0.0)) for st in stations)
    station_rx_max = station_tx_max * float(eta_wireless)
    seconds_per_step = max(float(sample_interval_s), 1e-9)

    max_initial_charge_steps = 0
    for rv in list(explore_rovers) + list(transport_rovers):
        if rv in transport_rovers:
            resume_soc = float(rv.get("soc_resume_transport", 0.60))
        else:
            resume_soc = float(rv.get("soc_resume", 0.40))

        soc0 = float(rv.get("soc0", resume_soc))
        if soc0 >= resume_soc:
            continue

        e_cap = float(rv.get("E_bat_kWh", 0.0))
        e_need = max(resume_soc - soc0, 0.0) * e_cap
        rover_rx_cap = min(float(rv.get("P_ch_max_kW", 0.0)), station_rx_max)
        e_gain_kW = float(eta_ch) * max(rover_rx_cap, 1e-9)
        charge_seconds = (e_need / e_gain_kW) * 3600.0
        max_initial_charge_steps = max(max_initial_charge_steps, int(np.ceil(charge_seconds / seconds_per_step)))

    max_explore_route_steps = 0
    for rv in explore_rovers:
        path_points = rv.get("path_points", [])
        path_len = _polyline_length(path_points)
        loop_factor = 1.0 if bool(rv.get("loop", True)) else 0.6
        route_steps = int(np.ceil((path_len * loop_factor) / max(float(v_explore_per_step), 1e-9)))
        max_explore_route_steps = max(max_explore_route_steps, route_steps)

    max_transport_route_steps = 0
    for rv in transport_rovers:
        leg_steps = max(len(rv.get("transport_power_profile_kW", [])), 1)
        seg_count = max(len(rv.get("cargo_pairs", [])), 1)
        max_transport_route_steps = max(max_transport_route_steps, round_trip_count * max(2 * seg_count, 2) * leg_steps)

    visibility_buffer_steps = int(np.ceil(180.0 / seconds_per_step))
    return max(
        720,
        max_transport_route_steps,
        max_initial_charge_steps + max_explore_route_steps + visibility_buffer_steps,
    )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    round_trip_count = 1

    stations, explore_rovers, transport_rovers, scene = build_manual_mixed_scenario()
    print("built_scene", flush=True)
    drone_profiles = [rv for rv in transport_rovers if str(rv.get("mobility", "ground")).lower() == "air"]
    if len(drone_profiles) == 0:
        raise ValueError("Manual route mode requires at least one transport drone")

    sample_interval_s = float(drone_profiles[0].get("transport_sample_interval_s", 0.1))
    dt_hours = sample_interval_s / 3600.0
    n_steps = _estimate_manual_horizon_steps(
        stations,
        explore_rovers,
        transport_rovers,
        sample_interval_s,
        round_trip_count=round_trip_count,
        v_explore_per_step=0.10,
    )
    print(f"before_sim n_steps={n_steps} dt_hours={dt_hours}", flush=True)

    pv_env = np.ones(int(n_steps), dtype=float)
    min_charge_steps = max(1, int(round(18.0 / max(dt_hours * 3600.0, 1e-9))))
    min_charge_steps_transport = max(1, int(round(28.0 / max(dt_hours * 3600.0, 1e-9))))

    res = simulate_mixed_system(
        pv_env=pv_env,
        stations=stations,
        explore_rovers=explore_rovers,
        transport_rovers=transport_rovers,
        scene=scene,
        N_steps=n_steps,
        dt_hours=dt_hours,
        v_transport_per_step=0.08,
        v_explore_per_step=0.10,
        v_to_station_per_step=0.12,
        arrival_eps=0.08,
        P_idle_kW=0.02,
        P_charge_aux_kW=0.01,
        enable_wait=True,
        soc_stop=0.28,
        soc_resume=0.40,
        soc_resume_transport=0.60,
        min_charge_steps=min_charge_steps,
        min_charge_steps_transport=min_charge_steps_transport,
        extra_leave_margin=0.04,
    )
    print("sim_done", flush=True)

    rovers = res["rovers"]
    explore_idx = res["explore_idx"]
    transport_idx = res["transport_idx"]

    cargo_pairs = []
    for rv in transport_rovers:
        for cp in rv.get("cargo_pairs", []):
            cargo_pairs.append(cp)

    out_traj_explore = os.path.join(base_dir, "mix_traj_explore.png")
    out_traj_transport = os.path.join(base_dir, "mix_traj_transport.png")
    out_traj_mixed = os.path.join(base_dir, "mix_traj_mixed.png")
    scene_boundary_dir = os.path.dirname(os.path.abspath(scene.get("source_file", base_dir))) if scene is not None else base_dir
    out_scene_boundary = os.path.join(scene_boundary_dir, "地形边界.jpg")

    plot_traj(
        out_traj_explore,
        res["x"][:, explore_idx],
        res["y"][:, explore_idx],
        [rovers[i]["name"] for i in explore_idx],
        stations,
        rover_start_pos=[explore_rovers[i]["start_pos"] for i in range(len(explore_rovers))],
        cargo_pairs=None,
        scene=scene,
        title="Explore Trajectories (Manual Routes)",
    )
    plot_traj(
        out_traj_transport,
        res["x"][:, transport_idx],
        res["y"][:, transport_idx],
        [rovers[i]["name"] for i in transport_idx],
        stations,
        rover_start_pos=[transport_rovers[i]["start_pos"] for i in range(len(transport_rovers))],
        cargo_pairs=cargo_pairs,
        scene=scene,
        title="Transport Drone Trajectories (Manual Routes)",
    )
    plot_traj(
        out_traj_mixed,
        res["x"],
        res["y"],
        [rv["name"] for rv in rovers],
        stations,
        rover_start_pos=[rv.get("start_pos", (0.0, 0.0)) for rv in rovers],
        cargo_pairs=cargo_pairs,
        scene=scene,
        title="Mixed Trajectories (Manual Routes)",
    )
    plot_scene_boundary(
        out_scene_boundary,
        stations,
        scene=scene,
        title=None,
    )

    step_index = np.arange(res["x"].shape[0])
    csv_path, meta_path = _write_positions_csv(base_dir, step_index, dt_hours, rovers, res["x"], res["y"])
    print("csv_done", flush=True)

    print("manual_mix_done")
    print(out_traj_explore)
    print(out_traj_transport)
    print(out_traj_mixed)
    print(out_scene_boundary)
    print(csv_path)
    print(meta_path)


if __name__ == "__main__":
    main()
