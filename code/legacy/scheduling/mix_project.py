
raise RuntimeError('Inspection-only historical code. Execution is disabled; use scripts/reproduce.py train with the reviewed protocol.')

import os
import numpy as np
import matplotlib.pyplot as plt

from io_utils import load_vstry_scaled_power_profile, read_power_series_from_excel, write_rover_positions_to_excel
from scenarios import build_default_mixed_scenario, build_real_mixed_scenario
from simulator import simulate_mixed_system
from plotting import (
    plot_soc,
    plot_power_breakdown_by_station,
    plot_traj,
    plot_station_soc_and_queue,
    plot_transport_tasks_colored,
)


def _find_pv_file(base_dir):
    cand1 = os.path.join(base_dir, "Solar station site 1.xlsx")
    cand2 = os.path.join(os.path.dirname(base_dir), "Solar station site 1.xlsx")
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    raise FileNotFoundError("Solar station site 1.xlsx not found in mix/ or its parent folder.")


def _build_fallback_pv_env(n_steps):
    t = np.linspace(0.0, 1.0, int(n_steps), endpoint=False)
    daylight = np.sin(np.pi * np.clip((t - 0.22) / 0.58, 0.0, 1.0))
    shoulder = 0.10 * np.sin(6.0 * np.pi * t + 0.4)
    pv_env = np.clip(daylight ** 1.7 + shoulder, 0.0, 1.0)
    return pv_env


def _repo_root(base_dir):
    return os.path.dirname(os.path.dirname(base_dir))


def _vstry_dir(base_dir):
    return os.path.join(_repo_root(base_dir), "VStry")


def _build_station_prediction_profiles(base_dir, stations, n_steps, dt_hours):
    vstry_dir = _vstry_dir(base_dir)
    start_times = {"pv": "2023-06-01 09:00:00", "wind": "2020-06-01 09:00:00"}

    # VSgo station ids are 0/1/2, corresponding to user-facing charging stations 1/2/3.
    station_sources = {
        0: {"domain": "pv", "source_name": "Inverter_1", "model_tag": "pinn"},
        1: {"domain": "wind", "source_name": "Turbine_2", "model_tag": "pinn"},
        2: {"domain": "pv", "source_name": "Inverter_3", "model_tag": "pinn"},
    }

    raw_profiles = {}
    for sid, spec in station_sources.items():
        raw_profiles[sid] = load_vstry_scaled_power_profile(
            vstry_dir=vstry_dir,
            domain=spec["domain"],
            source_name=spec["source_name"],
            model_tag=spec["model_tag"],
            start_time=start_times[spec["domain"]],
            n_steps=n_steps,
            dt_hours=dt_hours,
            scale_ratio=1.0,
        )

    station_by_id = {int(st["id"]): st for st in stations}
    pv_scale_ref_sid = 0 if 0 in station_by_id else 2
    pv_scale = float(station_by_id[pv_scale_ref_sid].get("pv_peak_kW", 0.20)) / max(float(np.max(raw_profiles[pv_scale_ref_sid])), 1e-9)
    wind_scale = float(station_by_id[1].get("pv_peak_kW", 0.20)) / max(float(np.max(raw_profiles[1])), 1e-9)

    scales = {0: pv_scale, 1: wind_scale, 2: pv_scale}
    labels = {"pv": "PV forecast", "wind": "Wind forecast"}
    for sid, spec in station_sources.items():
        st = station_by_id[sid]
        st["renewable_profile_kW"] = raw_profiles[sid] * scales[sid]
        st["renewable_label"] = labels[spec["domain"]]
        st["renewable_scale_ratio"] = scales[sid]
        st["renewable_source_name"] = spec["source_name"]
        st["renewable_start_time"] = start_times[spec["domain"]]

    return stations


def plot_station_power_io(out_path, t, pv_env, stations, charge_station_id, P_station_used, x_label="Time"):
    pv_env = np.asarray(pv_env, dtype=float).reshape(-1)
    charge_station_id = np.asarray(charge_station_id, dtype=int)   # (N,K)
    P_station_used = np.asarray(P_station_used, dtype=float)       # (N,K)

    N = int(len(t))
    pv_env = pv_env[:N]

    S = len(stations)
    station_ids = [int(st["id"]) for st in stations]

    P_in = np.zeros((N, S), dtype=float)
    P_out = np.zeros((N, S), dtype=float)
    P_aux = np.zeros(S, dtype=float)

    for i, st in enumerate(stations):
        sid = int(st["id"])
        P_aux[i] = float(st.get("P_aux_kW", 0.40))
        renewable_profile = st.get("renewable_profile_kW", None)
        if renewable_profile is not None:
            renewable_profile = np.asarray(renewable_profile, dtype=float).reshape(-1)
            P_in[:, i] = renewable_profile[:N]
        else:
            pv_peak = float(st.get("pv_peak_kW", 50.0))
            P_in[:, i] = pv_env * pv_peak
        mask = (charge_station_id == sid).astype(float)
        P_out[:, i] = np.sum(P_station_used * mask, axis=1)

    fig, axes = plt.subplots(S, 1, figsize=(10.8, 2.6 * S), sharex=True)
    if S == 1:
        axes = [axes]

    for i, sid in enumerate(station_ids):
        ax = axes[i]
        input_label = str(stations[i].get("renewable_label", "PV input"))
        ax.plot(t, P_in[:, i], label=f"{input_label} (kW)")
        ax.plot(t, P_out[:, i], label="Output to rovers (kW)")
        ax.axhline(P_aux[i], linestyle="--", linewidth=1.2, label="Station aux load (kW)")
        ax.set_ylabel("kW")
        source_name = stations[i].get("renewable_source_name", None)
        start_time = stations[i].get("renewable_start_time", None)
        title = f"Station S{sid}"
        if source_name and start_time:
            title += f" | {source_name} @ {start_time}"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

    axes[-1].set_xlabel(x_label)
    fig.suptitle("Charging Station Input/Output Power")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scene = None
    round_trip_count = 10
    dt_hours = 15 / 60
    N_steps = 288
    try:
        stations, explore_rovers, transport_rovers, scene = build_real_mixed_scenario()
        drone_profiles = [rv for rv in transport_rovers if str(rv.get("mobility", "ground")).lower() == "air"]
        if len(drone_profiles) > 0:
            sample_interval_s = float(drone_profiles[0].get("transport_sample_interval_s", 0.1))
            leg_steps = max(len(drone_profiles[0].get("transport_power_profile_kW", [])), 1)
            dt_hours = sample_interval_s / 3600.0
            N_steps = 2 * round_trip_count * leg_steps
    except (FileNotFoundError, ValueError):
        stations, explore_rovers, transport_rovers = build_default_mixed_scenario()
        scene = None

    stations = _build_station_prediction_profiles(base_dir, stations, N_steps, dt_hours)
    pv_env = np.ones(int(N_steps), dtype=float)

    min_charge_steps = max(1, int(round(18.0 / max(dt_hours * 3600.0, 1e-9))))
    min_charge_steps_transport = max(1, int(round(28.0 / max(dt_hours * 3600.0, 1e-9))))

    res = simulate_mixed_system(
        pv_env=pv_env,
        stations=stations,
        explore_rovers=explore_rovers,
        transport_rovers=transport_rovers,
        scene=scene,
        N_steps=N_steps,
        dt_hours=dt_hours,

        # ground-vehicle fallback speeds; air drones use per-vehicle profile timing.
        v_transport_per_step=0.08,
        v_explore_per_step=0.10,
        v_to_station_per_step=0.12,
        arrival_eps=0.08,

        # Small rover / drone scale: P_idle and P_charge_aux must be < station TX power
        # so the wireless charger can actually raise rover SoC (not just power its idle draw).
        P_idle_kW=0.02,
        P_charge_aux_kW=0.01,

        enable_wait=True,
        soc_stop=0.28,          # trigger charging when SoC falls to 28%
        soc_resume=0.40,         # must reach 40% before leaving (11 min charge, ~12% hysteresis)
        soc_resume_transport=0.60,  # drones must reach 60% before resuming flight (~14 min charge)
        min_charge_steps=min_charge_steps,
        min_charge_steps_transport=min_charge_steps_transport,
        extra_leave_margin=0.04
    )

    step_index = np.arange(res["soc"].shape[0])
    time_seconds = step_index * dt_hours * 3600.0
    if len(time_seconds) > 0 and float(time_seconds[-1]) >= 600.0:
        t = time_seconds / 60.0
        time_label = "Time (min)"
    else:
        t = time_seconds
        time_label = "Time (s)"

    rovers = res["rovers"]
    explore_idx = res["explore_idx"]
    transport_idx = res["transport_idx"]

    def pick(mat, idx):
        return mat[:, idx]

    explore_names = [rovers[i]["name"] for i in explore_idx]
    transport_names = [rovers[i]["name"] for i in transport_idx]

    out_soc_explore = os.path.join(base_dir, "mix_soc_explore.png")
    out_soc_transport = os.path.join(base_dir, "mix_soc_transport.png")
    out_pow_explore = os.path.join(base_dir, "mix_power_explore.png")
    out_pow_transport = os.path.join(base_dir, "mix_power_transport.png")

    out_traj_explore = os.path.join(base_dir, "mix_traj_explore.png")
    out_traj_transport = os.path.join(base_dir, "mix_traj_transport.png")
    out_traj_mixed = os.path.join(base_dir, "mix_traj_mixed.png")

    out_station_soc = os.path.join(base_dir, "mix_station_soc.png")
    out_station_queue = os.path.join(base_dir, "mix_station_queue.png")
    out_station_power_io = os.path.join(base_dir, "mix_station_power_io.png")

    cargo_pairs = []
    seen_cargo = set()
    for rv in transport_rovers:
        for cp in rv.get("cargo_pairs", []):
            cid = int(cp["cargo_id"])
            if cid in seen_cargo:
                continue
            seen_cargo.add(cid)
            cargo_pairs.append(cp)
    cargo_pairs.sort(key=lambda item: int(item["cargo_id"]))

    transport_plot_name = transport_rovers[0]["name"] if len(transport_rovers) > 0 else "transport"
    out_transport_tasks = os.path.join(base_dir, f"transport_tasks_colored_{transport_plot_name}.png")

    # ✅ Excel export (OVERWRITE every run)
    out_positions_excel = os.path.join(base_dir, "mix_rover_positions.xlsx")

    transport_group_label = "Transport Drones" if any(str(rovers[i].get("mobility", "ground")).lower() == "air" for i in transport_idx) else "Transport Rovers"

    plot_soc(out_soc_explore, t, pick(res["soc"], explore_idx), explore_names, title="Explore Rovers SOC", x_label=time_label)
    plot_soc(out_soc_transport, t, pick(res["soc"], transport_idx), transport_names, title=f"{transport_group_label} SOC", x_label=time_label)

    plot_power_breakdown_by_station(
        out_pow_explore, t,
        pick(res["P_load"], explore_idx),
        pick(res["P_pv"], explore_idx),
        pick(res["P_station_used"], explore_idx),
        pick(res["charge_station_id"], explore_idx),
        explore_names, stations,
        title="Explore Power Breakdown (Charging colored by station)",
        x_label=time_label,
    )
    plot_power_breakdown_by_station(
        out_pow_transport, t,
        pick(res["P_load"], transport_idx),
        pick(res["P_pv"], transport_idx),
        pick(res["P_station_used"], transport_idx),
        pick(res["charge_station_id"], transport_idx),
        transport_names, stations,
        title=f"{transport_group_label} Power Breakdown (Charging colored by station)",
        x_label=time_label,
    )

    explore_starts = [explore_rovers[i]["start_pos"] for i in range(len(explore_rovers))]
    scene_title = "Real Terrain" if scene is not None else "105m × 68m field"

    plot_traj(out_traj_explore,
              pick(res["x"], explore_idx), pick(res["y"], explore_idx),
              explore_names, stations,
              rover_start_pos=explore_starts,
              cargo_pairs=None,
              scene=scene,
              title=f"Explore Trajectories ({scene_title})")

    transport_starts = [transport_rovers[i]["start_pos"] for i in range(len(transport_rovers))]
    plot_traj(out_traj_transport,
              pick(res["x"], transport_idx), pick(res["y"], transport_idx),
              transport_names, stations,
              rover_start_pos=transport_starts,
              cargo_pairs=cargo_pairs,
              scene=scene,
              title=f"{transport_group_label} Trajectories ({scene_title})")

    all_starts = [rv.get("start_pos", (0.0, 0.0)) for rv in rovers]
    plot_traj(out_traj_mixed,
              res["x"], res["y"],
              [rv["name"] for rv in rovers], stations,
              rover_start_pos=all_starts,
              cargo_pairs=cargo_pairs,
              scene=scene,
              title=f"Mixed Trajectories ({scene_title})")

    plot_station_soc_and_queue(out_station_soc, out_station_queue, t,
                               res["station_soc"], res["station_queue"], stations, x_label=time_label)

    plot_station_power_io(
        out_station_power_io, t,
        pv_env=pv_env,
        stations=stations,
        charge_station_id=res["charge_station_id"],
        P_station_used=res["P_station_used"],
        x_label=time_label,
    )

    # Export raw simulation coordinates without additional offset so downstream checks match the plots.
    write_rover_positions_to_excel(
        excel_path=out_positions_excel,
        t=step_index,
        rover_names=[rv["name"] for rv in rovers],
        x_mat=res["x"],
        y_mat=res["y"],
        dt_hours=dt_hours,
        origin_lower_left_zero=False,
    )

    idx_A = transport_idx[0]

    plot_transport_tasks_colored(
        out_transport_tasks,
        x=res["x"][:, idx_A],
        y=res["y"][:, idx_A],
        job_id=res["transport_job_hist"][:, idx_A],
        cargo_id=res["transport_cargo_hist"][:, idx_A],
        cargo_pairs=cargo_pairs,
        stations=stations,
        scene=scene,
        rover_name=rovers[idx_A]["name"]
    )

    print(f"\n===== Mixed Simulation Finished ({scene_title}) =====")
    print("Saved figures:")
    for p in [out_soc_explore, out_soc_transport, out_pow_explore, out_pow_transport,
              out_traj_explore, out_traj_transport, out_traj_mixed,
              out_station_soc, out_station_queue, out_station_power_io,
              out_transport_tasks]:
        print(" -", p)

    print("Saved logs:")
    print(" -", out_positions_excel)


if __name__ == "__main__":
    main()
