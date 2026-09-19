import argparse
import copy
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from io_utils import load_vstry_power_frame, resample_power_frame, write_rover_positions_to_excel
from plotting import (
    plot_power_breakdown_by_station,
    plot_scene_boundary,
    plot_soc,
    plot_station_soc_and_queue,
    plot_traj,
    plot_transport_tasks_colored,
)
from scenarios import build_manual_mixed_scenario
from simulator import simulate_mixed_system
from scheduling_paths import ROOT, VEHICLE_POWER


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "runs/scheduling/manual_first_frame"
ROUND_TRIP_COUNT = 10
LATEX_EOL = r"\\"

SOURCE_MAP = {
    0: {"domain": "pv", "source_name": "Inverter_1", "model_tag": "pinn"},
    1: {"domain": "wind", "source_name": "Turbine_2", "model_tag": "pinn"},
    2: {"domain": "pv", "source_name": "Inverter_3", "model_tag": "pinn"},
}
PERIOD_STARTS = {
    "AM": {"pv": "2023-06-01 07:30:00", "wind": "2020-06-01 07:30:00"},
    "NN": {"pv": "2023-06-01 11:00:00", "wind": "2020-06-01 11:00:00"},
    "PM": {"pv": "2023-06-01 14:00:00", "wind": "2020-06-01 14:00:00"},
}
# Init. SoC levels: H=High, M=Medium, L=Low, X=Mixed (per period: H, M, L, X)
GROUP_SEQUENCE = [
    ("G1", "AM", "H"),
    ("G2", "AM", "M"),
    ("G3", "AM", "L"),
    ("G4", "AM", "X"),
    ("G5", "NN", "H"),
    ("G6", "NN", "M"),
    ("G7", "NN", "L"),
    ("G8", "NN", "X"),
    ("G9", "PM", "H"),
    ("G10", "PM", "M"),
    ("G11", "PM", "L"),
    ("G12", "PM", "X"),
]
CASE_SEQUENCE = [
    {"folder": "actual_generation", "renewable_kind": "actual"},
    {"folder": "predicted_generation", "renewable_kind": "predicted"},
]

REAL_STAGGERED = {
    "rovers": [0.28, 0.36, 0.44],
    "drones": [0.68, 0.50, 0.36],
    "stations": [0.60, 0.55, 0.65],
}
INIT_TEMPLATES = {
    # H = High: medium-high uniform initial SoC across all entities
    "H": {
        "rovers": [0.70, 0.70, 0.70],
        "drones": [0.70, 0.70, 0.70],
        "stations": [0.80, 0.80, 0.80],
    },
    # M = Medium: staggered initial SoC
    "M": REAL_STAGGERED,
    # L = Low: uniformly low initial SoC
    "L": {
        "rovers": [0.28, 0.28, 0.28],
        "drones": [0.30, 0.30, 0.30],
        "stations": [0.40, 0.40, 0.40],
    },
    # X = Mixed: one entity each at high / medium / low
    "X": {
        "rovers": [0.70, REAL_STAGGERED["rovers"][1], 0.28],
        "drones": [0.70, REAL_STAGGERED["drones"][1], 0.30],
        "stations": [0.80, REAL_STAGGERED["stations"][1], 0.40],
    },
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run paired actual-vs-predicted renewable-generation validation on the "
            "manual first-frame mixed scene."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Root directory for the 12 group folders and summary files.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional subset of group ids to run, for example: --groups G1 G2.",
    )
    parser.add_argument(
        "--vehicle-output-mode",
        default="actual",
        choices=["actual"],
        help=(
            "Reserved interface for future rover/drone predicted-output-power studies. "
            "Only 'actual' is implemented in this batch."
        ),
    )
    parser.add_argument('--comparison-protocol',choices=['fixed_load','historical_paired'],default='fixed_load',
                        help='Fixed-load replay changes station supply only. Historical paired replay also changes vehicle load.')
    return parser.parse_args()


def _vstry_dir() -> Path:
    return ROOT


def _runtime_spec(transport_rovers):
    drone_profiles = [rv for rv in transport_rovers if str(rv.get("mobility", "ground")).lower() == "air"]
    if len(drone_profiles) == 0:
        return 15.0 / 60.0, 288, 0
    sample_interval_s = float(drone_profiles[0].get("transport_sample_interval_s", 0.1))
    leg_steps = max(len(drone_profiles[0].get("transport_power_profile_kW", [])), 1)
    dt_hours = sample_interval_s / 3600.0
    # Keep the confirmed ~17.7 min horizon: 10 round trips of ideal flight time.
    n_steps = 2 * ROUND_TRIP_COUNT * leg_steps
    # Rovers keep exploring while the drones run their missions; in the final `return_tail`
    # steps the rovers head back to the nearest station so their trajectories end at a charger.
    # The tail is carved out of the horizon (it does NOT lengthen it).
    return_tail = max(int(round(leg_steps * 0.6)), 1)
    return dt_hours, n_steps, return_tail


def _prepare_soc_case(stations, explore_rovers, transport_rovers, init_tag):
    case = INIT_TEMPLATES[init_tag]
    stations = copy.deepcopy(stations)
    explore_rovers = copy.deepcopy(explore_rovers)
    transport_rovers = copy.deepcopy(transport_rovers)

    for idx, rv in enumerate(explore_rovers):
        rv["soc0"] = float(case["rovers"][idx])
    for idx, rv in enumerate(transport_rovers):
        rv["soc0"] = float(case["drones"][idx])
    for idx, st in enumerate(stations):
        st["soc0_station"] = float(case["stations"][idx])
    return stations, explore_rovers, transport_rovers


EXPERIMENT_DIR = VEHICLE_POWER
VEHICLE_POWER_CSV = EXPERIMENT_DIR / "power_25min_5s_all_vehicles.csv"
DRONE_POWER_CSV = EXPERIMENT_DIR / "power_25min_5s_all_drones.csv"
EXPERIMENT_SAMPLE_INTERVAL_S = 5.0
NORMAL_COLUMN = "normal"
PERIOD_COLUMN_PREFIX = {"AM": "mor", "NN": "noon", "PM": "eve"}
# Init SoC level -> 1-based device column indices inside each period block (3 devices each).
INIT_COLUMN_TRIPLE = {"H": (1, 2, 3), "M": (4, 5, 6), "L": (7, 8, 9), "X": (10, 11, 12)}

_EXPERIMENT_FRAME_CACHE = {}


def _experiment_frame(csv_path):
    key = str(csv_path)
    if key not in _EXPERIMENT_FRAME_CACHE:
        _EXPERIMENT_FRAME_CACHE[key] = pd.read_csv(csv_path)
    return _EXPERIMENT_FRAME_CACHE[key]


def _experiment_column_names(period_tag, init_tag, renewable_kind):
    # Predicted/forecast side draws the indoor-proxy 'normal' output power for every device;
    # actual side draws the measured output power of the matching period + SoC-level columns.
    if renewable_kind == "predicted":
        return [NORMAL_COLUMN, NORMAL_COLUMN, NORMAL_COLUMN]
    prefix = PERIOD_COLUMN_PREFIX[period_tag]
    return [f"{prefix}{i}" for i in INIT_COLUMN_TRIPLE[init_tag]]


def _experiment_traces_kW(csv_path, period_tag, init_tag, renewable_kind):
    frame = _experiment_frame(csv_path)
    cols = _experiment_column_names(period_tag, init_tag, renewable_kind)
    # Measured output power is logged in watts; convert to kW for the kWh-based simulator.
    return [np.asarray(frame[col].to_numpy(dtype=float)) / 1000.0 for col in cols]


def _apply_vehicle_output_mode(explore_rovers, transport_rovers, period_tag, init_tag, renewable_kind, comparison_protocol='historical_paired'):
    explore_rovers = copy.deepcopy(explore_rovers)
    transport_rovers = copy.deepcopy(transport_rovers)

    load_kind = 'actual' if comparison_protocol=='fixed_load' else renewable_kind
    vehicle_cols = _experiment_column_names(period_tag, init_tag, load_kind)
    drone_cols = _experiment_column_names(period_tag, init_tag, load_kind)
    vehicle_traces = _experiment_traces_kW(VEHICLE_POWER_CSV, period_tag, init_tag, load_kind)
    drone_traces = _experiment_traces_kW(DRONE_POWER_CSV, period_tag, init_tag, load_kind)

    for idx, rv in enumerate(explore_rovers):
        rv["work_power_profile_kW"] = vehicle_traces[idx % len(vehicle_traces)]
        rv["work_sample_interval_s"] = EXPERIMENT_SAMPLE_INTERVAL_S

    for idx, rv in enumerate(transport_rovers):
        rv["work_power_profile_kW"] = drone_traces[idx % len(drone_traces)]
        rv["work_sample_interval_s"] = EXPERIMENT_SAMPLE_INTERVAL_S

    meta = {
        "vehicle_output_mode": "measured_experiment_power",
        "comparison_protocol": comparison_protocol,
        "renewable_kind": renewable_kind,
        "sample_interval_s": EXPERIMENT_SAMPLE_INTERVAL_S,
        "unit_conversion": "W->kW (divided by 1000)",
        "vehicle_csv": VEHICLE_POWER_CSV.name,
        "drone_csv": DRONE_POWER_CSV.name,
        "explore_vehicle_columns": vehicle_cols,
        "transport_drone_columns": drone_cols,
        "note": (
            "Actual case drives rover/drone WORK-phase load from the period+SoC-level measured "
            "output-power columns; predicted case drives it from the indoor-proxy 'normal' column. "
            "Drone leg timing still follows transport_power_profile_kW; only the work-phase load "
            "magnitude comes from the measured experiment trace."
        ),
    }
    if comparison_protocol=='fixed_load':
        meta['note']='Both cases use identical period/initial-charge measured WORK-load columns; only station supply changes.'
    return explore_rovers, transport_rovers, meta


def _build_station_profiles(stations, period_tag, n_steps, dt_hours, renewable_kind, comparison_protocol='historical_paired'):
    vstry_dir = str(_vstry_dir())
    stations = copy.deepcopy(stations)
    station_by_id = {int(st["id"]): st for st in stations}

    for sid, spec in SOURCE_MAP.items():
        start_time = PERIOD_STARTS[period_tag][spec["domain"]]
        actual_frame = load_vstry_power_frame(
            vstry_dir=vstry_dir,
            domain=spec["domain"],
            source_name=spec["source_name"],
            kind="actual",
            model_tag=spec["model_tag"],
        )
        case_frame = load_vstry_power_frame(
            vstry_dir=vstry_dir,
            domain=spec["domain"],
            source_name=spec["source_name"],
            kind=renewable_kind,
            model_tag=spec["model_tag"],
        )
        calibration = actual_frame
        if comparison_protocol=='fixed_load':
            # Use a fixed prefix ending before the earliest replay period.
            earliest = min(pd.Timestamp(p[spec['domain']]) for p in PERIOD_STARTS.values())
            calibration = actual_frame.loc[actual_frame.Time < earliest.normalize()]
            if calibration.empty: raise ValueError('No pre-replay data available for station calibration')
        actual_power_full = calibration["Power"].to_numpy(dtype=float)
        scale_ratio = float(station_by_id[sid].get("pv_peak_kW", 0.20)) / max(float(np.max(actual_power_full)), 1e-9)
        sampled = resample_power_frame(case_frame, start_time=start_time, n_steps=n_steps, dt_hours=dt_hours) * scale_ratio
        if comparison_protocol=='fixed_load': sampled=np.clip(sampled,0,float(station_by_id[sid].get('pv_peak_kW',0.20)))

        station_by_id[sid]["renewable_profile_kW"] = sampled
        station_by_id[sid]["renewable_kind"] = renewable_kind
        station_by_id[sid]["renewable_label"] = (
            "PV actual"
            if renewable_kind == "actual" and spec["domain"] == "pv"
            else "Wind actual"
            if renewable_kind == "actual"
            else "PV forecast"
            if spec["domain"] == "pv"
            else "Wind forecast"
        )
        station_by_id[sid]["renewable_source_name"] = spec["source_name"]
        station_by_id[sid]["renewable_start_time"] = start_time
        station_by_id[sid]["renewable_domain"] = spec["domain"]
        station_by_id[sid]["renewable_model_tag"] = spec["model_tag"]
        station_by_id[sid]["renewable_scale_ratio"] = scale_ratio
    return stations


def _simulate_once(stations, explore_rovers, transport_rovers, scene, dt_hours, n_steps, return_tail_steps=0):
    min_charge_steps = max(1, int(round(18.0 / max(dt_hours * 3600.0, 1e-9))))
    min_charge_steps_transport = max(1, int(round(28.0 / max(dt_hours * 3600.0, 1e-9))))
    return simulate_mixed_system(
        pv_env=np.ones(int(n_steps), dtype=float),
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
        round_trip_count=ROUND_TRIP_COUNT,
        return_tail_steps=return_tail_steps,
    )


def _class_mae_rmse(pred_mat, actual_mat):
    diff = np.asarray(pred_mat, dtype=float) - np.asarray(actual_mat, dtype=float)
    mae_per_entity = np.mean(np.abs(diff), axis=0)
    rmse_per_entity = np.sqrt(np.mean(diff ** 2, axis=0))
    return float(np.mean(mae_per_entity)), float(np.mean(rmse_per_entity))


def _entity_mae_rmse(names, pred_mat, actual_mat):
    diff = np.asarray(pred_mat, dtype=float) - np.asarray(actual_mat, dtype=float)
    mae_per_entity = np.mean(np.abs(diff), axis=0)
    rmse_per_entity = np.sqrt(np.mean(diff ** 2, axis=0))
    return [
        {"name": str(names[idx]), "mae": float(mae_per_entity[idx]), "rmse": float(rmse_per_entity[idx])}
        for idx in range(len(names))
    ]


def _ratio_percent(pred_value, actual_value):
    pred_value = float(pred_value)
    actual_value = float(actual_value)
    if actual_value <= 1e-9:
        return 100.0 if pred_value <= 1e-9 else 0.0
    # Completion ratio is capped at 100%: 100% means all tasks were completed.
    return min(100.0, 100.0 * pred_value / actual_value)


def _total_station_energy(res, dt_hours):
    # Total energy the charging stations actually delivered to vehicles over the horizon.
    # This depends on each trial's initial SoC and vehicle dynamics, so it differs per group
    # (unlike pure renewable generation, which only depends on the period).
    delivered = np.asarray(res["P_station_used"], dtype=float)
    return float(np.sum(delivered)) * float(dt_hours)


def _safe_name(name):
    text = re.sub(r"\s+", "_", str(name).strip())
    text = re.sub(r"[^0-9A-Za-z_\-]+", "", text)
    return text or "entity"


def _time_frame(n_steps, dt_hours):
    step = np.arange(int(n_steps), dtype=int)
    time_hours = step.astype(float) * float(dt_hours)
    time_seconds = time_hours * 3600.0
    time_minutes = time_seconds / 60.0
    return pd.DataFrame(
        {
            "step": step,
            "time_hours": time_hours,
            "time_seconds": time_seconds,
            "time_minutes": time_minutes,
        }
    )


def _plot_time_axis(n_steps, dt_hours):
    base = _time_frame(n_steps, dt_hours)
    if len(base) > 0 and float(base["time_seconds"].iloc[-1]) >= 600.0:
        return "Time (min)", base["time_minutes"].to_numpy(dtype=float)
    return "Time (s)", base["time_seconds"].to_numpy(dtype=float)


def _station_power_timeseries(stations, res):
    n_steps = int(res["soc"].shape[0])
    n_stations = len(stations)
    p_in = np.zeros((n_steps, n_stations), dtype=float)
    p_out = np.zeros((n_steps, n_stations), dtype=float)

    charge_station_id = np.asarray(res["charge_station_id"], dtype=int)
    p_station_used = np.asarray(res["P_station_used"], dtype=float)

    for station_idx, st in enumerate(stations):
        renewable_profile = np.asarray(st.get("renewable_profile_kW", np.zeros(n_steps)), dtype=float).reshape(-1)
        p_in[:, station_idx] = renewable_profile[:n_steps]
        sid = int(st["id"])
        mask = (charge_station_id == sid).astype(float)
        p_out[:, station_idx] = np.sum(p_station_used * mask, axis=1)
    return p_in, p_out


def _plot_station_power_io(out_path, t_axis, stations, res, x_label):
    p_in, p_out = _station_power_timeseries(stations, res)
    n_stations = len(stations)
    fig, axes = plt.subplots(n_stations, 1, figsize=(10.8, 2.6 * n_stations), sharex=True)
    if n_stations == 1:
        axes = [axes]

    for idx, st in enumerate(stations):
        ax = axes[idx]
        sid = int(st["id"])
        ax.plot(t_axis, p_in[:, idx], label=f"{st.get('renewable_label', 'Renewable input')} (kW)")
        ax.plot(t_axis, p_out[:, idx], label="Output to rovers (kW)")
        ax.axhline(float(st.get("P_aux_kW", 0.0)), linestyle="--", linewidth=1.2, label="Station aux load (kW)")
        title = f"Station S{sid}"
        source_name = st.get("renewable_source_name")
        start_time = st.get("renewable_start_time")
        if source_name and start_time:
            title += f" | {source_name} @ {start_time}"
        ax.set_title(title)
        ax.set_ylabel("kW")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

    axes[-1].set_xlabel(x_label)
    fig.suptitle("Charging Station Input/Output Power")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _unique_cargo_pairs(transport_rovers):
    cargo_pairs = []
    seen = set()
    for rv in transport_rovers:
        for cp in rv.get("cargo_pairs", []):
            cid = int(cp["cargo_id"])
            if cid in seen:
                continue
            seen.add(cid)
            cargo_pairs.append(cp)
    cargo_pairs.sort(key=lambda item: int(item["cargo_id"]))
    return cargo_pairs


def _write_vehicle_soc_csv(out_path, res, dt_hours):
    df = _time_frame(res["soc"].shape[0], dt_hours)
    for idx, rv in enumerate(res["rovers"]):
        key = _safe_name(rv["name"])
        df[f"{key}_soc"] = np.asarray(res["soc"][:, idx], dtype=float)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def _write_vehicle_power_csv(out_path, res, dt_hours):
    df = _time_frame(res["soc"].shape[0], dt_hours)
    for idx, rv in enumerate(res["rovers"]):
        key = _safe_name(rv["name"])
        df[f"{key}_P_load_kW"] = np.asarray(res["P_load"][:, idx], dtype=float)
        df[f"{key}_P_pv_kW"] = np.asarray(res["P_pv"][:, idx], dtype=float)
        df[f"{key}_P_station_used_kW"] = np.asarray(res["P_station_used"][:, idx], dtype=float)
        df[f"{key}_P_bat_kW"] = np.asarray(res["P_bat"][:, idx], dtype=float)
        df[f"{key}_unmet_kW"] = np.asarray(res["unmet"][:, idx], dtype=float)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def _write_vehicle_state_csv(out_path, res, dt_hours):
    df = _time_frame(res["soc"].shape[0], dt_hours)
    for idx, rv in enumerate(res["rovers"]):
        key = _safe_name(rv["name"])
        df[f"{key}_x_m"] = np.asarray(res["x"][:, idx], dtype=float)
        df[f"{key}_y_m"] = np.asarray(res["y"][:, idx], dtype=float)
        df[f"{key}_mode"] = np.asarray(res["mode"][:, idx], dtype=object)
        df[f"{key}_charge_station_id"] = np.asarray(res["charge_station_id"][:, idx], dtype=int)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def _write_station_history_csv(out_path, stations, res, dt_hours):
    df = _time_frame(res["soc"].shape[0], dt_hours)
    p_in, p_out = _station_power_timeseries(stations, res)
    for idx, st in enumerate(stations):
        sid = int(st["id"])
        key = f"S{sid}"
        df[f"{key}_soc"] = np.asarray(res["station_soc"][:, idx], dtype=float)
        df[f"{key}_queue"] = np.asarray(res["station_queue"][:, idx], dtype=int)
        df[f"{key}_renewable_in_kW"] = p_in[:, idx]
        df[f"{key}_wireless_out_kW"] = p_out[:, idx]
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def _write_transport_mission_csv(out_path, res, dt_hours):
    df = _time_frame(res["soc"].shape[0], dt_hours)
    for idx in res["transport_idx"]:
        rv = res["rovers"][idx]
        key = _safe_name(rv["name"])
        df[f"{key}_job_id"] = np.asarray(res["transport_job_hist"][:, idx], dtype=int)
        df[f"{key}_cargo_id"] = np.asarray(res["transport_cargo_hist"][:, idx], dtype=int)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def _extract_mode_transitions(res, dt_hours):
    columns = [
        "rover_name",
        "rover_type",
        "step",
        "time_hours",
        "time_seconds",
        "time_minutes",
        "from_mode",
        "to_mode",
        "x_m",
        "y_m",
        "charge_station_id",
    ]
    rows = []
    mode_hist = np.asarray(res["mode"], dtype=object)
    charge_station_id = np.asarray(res["charge_station_id"], dtype=int)
    x_hist = np.asarray(res["x"], dtype=float)
    y_hist = np.asarray(res["y"], dtype=float)

    for idx, rv in enumerate(res["rovers"]):
        prev_mode = str(mode_hist[0, idx])
        for step in range(1, mode_hist.shape[0]):
            cur_mode = str(mode_hist[step, idx])
            if cur_mode == prev_mode:
                continue
            rows.append(
                {
                    "rover_name": rv["name"],
                    "rover_type": rv.get("type"),
                    "step": int(step),
                    "time_hours": float(step) * float(dt_hours),
                    "time_seconds": float(step) * float(dt_hours) * 3600.0,
                    "time_minutes": float(step) * float(dt_hours) * 60.0,
                    "from_mode": prev_mode,
                    "to_mode": cur_mode,
                    "x_m": float(x_hist[step, idx]),
                    "y_m": float(y_hist[step, idx]),
                    "charge_station_id": int(charge_station_id[step, idx]),
                }
            )
            prev_mode = cur_mode
    return pd.DataFrame(rows, columns=columns)


def _extract_key_event_summary(res, stations, dt_hours):
    summary = {"rovers": [], "stations": []}
    mode_hist = np.asarray(res["mode"], dtype=object)
    soc_hist = np.asarray(res["soc"], dtype=float)
    station_soc = np.asarray(res["station_soc"], dtype=float)
    station_queue = np.asarray(res["station_queue"], dtype=int)

    for idx, rv in enumerate(res["rovers"]):
        entry = {
            "name": rv["name"],
            "type": rv.get("type"),
            "final_soc": float(soc_hist[-1, idx]),
            "final_mode": str(mode_hist[-1, idx]),
            "first_modes": {},
        }
        if rv.get("type") == "explore":
            entry["completion_count"] = int(res["explore_completion_count"][idx])
        else:
            entry["delivery_count"] = int(res["transport_delivery_count"][idx])

        for target_mode in ["GO_STATION", "WAIT_STATION", "CHARGE", "RETURN_RESUME", "WAIT_PV"]:
            hit = np.where(mode_hist[:, idx] == target_mode)[0]
            if len(hit) == 0:
                continue
            step = int(hit[0])
            entry["first_modes"][target_mode] = {
                "step": step,
                "time_hours": float(step) * float(dt_hours),
                "time_seconds": float(step) * float(dt_hours) * 3600.0,
                "time_minutes": float(step) * float(dt_hours) * 60.0,
            }
        summary["rovers"].append(entry)

    for idx, st in enumerate(stations):
        queue_series = station_queue[:, idx]
        soc_series = station_soc[:, idx]
        max_queue_idx = int(np.argmax(queue_series))
        min_soc_idx = int(np.argmin(soc_series))
        summary["stations"].append(
            {
                "id": int(st["id"]),
                "renewable_label": st.get("renewable_label"),
                "renewable_source_name": st.get("renewable_source_name"),
                "renewable_start_time": st.get("renewable_start_time"),
                "max_queue": int(queue_series[max_queue_idx]),
                "max_queue_step": max_queue_idx,
                "max_queue_time_minutes": float(max_queue_idx) * float(dt_hours) * 60.0,
                "min_soc": float(soc_series[min_soc_idx]),
                "min_soc_step": min_soc_idx,
                "min_soc_time_minutes": float(min_soc_idx) * float(dt_hours) * 60.0,
            }
        )
    return summary


def _case_metadata(group, period_tag, init_tag, case_spec, stations, scene, dt_hours, n_steps, vehicle_meta):
    return {
        "group": group,
        "period": period_tag,
        "period_start": PERIOD_STARTS[period_tag],
        "init_tag": init_tag,
        "init_config": INIT_TEMPLATES[init_tag],
        "renewable_case": case_spec["folder"],
        "renewable_kind": case_spec["renewable_kind"],
        "dt_hours": float(dt_hours),
        "n_steps": int(n_steps),
        "horizon_minutes": float(n_steps) * float(dt_hours) * 60.0,
        "scene_source_file": scene.get("source_file") if scene is not None else None,
        "manual_routes_file": scene.get("manual_routes_file") if scene is not None else None,
        "station_sources": [
            {
                "id": int(st["id"]),
                "renewable_label": st.get("renewable_label"),
                "renewable_source_name": st.get("renewable_source_name"),
                "renewable_start_time": st.get("renewable_start_time"),
                "renewable_scale_ratio": float(st.get("renewable_scale_ratio", 0.0)),
                "renewable_domain": st.get("renewable_domain"),
                "renewable_model_tag": st.get("renewable_model_tag"),
            }
            for st in stations
        ],
        "vehicle_output_interface": vehicle_meta,
    }


def _save_case_artifacts(
    case_dir,
    group,
    period_tag,
    init_tag,
    case_spec,
    stations,
    explore_rovers,
    transport_rovers,
    scene,
    res,
    dt_hours,
    n_steps,
    vehicle_meta,
):
    case_dir.mkdir(parents=True, exist_ok=True)

    x_label, t_axis = _plot_time_axis(n_steps, dt_hours)
    rovers = res["rovers"]
    explore_idx = list(res["explore_idx"])
    transport_idx = list(res["transport_idx"])
    explore_names = [rovers[idx]["name"] for idx in explore_idx]
    transport_names = [rovers[idx]["name"] for idx in transport_idx]
    cargo_pairs = _unique_cargo_pairs(transport_rovers)

    plot_soc(case_dir / "mix_soc_explore.png", t_axis, res["soc"][:, explore_idx], explore_names, title=f"{group} Explore Rovers SOC", x_label=x_label)
    plot_soc(case_dir / "mix_soc_transport.png", t_axis, res["soc"][:, transport_idx], transport_names, title=f"{group} Transport Drones SOC", x_label=x_label)
    plot_power_breakdown_by_station(
        case_dir / "mix_power_explore.png",
        t_axis,
        res["P_load"][:, explore_idx],
        res["P_pv"][:, explore_idx],
        res["P_station_used"][:, explore_idx],
        res["charge_station_id"][:, explore_idx],
        explore_names,
        stations,
        title=f"{group} Explore Power Breakdown",
        x_label=x_label,
    )
    plot_power_breakdown_by_station(
        case_dir / "mix_power_transport.png",
        t_axis,
        res["P_load"][:, transport_idx],
        res["P_pv"][:, transport_idx],
        res["P_station_used"][:, transport_idx],
        res["charge_station_id"][:, transport_idx],
        transport_names,
        stations,
        title=f"{group} Transport Drone Power Breakdown",
        x_label=x_label,
    )
    plot_traj(
        case_dir / "mix_traj_explore.png",
        res["x"][:, explore_idx],
        res["y"][:, explore_idx],
        explore_names,
        stations,
        rover_start_pos=[rv["start_pos"] for rv in explore_rovers],
        cargo_pairs=None,
        scene=scene,
        title=f"{group} Explore Trajectories ({case_spec['folder']})",
    )
    plot_traj(
        case_dir / "mix_traj_transport.png",
        res["x"][:, transport_idx],
        res["y"][:, transport_idx],
        transport_names,
        stations,
        rover_start_pos=[rv["start_pos"] for rv in transport_rovers],
        cargo_pairs=cargo_pairs,
        scene=scene,
        title=f"{group} Transport Drone Trajectories ({case_spec['folder']})",
    )
    plot_traj(
        case_dir / "mix_traj_mixed.png",
        res["x"],
        res["y"],
        [rv["name"] for rv in rovers],
        stations,
        rover_start_pos=[rv.get("start_pos", (0.0, 0.0)) for rv in rovers],
        cargo_pairs=cargo_pairs,
        scene=scene,
        title=f"{group} Mixed Trajectories ({case_spec['folder']})",
    )
    plot_scene_boundary(case_dir / "scene_boundary.png", stations, scene=scene, title=f"{group} Scene Boundary")
    plot_station_soc_and_queue(
        case_dir / "mix_station_soc.png",
        case_dir / "mix_station_queue.png",
        t_axis,
        res["station_soc"],
        res["station_queue"],
        stations,
        x_label=x_label,
    )
    _plot_station_power_io(case_dir / "mix_station_power_io.png", t_axis, stations, res, x_label)

    for idx in transport_idx:
        rv = rovers[idx]
        plot_transport_tasks_colored(
            case_dir / f"transport_tasks_colored_{_safe_name(rv['name'])}.png",
            x=res["x"][:, idx],
            y=res["y"][:, idx],
            job_id=res["transport_job_hist"][:, idx],
            cargo_id=res["transport_cargo_hist"][:, idx],
            cargo_pairs=cargo_pairs,
            stations=stations,
            scene=scene,
            rover_name=rv["name"],
        )

    write_rover_positions_to_excel(
        excel_path=str(case_dir / "mix_rover_positions.xlsx"),
        t=np.arange(int(n_steps)),
        rover_names=[rv["name"] for rv in rovers],
        x_mat=res["x"],
        y_mat=res["y"],
        dt_hours=dt_hours,
        origin_lower_left_zero=False,
    )

    _write_vehicle_soc_csv(case_dir / "vehicle_soc_history.csv", res, dt_hours)
    _write_vehicle_power_csv(case_dir / "vehicle_output_power_history.csv", res, dt_hours)
    _write_vehicle_state_csv(case_dir / "vehicle_state_history.csv", res, dt_hours)
    _write_station_history_csv(case_dir / "station_history.csv", stations, res, dt_hours)
    _write_transport_mission_csv(case_dir / "transport_mission_history.csv", res, dt_hours)
    _extract_mode_transitions(res, dt_hours).to_csv(case_dir / "mode_transitions.csv", index=False, encoding="utf-8-sig")

    key_events = _extract_key_event_summary(res, stations, dt_hours)
    (case_dir / "key_events.json").write_text(json.dumps(key_events, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = _case_metadata(group, period_tag, init_tag, case_spec, stations, scene, dt_hours, n_steps, vehicle_meta)
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_trial(group, period_tag, init_tag, actual_case, pred_case, actual_res, pred_res, dt_hours):
    rover_pred = pred_res["soc"][:, pred_res["explore_idx"]]
    rover_actual = actual_res["soc"][:, actual_res["explore_idx"]]
    drone_pred = pred_res["soc"][:, pred_res["transport_idx"]]
    drone_actual = actual_res["soc"][:, actual_res["transport_idx"]]
    station_pred = pred_res["station_soc"]
    station_actual = actual_res["station_soc"]

    rover_mae, rover_rmse = _class_mae_rmse(rover_pred, rover_actual)
    drone_mae, drone_rmse = _class_mae_rmse(drone_pred, drone_actual)
    station_mae, station_rmse = _class_mae_rmse(station_pred, station_actual)

    cr_r = _ratio_percent(
        np.sum(pred_res["explore_completion_count"][pred_res["explore_idx"]]),
        np.sum(actual_res["explore_completion_count"][actual_res["explore_idx"]]),
    )
    cr_d = _ratio_percent(
        np.sum(pred_res["transport_delivery_count"][pred_res["transport_idx"]]),
        np.sum(actual_res["transport_delivery_count"][actual_res["transport_idx"]]),
    )

    e_actual = _total_station_energy(actual_res, dt_hours)
    e_pred = _total_station_energy(pred_res, dt_hours)
    epsilon_e = 0.0 if e_actual <= 1e-9 else abs(e_pred - e_actual) / e_actual * 100.0

    rover_names = [actual_res["rovers"][idx]["name"] for idx in actual_res["explore_idx"]]
    drone_names = [actual_res["rovers"][idx]["name"] for idx in actual_res["transport_idx"]]
    station_names = [f"S{int(st['id'])}" for st in actual_case]

    row = {
        "group": group,
        "period": period_tag,
        "init": init_tag,
        "rover_mae": rover_mae,
        "rover_rmse": rover_rmse,
        "drone_mae": drone_mae,
        "drone_rmse": drone_rmse,
        "station_mae": station_mae,
        "station_rmse": station_rmse,
        "cr_r": cr_r,
        "cr_d": cr_d,
        "epsilon_e": epsilon_e,
        "actual_explore_completed": int(np.sum(actual_res["explore_completion_count"][actual_res["explore_idx"]])),
        "pred_explore_completed": int(np.sum(pred_res["explore_completion_count"][pred_res["explore_idx"]])),
        "actual_transport_completed": int(np.sum(actual_res["transport_delivery_count"][actual_res["transport_idx"]])),
        "pred_transport_completed": int(np.sum(pred_res["transport_delivery_count"][pred_res["transport_idx"]])),
        "actual_station_energy_kwh": e_actual,
        "pred_station_energy_kwh": e_pred,
    }
    detail = {
        "row": row,
        "entity_errors": {
            "rovers": _entity_mae_rmse(rover_names, rover_pred, rover_actual),
            "drones": _entity_mae_rmse(drone_names, drone_pred, drone_actual),
            "stations": _entity_mae_rmse(station_names, station_pred, station_actual),
        },
    }
    return row, detail


def _overall_mean(rows):
    keys = [
        "rover_mae",
        "rover_rmse",
        "drone_mae",
        "drone_rmse",
        "station_mae",
        "station_rmse",
        "cr_r",
        "cr_d",
        "epsilon_e",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _latex_row(row):
    return (
        f"{row['group']:<4} & {row['period']} & {row['init']} & "
        f"{row['rover_mae']:.3f} & {row['rover_rmse']:.3f} & "
        f"{row['drone_mae']:.3f} & {row['drone_rmse']:.3f} & "
        f"{row['station_mae']:.3f} & {row['station_rmse']:.3f} & "
        f"{row['cr_r']:.1f} & {row['cr_d']:.1f} & {row['epsilon_e']:.2f} {LATEX_EOL}"
    )


def _selected_groups(requested_groups):
    if not requested_groups:
        return GROUP_SEQUENCE
    wanted = {str(group).strip().upper() for group in requested_groups}
    selected = [item for item in GROUP_SEQUENCE if item[0] in wanted]
    missing = sorted(wanted.difference({item[0] for item in selected}))
    if missing:
        raise ValueError(f"Unknown group ids: {', '.join(missing)}")
    return selected


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_stations, base_explore_rovers, base_transport_rovers, scene = build_manual_mixed_scenario()
    dt_hours, n_steps, return_tail = _runtime_spec(base_transport_rovers)
    selected_groups = _selected_groups(args.groups)

    rows = []
    detailed_rows = []
    run_manifest = {
        "output_dir": str(output_dir),
        "vehicle_output_mode": args.vehicle_output_mode,
        "comparison_protocol": args.comparison_protocol,
        "evaluation_scope": "Offline replay with archived models; not an independent forecasting test.",
        "dt_hours": float(dt_hours),
        "n_steps": int(n_steps),
        "horizon_minutes": float(n_steps) * float(dt_hours) * 60.0,
        "groups": [],
        "note": (
            "Paired predicted-vs-actual batch. Stations use forecast vs measured renewable generation; "
            "rover/drone WORK load uses the indoor-proxy 'normal' vs measured period+SoC-level output "
            "power from VSgo/experiment. Rover/drone MAE/RMSE capture the measured-vs-proxy SoC gap."
        ),
    }

    if args.comparison_protocol=='fixed_load':
        run_manifest['note']='Fixed-load offline replay: identical measured vehicle loads, different station renewable supply; calibration uses only data before the first replay day.'
    for group, period_tag, init_tag in selected_groups:
        group_dir = output_dir / f"{group}_{period_tag}_{init_tag}"
        group_dir.mkdir(parents=True, exist_ok=True)

        stations_case, explore_case, transport_case = _prepare_soc_case(
            base_stations,
            base_explore_rovers,
            base_transport_rovers,
            init_tag,
        )

        actual_res = None
        pred_res = None
        actual_stations = None
        pred_stations = None
        case_metas = {}
        group_manifest = {
            "group": group,
            "period": period_tag,
            "init": init_tag,
            "folder": str(group_dir),
            "cases": [],
        }

        for case_spec in CASE_SEQUENCE:
            case_dir = group_dir / case_spec["folder"]
            stations_for_case = _build_station_profiles(stations_case, period_tag, n_steps, dt_hours, case_spec["renewable_kind"], args.comparison_protocol)
            explore_for_case, transport_for_case, vehicle_meta = _apply_vehicle_output_mode(
                explore_case,
                transport_case,
                period_tag,
                init_tag,
                case_spec["renewable_kind"],
                args.comparison_protocol,
            )
            case_metas[case_spec["renewable_kind"]] = vehicle_meta
            res = _simulate_once(stations_for_case, explore_for_case, transport_for_case, scene, dt_hours, n_steps, return_tail_steps=return_tail)

            _save_case_artifacts(
                case_dir,
                group,
                period_tag,
                init_tag,
                case_spec,
                stations_for_case,
                explore_for_case,
                transport_for_case,
                scene,
                res,
                dt_hours,
                n_steps,
                vehicle_meta,
            )

            group_manifest["cases"].append({"folder": str(case_dir), "renewable_kind": case_spec["renewable_kind"]})
            if case_spec["renewable_kind"] == "actual":
                actual_res = res
                actual_stations = stations_for_case
            else:
                pred_res = res
                pred_stations = stations_for_case

        row, detail = _summarize_trial(group, period_tag, init_tag, actual_stations, pred_stations, actual_res, pred_res, dt_hours)
        comparison_summary = {
            "row": row,
            "entity_errors": detail["entity_errors"],
            "period_start": PERIOD_STARTS[period_tag],
            "init_config": INIT_TEMPLATES[init_tag],
            "vehicle_output_interface": case_metas,
            "comparison_note": (
                "Paired trial: the actual case drives station renewable input with measured generation "
                "and rover/drone WORK load with the measured period+SoC-level output power; the predicted "
                "case uses forecast generation and the indoor-proxy 'normal' output power. Rover/drone "
                "MAE/RMSE therefore capture the SoC gap between measured and proxy vehicle output power."
            ),
        }
        comparison_summary['comparison_protocol']=args.comparison_protocol
        if args.comparison_protocol=='fixed_load':
            comparison_summary['comparison_note']='Only station renewable inputs differ; vehicle WORK-load columns are identical. This remains an offline replay with archived predictors.'
        (group_dir / "comparison_summary.json").write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        rows.append(row)
        detailed_rows.append(comparison_summary)
        run_manifest["groups"].append(group_manifest)
        print(json.dumps(row, ensure_ascii=False))

    overall = _overall_mean(rows)
    pd.DataFrame(rows).to_csv(output_dir / "validation_matrix_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "validation_matrix_results.json").write_text(
        json.dumps({"rows": detailed_rows, "overall_mean": overall, "run_manifest": run_manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latex_lines = [_latex_row(row) for row in rows]
    latex_lines.append(
        "\\multicolumn{3}{l}{\\textbf{Overall Mean}} & "
        f"{overall['rover_mae']:.3f} & {overall['rover_rmse']:.3f} & "
        f"{overall['drone_mae']:.3f} & {overall['drone_rmse']:.3f} & "
        f"{overall['station_mae']:.3f} & {overall['station_rmse']:.3f} & "
        f"{overall['cr_r']:.1f} & {overall['cr_d']:.1f} & {overall['epsilon_e']:.2f} {LATEX_EOL}"
    )
    (output_dir / "validation_matrix_rows.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved summary CSV: {output_dir / 'validation_matrix_results.csv'}")
    print(f"Saved summary JSON: {output_dir / 'validation_matrix_results.json'}")
    print(f"Saved LaTeX rows: {output_dir / 'validation_matrix_rows.tex'}")


if __name__ == "__main__":
    main()
