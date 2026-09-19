
raise RuntimeError('Inspection-only historical code. Execution is disabled; use scripts/reproduce.py train with the reviewed protocol.')

import copy
import csv
import json
from pathlib import Path

import numpy as np

from io_utils import load_vstry_power_frame, resample_power_frame
from scenarios import build_real_mixed_scenario
from simulator import simulate_mixed_system


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "validation_matrix"
RESULTS_DIR.mkdir(exist_ok=True)

ROUND_TRIP_COUNT = 10
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
GROUP_SEQUENCE = [
    ("G1", "AM", "H"), ("G2", "AM", "M"), ("G3", "AM", "L"), ("G4", "AM", "X"),
    ("G5", "NN", "H"), ("G6", "NN", "M"), ("G7", "NN", "L"), ("G8", "NN", "X"),
    ("G9", "PM", "H"), ("G10", "PM", "M"), ("G11", "PM", "L"), ("G12", "PM", "X"),
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


def _vstry_dir() -> Path:
    return BASE_DIR.parents[1] / "VStry"


def _runtime_spec(transport_rovers):
    drone_profiles = [rv for rv in transport_rovers if str(rv.get("mobility", "ground")).lower() == "air"]
    if len(drone_profiles) == 0:
        return 15 / 60, 288
    sample_interval_s = float(drone_profiles[0].get("transport_sample_interval_s", 0.1))
    leg_steps = max(len(drone_profiles[0].get("transport_power_profile_kW", [])), 1)
    dt_hours = sample_interval_s / 3600.0
    n_steps = 2 * ROUND_TRIP_COUNT * leg_steps
    return dt_hours, n_steps


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


def _build_station_profiles(stations, period_tag, n_steps, dt_hours, kind):
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
        frame = load_vstry_power_frame(
            vstry_dir=vstry_dir,
            domain=spec["domain"],
            source_name=spec["source_name"],
            kind=kind,
            model_tag=spec["model_tag"],
        )
        actual_power_full = actual_frame["Power"].to_numpy(dtype=float)
        scale_ratio = float(station_by_id[sid].get("pv_peak_kW", 0.20)) / max(float(np.max(actual_power_full)), 1e-9)
        sampled = resample_power_frame(frame, start_time=start_time, n_steps=n_steps, dt_hours=dt_hours) * scale_ratio
        station_by_id[sid]["renewable_profile_kW"] = sampled
        station_by_id[sid]["renewable_label"] = "PV actual" if (kind == "actual" and spec["domain"] == "pv") else \
            "Wind actual" if (kind == "actual") else \
            "PV forecast" if (spec["domain"] == "pv") else "Wind forecast"
        station_by_id[sid]["renewable_source_name"] = spec["source_name"]
        station_by_id[sid]["renewable_start_time"] = start_time
        station_by_id[sid]["renewable_scale_ratio"] = scale_ratio
    return stations


def _simulate_once(stations, explore_rovers, transport_rovers, scene, dt_hours, n_steps):
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
    )


def _class_mae_rmse(pred_mat, actual_mat):
    diff = np.asarray(pred_mat, dtype=float) - np.asarray(actual_mat, dtype=float)
    mae_per_entity = np.mean(np.abs(diff), axis=0)
    rmse_per_entity = np.sqrt(np.mean(diff ** 2, axis=0))
    return float(np.mean(mae_per_entity)), float(np.mean(rmse_per_entity))


def _ratio_percent(pred_value, actual_value):
    pred_value = float(pred_value)
    actual_value = float(actual_value)
    if actual_value <= 1e-9:
        return 100.0 if pred_value <= 1e-9 else 0.0
    return 100.0 * pred_value / actual_value


def _total_station_energy(stations, dt_hours):
    total = 0.0
    for st in stations:
        total += float(np.sum(np.asarray(st["renewable_profile_kW"], dtype=float))) * float(dt_hours)
    return total


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

    e_actual = _total_station_energy(actual_case, dt_hours)
    e_pred = _total_station_energy(pred_case, dt_hours)
    epsilon_e = 0.0 if e_actual <= 1e-9 else abs(e_pred - e_actual) / e_actual * 100.0

    return {
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


def _overall_mean(rows):
    keys = [
        "rover_mae", "rover_rmse", "drone_mae", "drone_rmse",
        "station_mae", "station_rmse", "cr_r", "cr_d", "epsilon_e",
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _latex_row(row):
    return (
        f"{row['group']:<4} & {row['period']} & {row['init']} & "
        f"{row['rover_mae']:.3f} & {row['rover_rmse']:.3f} & "
        f"{row['drone_mae']:.3f} & {row['drone_rmse']:.3f} & "
        f"{row['station_mae']:.3f} & {row['station_rmse']:.3f} & "
        f"{row['cr_r']:.1f} & {row['cr_d']:.1f} & {row['epsilon_e']:.2f} \\\\" 
    )


def main():
    stations, explore_rovers, transport_rovers, scene = build_real_mixed_scenario()
    dt_hours, n_steps = _runtime_spec(transport_rovers)

    rows = []
    detailed = []
    for group, period_tag, init_tag in GROUP_SEQUENCE:
        st_case, ex_case, tr_case = _prepare_soc_case(stations, explore_rovers, transport_rovers, init_tag)

        actual_stations = _build_station_profiles(st_case, period_tag, n_steps, dt_hours, kind="actual")
        pred_stations = _build_station_profiles(st_case, period_tag, n_steps, dt_hours, kind="predicted")

        actual_res = _simulate_once(actual_stations, ex_case, tr_case, scene, dt_hours, n_steps)
        pred_res = _simulate_once(pred_stations, copy.deepcopy(ex_case), copy.deepcopy(tr_case), scene, dt_hours, n_steps)

        row = _summarize_trial(group, period_tag, init_tag, actual_stations, pred_stations, actual_res, pred_res, dt_hours)
        rows.append(row)
        detailed.append({
            "row": row,
            "period_start": PERIOD_STARTS[period_tag],
            "init_config": INIT_TEMPLATES[init_tag],
        })
        print(json.dumps(row, ensure_ascii=False))

    overall = _overall_mean(rows)

    csv_path = RESULTS_DIR / "validation_matrix_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULTS_DIR / "validation_matrix_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": detailed, "overall_mean": overall}, f, ensure_ascii=False, indent=2)

    latex_lines = [_latex_row(row) for row in rows]
    latex_lines.append(
        "\\multicolumn{3}{l}{\\textbf{Overall Mean}} & "
        f"{overall['rover_mae']:.3f} & {overall['rover_rmse']:.3f} & "
        f"{overall['drone_mae']:.3f} & {overall['drone_rmse']:.3f} & "
        f"{overall['station_mae']:.3f} & {overall['station_rmse']:.3f} & "
        f"{overall['cr_r']:.1f} & {overall['cr_d']:.1f} & {overall['epsilon_e']:.2f} \\\\" 
    )
    tex_path = RESULTS_DIR / "validation_matrix_rows.tex"
    tex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved LaTeX rows: {tex_path}")


if __name__ == "__main__":
    main()
