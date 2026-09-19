import csv
import os

import numpy as np
from scheduling_paths import FLIGHT_PROCESSED


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_processed_dir():
    return str(FLIGHT_PROCESSED)


def _read_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _float_or(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _resample_power_profile(csv_path, duration_s, sample_interval_s):
    rows = _read_csv_rows(csv_path)
    samples = {}
    for row in rows:
        t = round(_float_or(row.get("aligned_time_s"), 0.0), 6)
        power_w = max(_float_or(row.get("estimated_power(W)"), 0.0), 0.0)
        samples.setdefault(t, []).append(power_w)

    if not samples:
        grid = np.round(np.arange(0.0, duration_s + sample_interval_s * 0.5, sample_interval_s), 10)
        return grid, np.zeros(len(grid), dtype=float)

    sample_times = np.asarray(sorted(samples.keys()), dtype=float)
    sample_power = np.asarray([np.mean(samples[t]) for t in sample_times], dtype=float)

    grid = np.round(np.arange(0.0, duration_s + sample_interval_s * 0.5, sample_interval_s), 10)
    if len(grid) == 0:
        grid = np.asarray([0.0], dtype=float)

    power_interp_w = np.interp(grid, sample_times, sample_power)
    return grid, np.maximum(power_interp_w, 0.0) / 1000.0


def load_drone_profiles(processed_dir=None, sample_interval_s=0.1):
    processed_dir = processed_dir or _default_processed_dir()
    flight_summary_path = os.path.join(processed_dir, "flight_summary.csv")
    sim_summary_path = os.path.join(processed_dir, "same_soc_three_drones", "simulation_summary.csv")

    flight_rows = _read_csv_rows(flight_summary_path)
    sim_rows = _read_csv_rows(sim_summary_path)
    sim_by_label = {str(row.get("label", "")).strip(): row for row in sim_rows}

    profiles = []
    for row in flight_rows:
        file_name = str(row.get("file", "")).strip()
        duration_s = _float_or(row.get("aligned_duration_s"), 0.0)
        target_height_m = _float_or(row.get("target_height_m"), 0.0)
        label = f"{int(round(target_height_m))}m"

        profile_csv = os.path.join(processed_dir, file_name)
        time_grid_s, power_profile_kW = _resample_power_profile(profile_csv, duration_s, sample_interval_s)

        sim_row = sim_by_label.get(label, {})
        battery_capacity_kWh = _float_or(sim_row.get("battery_capacity_wh"), 64.439) / 1000.0
        average_power_kW = _float_or(sim_row.get("average_power_w"), np.mean(power_profile_kW) * 1000.0) / 1000.0

        profiles.append({
            "label": label,
            "file_name": file_name,
            "target_height_m": target_height_m,
            "duration_s": duration_s,
            "sample_interval_s": float(sample_interval_s),
            "time_grid_s": time_grid_s.tolist(),
            "power_profile_kW": power_profile_kW.tolist(),
            "average_power_kW": average_power_kW,
            "battery_capacity_kWh": battery_capacity_kWh,
        })

    profiles.sort(key=lambda item: item["target_height_m"])
    return profiles
