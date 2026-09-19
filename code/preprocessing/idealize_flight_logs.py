from __future__ import annotations

import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
FTPS_TO_MPH = 0.6818181818181818
AIRBORNE_THRESHOLD_FT = 1.0
TARGET_REACHED_RATIO = 0.95


@dataclass
class FlightProfile:
    input_path: Path
    output_path: Path
    label: str
    target_height_ft: float
    target_height_m: float
    original_duration_s: float
    takeoff_progress: float
    ascent_end_progress: float
    descent_start_progress: float
    touchdown_progress: float
    hover_power_w: float


def discover_input_files(base_dir: Path) -> list[Path]:
    files = sorted(base_dir.glob("*-Flight-Airdata-*.csv"))
    if not files:
        raise FileNotFoundError("No renamed Airdata CSV files were found under fly/.")
    return files


def parse_target_height_ft(path: Path, df: pd.DataFrame) -> float:
    match = re.search(r"-(\d+)m\.csv$", path.name)
    if match:
        return float(match.group(1)) * M_TO_FT

    heights_ft = pd.to_numeric(df["height_above_takeoff(feet)"], errors="coerce").fillna(0.0)
    stable_ft = heights_ft[heights_ft >= 3.0]
    if stable_ft.empty:
        return float(heights_ft.max())
    rounded_ft = stable_ft.round(1)
    return float(rounded_ft.mode().iloc[0])


def compute_pack_voltage(df: pd.DataFrame) -> pd.Series:
    total_v = pd.to_numeric(df.get("voltage(v)"), errors="coerce")
    cell_cols = [col for col in df.columns if re.fullmatch(r"voltageCell\d+", col)]
    if not cell_cols:
        return total_v

    cell_sum = (
        df[cell_cols]
        .apply(pd.to_numeric, errors="coerce")
        .where(lambda frame: frame > 0.0)
        .sum(axis=1, min_count=1)
    )
    return total_v.where(total_v > 0.0, cell_sum)


def find_progress_markers(progress: np.ndarray, height_ft: np.ndarray, target_height_ft: float) -> tuple[float, float, float, float]:
    airborne_idx = np.flatnonzero(height_ft > AIRBORNE_THRESHOLD_FT)
    if airborne_idx.size == 0:
        return 0.0, 0.1, 0.9, 1.0

    takeoff_idx = int(airborne_idx[0])
    touchdown_idx = int(airborne_idx[-1])
    if touchdown_idx < len(height_ft) - 1:
        touchdown_idx += 1

    reached_target = np.flatnonzero((height_ft >= TARGET_REACHED_RATIO * target_height_ft) & (np.arange(len(height_ft)) >= takeoff_idx))
    ascent_end_idx = int(reached_target[0]) if reached_target.size else int(np.argmax(height_ft))

    target_before_touchdown = np.flatnonzero((height_ft >= TARGET_REACHED_RATIO * target_height_ft) & (np.arange(len(height_ft)) <= touchdown_idx))
    descent_start_idx = int(target_before_touchdown[-1]) if target_before_touchdown.size else ascent_end_idx

    if descent_start_idx < ascent_end_idx:
        middle_idx = int(round((ascent_end_idx + touchdown_idx) / 2.0))
        ascent_end_idx = min(ascent_end_idx, middle_idx)
        descent_start_idx = max(middle_idx, ascent_end_idx)

    takeoff_progress = float(progress[takeoff_idx])
    ascent_end_progress = float(progress[ascent_end_idx])
    descent_start_progress = float(progress[descent_start_idx])
    touchdown_progress = float(progress[touchdown_idx])

    ascent_end_progress = max(ascent_end_progress, takeoff_progress + 1e-6)
    descent_start_progress = max(descent_start_progress, ascent_end_progress)
    touchdown_progress = max(touchdown_progress, descent_start_progress + 1e-6)
    return takeoff_progress, ascent_end_progress, descent_start_progress, touchdown_progress


def build_ideal_height_ft(progress: np.ndarray, target_height_ft: float, markers: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    takeoff_progress, ascent_end_progress, descent_start_progress, touchdown_progress = markers
    ideal_height_ft = np.zeros_like(progress, dtype=float)
    phase = np.full(progress.shape, "ground", dtype=object)

    ascent_mask = (progress >= takeoff_progress) & (progress < ascent_end_progress)
    cruise_mask = (progress >= ascent_end_progress) & (progress < descent_start_progress)
    descent_mask = (progress >= descent_start_progress) & (progress <= touchdown_progress)

    if ascent_mask.any():
        ascent_progress = (progress[ascent_mask] - takeoff_progress) / max(ascent_end_progress - takeoff_progress, 1e-6)
        ideal_height_ft[ascent_mask] = target_height_ft * ascent_progress
        phase[ascent_mask] = "ascent"

    if cruise_mask.any():
        ideal_height_ft[cruise_mask] = target_height_ft
        phase[cruise_mask] = "cruise"

    if descent_mask.any():
        descent_progress = (touchdown_progress - progress[descent_mask]) / max(touchdown_progress - descent_start_progress, 1e-6)
        ideal_height_ft[descent_mask] = np.clip(target_height_ft * descent_progress, 0.0, target_height_ft)
        phase[descent_mask] = "descent"

    phase[(progress > touchdown_progress)] = "ground"
    return ideal_height_ft, phase


def format_datetime(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]


def process_one_file(path: Path, output_dir: Path, base_datetime: pd.Timestamp, common_duration_s: float) -> FlightProfile:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]

    if "time(millisecond)" not in df.columns or "datetime(utc)" not in df.columns:
        raise KeyError(f"Missing time columns in {path.name}")

    time_ms = pd.to_numeric(df["time(millisecond)"], errors="coerce").ffill().fillna(0.0)
    elapsed_s = (time_ms - time_ms.iloc[0]) / 1000.0
    original_duration_s = float(elapsed_s.iloc[-1]) if len(elapsed_s) > 1 else 0.0
    if original_duration_s <= 0.0:
        progress = np.linspace(0.0, 1.0, len(df))
    else:
        progress = (elapsed_s / original_duration_s).to_numpy(dtype=float)

    target_height_ft = parse_target_height_ft(path, df)
    height_ft = pd.to_numeric(df["height_above_takeoff(feet)"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    markers = find_progress_markers(progress, height_ft, target_height_ft)
    ideal_height_ft, phase = build_ideal_height_ft(progress, target_height_ft, markers)

    aligned_time_s = progress * common_duration_s
    aligned_time_ms = np.rint(aligned_time_s * 1000.0).astype(int)
    aligned_datetime = base_datetime + pd.to_timedelta(aligned_time_s, unit="s")

    ideal_height_m = ideal_height_ft * FT_TO_M
    if len(df) > 1:
        delta_height_ft = np.diff(ideal_height_ft, prepend=ideal_height_ft[0])
        delta_time_s = np.diff(aligned_time_s, prepend=aligned_time_s[0])
        ideal_vz_ftps = np.divide(
            delta_height_ft,
            delta_time_s,
            out=np.zeros_like(delta_height_ft, dtype=float),
            where=delta_time_s > 0.0,
        )
    else:
        ideal_vz_ftps = np.zeros(len(df), dtype=float)
    ideal_zspeed_mph = ideal_vz_ftps * FTPS_TO_MPH

    x_speed = pd.to_numeric(df.get("xSpeed(mph)"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_speed = pd.to_numeric(df.get("ySpeed(mph)"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ideal_speed_mph = np.sqrt(x_speed ** 2 + y_speed ** 2 + ideal_zspeed_mph ** 2)

    base_altitude_ft = float(pd.to_numeric(df["altitude(feet)"], errors="coerce").fillna(0.0).iloc[0] - height_ft[0]) if "altitude(feet)" in df.columns else 0.0
    base_sea_level_ft = float(pd.to_numeric(df["altitude_above_seaLevel(feet)"], errors="coerce").fillna(0.0).iloc[0] - height_ft[0]) if "altitude_above_seaLevel(feet)" in df.columns else base_altitude_ft

    current_a = pd.to_numeric(df.get("current(A)"), errors="coerce")
    pack_voltage_v = compute_pack_voltage(df)
    estimated_power_w = current_a * pack_voltage_v

    rename_pairs = {
        "time(millisecond)": "original_time(millisecond)",
        "datetime(utc)": "original_datetime(utc)",
        "height_above_takeoff(feet)": "original_height_above_takeoff(feet)",
        "ascent(feet)": "original_ascent(feet)",
        "altitude(feet)": "original_altitude(feet)",
        "altitude_above_seaLevel(feet)": "original_altitude_above_seaLevel(feet)",
        "height_sonar(feet)": "original_height_sonar(feet)",
        "zSpeed(mph)": "original_zSpeed(mph)",
        "speed(mph)": "original_speed(mph)",
        "pitch(degrees)": "original_pitch(degrees)",
        "roll(degrees)": "original_roll(degrees)",
    }
    existing_rename_pairs = {key: value for key, value in rename_pairs.items() if key in df.columns}
    df = df.rename(columns=existing_rename_pairs)

    df.insert(0, "time(millisecond)", aligned_time_ms)
    df.insert(1, "datetime(utc)", format_datetime(pd.Series(aligned_datetime)))
    df.insert(2, "aligned_time_s", aligned_time_s)
    df.insert(3, "aligned_phase", progress)
    df.insert(4, "ideal_flight_phase", phase)

    df["height_above_takeoff(feet)"] = ideal_height_ft
    df["height_above_takeoff(m)"] = ideal_height_m
    df["ascent(feet)"] = ideal_height_ft
    df["altitude(feet)"] = base_altitude_ft + ideal_height_ft
    df["altitude_above_seaLevel(feet)"] = base_sea_level_ft + ideal_height_ft
    if "original_height_sonar(feet)" in df.columns:
        df["height_sonar(feet)"] = ideal_height_ft
    if "original_zSpeed(mph)" in df.columns:
        df["zSpeed(mph)"] = ideal_zspeed_mph
    if "original_speed(mph)" in df.columns:
        df["speed(mph)"] = ideal_speed_mph
    if "original_pitch(degrees)" in df.columns:
        df["pitch(degrees)"] = np.where(phase == "ground", df["original_pitch(degrees)"], 0.0)
    if "original_roll(degrees)" in df.columns:
        df["roll(degrees)"] = np.where(phase == "ground", df["original_roll(degrees)"], 0.0)

    df["estimated_pack_voltage(V)"] = pack_voltage_v
    df["estimated_power(W)"] = estimated_power_w

    output_path = output_dir / path.name.replace(".csv", "-idealized.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    hover_mask = (phase == "cruise") & estimated_power_w.notna().to_numpy()
    hover_power_w = float(estimated_power_w[hover_mask].mean()) if hover_mask.any() else float(estimated_power_w.mean())
    target_height_m = target_height_ft * FT_TO_M
    label = f"{int(round(target_height_m))} m"
    return FlightProfile(
        input_path=path,
        output_path=output_path,
        label=label,
        target_height_ft=target_height_ft,
        target_height_m=target_height_m,
        original_duration_s=original_duration_s,
        takeoff_progress=markers[0],
        ascent_end_progress=markers[1],
        descent_start_progress=markers[2],
        touchdown_progress=markers[3],
        hover_power_w=hover_power_w,
    )


def plot_height_power(output_dir: Path, processed_files: Iterable[Path]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for path in processed_files:
        df = pd.read_csv(path)
        label = path.stem.split("-")[-2]
        axes[0].plot(df["aligned_time_s"], df["height_above_takeoff(m)"], linewidth=2.0, label=label)
        axes[1].plot(df["aligned_time_s"], df["estimated_power(W)"], linewidth=1.8, label=label)
        axes[2].plot(df["aligned_time_s"], df.get("zSpeed(mph)", pd.Series(np.zeros(len(df)))), linewidth=1.8, label=label)

    axes[0].set_ylabel("Height (m)")
    axes[0].set_title("Idealized Height Profiles")
    axes[1].set_ylabel("Power (W)")
    axes[1].set_title("Electrical Output Power")
    axes[2].set_ylabel("zSpeed (mph)")
    axes[2].set_xlabel("Aligned Time (s)")
    axes[2].set_title("Idealized Vertical Speed")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "aligned_height_power.png", dpi=300)
    plt.close(fig)


def plot_battery_channels(output_dir: Path, processed_files: Iterable[Path]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for path in processed_files:
        df = pd.read_csv(path)
        label = path.stem.split("-")[-2]
        axes[0].plot(df["aligned_time_s"], df["battery_percent"], linewidth=2.0, label=label)
        axes[1].plot(df["aligned_time_s"], df["estimated_pack_voltage(V)"], linewidth=1.8, label=label)
        axes[2].plot(df["aligned_time_s"], df["current(A)"], linewidth=1.8, label=label)

    axes[0].set_ylabel("SoC (%)")
    axes[0].set_title("Battery SoC")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].set_title("Pack Voltage")
    axes[2].set_ylabel("Current (A)")
    axes[2].set_xlabel("Aligned Time (s)")
    axes[2].set_title("Pack Current")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "aligned_battery_channels.png", dpi=300)
    plt.close(fig)


def plot_original_vs_ideal_height(output_dir: Path, processed_files: Iterable[Path]) -> None:
    processed_files = list(processed_files)
    fig, axes = plt.subplots(len(processed_files), 1, figsize=(12, 3.6 * len(processed_files)), sharex=True)
    if len(processed_files) == 1:
        axes = [axes]

    for ax, path in zip(axes, processed_files):
        df = pd.read_csv(path)
        label = path.stem.split("-")[-2]
        ax.plot(df["aligned_time_s"], pd.to_numeric(df["original_height_above_takeoff(feet)"], errors="coerce") * FT_TO_M, linewidth=1.5, alpha=0.75, label=f"{label} original")
        ax.plot(df["aligned_time_s"], df["height_above_takeoff(m)"], linewidth=2.2, label=f"{label} idealized")
        ax.set_ylabel("Height (m)")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best")

    axes[0].set_title("Original vs Idealized Height")
    axes[-1].set_xlabel("Aligned Time (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "original_vs_ideal_height.png", dpi=300)
    plt.close(fig)


def save_summary(output_dir: Path, profiles: Iterable[FlightProfile], common_duration_s: float) -> None:
    rows = []
    for profile in profiles:
        rows.append(
            {
                "file": profile.output_path.name,
                "target_height_m": round(profile.target_height_m, 3),
                "original_duration_s": round(profile.original_duration_s, 3),
                "aligned_duration_s": round(common_duration_s, 3),
                "takeoff_progress": round(profile.takeoff_progress, 5),
                "ascent_end_progress": round(profile.ascent_end_progress, 5),
                "descent_start_progress": round(profile.descent_start_progress, 5),
                "touchdown_progress": round(profile.touchdown_progress, 5),
                "hover_power_w": round(profile.hover_power_w, 3),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "flight_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Align flight timing and derive idealized height and electrical profiles.")
    parser.add_argument('--input-dir', type=Path, default=root / 'data/experiments/flight/raw_redacted')
    parser.add_argument('--output-dir', type=Path, default=root / 'runs/flight/processed')
    args = parser.parse_args()
    base_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(base_dir)

    starts = []
    durations = []
    for path in input_files:
        df = pd.read_csv(path)
        df.columns = [col.strip() for col in df.columns]
        starts.append(pd.to_datetime(df["datetime(utc)"], errors="coerce").iloc[0])
        time_ms = pd.to_numeric(df["time(millisecond)"], errors="coerce").ffill().fillna(0.0)
        durations.append(float((time_ms.iloc[-1] - time_ms.iloc[0]) / 1000.0))

    base_datetime = min(starts)
    common_duration_s = max(durations)

    profiles = []
    processed_files = []
    for path in input_files:
        profile = process_one_file(path, output_dir, base_datetime, common_duration_s)
        profiles.append(profile)
        processed_files.append(profile.output_path)

    save_summary(output_dir, profiles, common_duration_s)
    plot_height_power(output_dir, processed_files)
    plot_battery_channels(output_dir, processed_files)
    plot_original_vs_ideal_height(output_dir, processed_files)

    print("Generated files:")
    for profile in profiles:
        print(f"- {profile.output_path.name}")
    print("- flight_summary.csv")
    print("- aligned_height_power.png")
    print("- aligned_battery_channels.png")
    print("- original_vs_ideal_height.png")


if __name__ == "__main__":
    main()
