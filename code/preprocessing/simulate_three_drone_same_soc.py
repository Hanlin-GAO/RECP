from __future__ import annotations

import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


START_SOC_PCT = 95.0
END_SOC_PCT = 10.0
VOLTAGE_MIN_V = 12.0
VOLTAGE_MAX_V = 17.0


@dataclass
class BatteryModel:
    capacity_wh: float
    voltage_slope: float
    voltage_intercept: float

    def voltage_from_soc(self, soc_pct: float) -> float:
        voltage_v = self.voltage_slope * soc_pct + self.voltage_intercept
        return float(np.clip(voltage_v, VOLTAGE_MIN_V, VOLTAGE_MAX_V))


@dataclass
class CycleTemplate:
    label: str
    target_height_m: float
    time_s: np.ndarray
    dt_s: np.ndarray
    height_m: np.ndarray
    phase: np.ndarray
    base_power_w: np.ndarray
    source_file: Path


@dataclass
class SimulationResult:
    label: str
    target_height_m: float
    output_csv: Path
    total_time_s: float
    completed_cycles: int
    average_power_w: float
    average_current_a: float
    start_voltage_v: float
    end_voltage_v: float


def parse_height_label(path: Path) -> tuple[str, float]:
    match = re.search(r"-(\d+)m-idealized\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not parse target height from {path.name}")
    target_height_m = float(match.group(1))
    return f"{int(target_height_m)}m", target_height_m


def discover_idealized_files(base_dir: Path) -> list[Path]:
    files = sorted((base_dir / "processed").glob("*-idealized.csv"))
    if not files:
        raise FileNotFoundError("No idealized CSV files were found under fly/processed.")
    return files


def compute_battery_model(idealized_files: Iterable[Path]) -> BatteryModel:
    capacities_wh = []
    all_soc = []
    all_voltage = []

    for path in idealized_files:
        df = pd.read_csv(path)
        time_s = pd.to_numeric(df["time(millisecond)"], errors="coerce").ffill().fillna(0.0) / 1000.0
        dt_s = time_s.diff().fillna(0.0)
        power_w = pd.to_numeric(df["estimated_power(W)"], errors="coerce").ffill().bfill().fillna(0.0)
        soc_pct = pd.to_numeric(df["battery_percent"], errors="coerce").ffill().bfill()
        voltage_v = pd.to_numeric(df["estimated_pack_voltage(V)"], errors="coerce").ffill().bfill()

        energy_wh = float((power_w * dt_s).sum() / 3600.0)
        soc_drop_pct = float(soc_pct.iloc[0] - soc_pct.iloc[-1])
        if soc_drop_pct > 0.0:
            capacities_wh.append(energy_wh / (soc_drop_pct / 100.0))

        all_soc.extend(soc_pct.tolist())
        all_voltage.extend(voltage_v.tolist())

    if not capacities_wh:
        raise ValueError("Unable to infer battery capacity from the idealized files.")

    voltage_slope, voltage_intercept = np.polyfit(np.asarray(all_soc, dtype=float), np.asarray(all_voltage, dtype=float), 1)
    return BatteryModel(
        capacity_wh=float(np.mean(capacities_wh)),
        voltage_slope=float(voltage_slope),
        voltage_intercept=float(voltage_intercept),
    )


def extract_cycle_template(path: Path) -> CycleTemplate:
    df = pd.read_csv(path)
    label, target_height_m = parse_height_label(path)

    time_s_full = pd.to_numeric(df["time(millisecond)"], errors="coerce").ffill().fillna(0.0) / 1000.0
    height_m_full = pd.to_numeric(df["height_above_takeoff(m)"], errors="coerce").fillna(0.0)
    power_w_full = pd.to_numeric(df["estimated_power(W)"], errors="coerce").ffill().bfill().fillna(0.0)
    phase_full = df["ideal_flight_phase"].fillna("ground").astype(str)

    active_mask = phase_full != "ground"
    if not active_mask.any():
        raise ValueError(f"No active flight phase found in {path.name}")

    start_idx = max(int(np.flatnonzero(active_mask.to_numpy())[0]) - 1, 0)
    end_idx = min(int(np.flatnonzero(active_mask.to_numpy())[-1]) + 1, len(df) - 1)

    cycle = df.iloc[start_idx : end_idx + 1].copy().reset_index(drop=True)
    time_s = pd.to_numeric(cycle["time(millisecond)"], errors="coerce").ffill().fillna(0.0) / 1000.0
    time_s = time_s - float(time_s.iloc[0])
    # pandas 3 may expose a read-only view; this array is modified below.
    dt_s = time_s.diff().fillna(0.0).to_numpy(dtype=float, copy=True)
    if len(dt_s) > 1:
        dt_s[0] = dt_s[1]

    height_m = pd.to_numeric(cycle["height_above_takeoff(m)"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base_power_w = pd.to_numeric(cycle["estimated_power(W)"], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    phase = cycle["ideal_flight_phase"].fillna("ground").astype(str).to_numpy(dtype=object)

    return CycleTemplate(
        label=label,
        target_height_m=target_height_m,
        time_s=time_s.to_numpy(dtype=float),
        dt_s=dt_s,
        height_m=height_m,
        phase=phase,
        base_power_w=base_power_w,
        source_file=path,
    )


def simulate_template(template: CycleTemplate, battery_model: BatteryModel, output_dir: Path) -> SimulationResult:
    rows: list[dict[str, float | int | str]] = []
    soc_pct = START_SOC_PCT
    absolute_time_s = 0.0
    cycle_index = 0

    while soc_pct > END_SOC_PCT:
        cycle_index += 1
        cycle_start_time_s = absolute_time_s

        for sample_index in range(len(template.time_s)):
            power_w = float(template.base_power_w[sample_index])
            voltage_v = battery_model.voltage_from_soc(soc_pct)
            current_a = power_w / voltage_v if voltage_v > 0.0 else 0.0

            rows.append(
                {
                    "time_s": cycle_start_time_s + float(template.time_s[sample_index]),
                    "cycle_index": cycle_index,
                    "sample_index_in_cycle": sample_index,
                    "flight_phase": str(template.phase[sample_index]),
                    "target_height_m": template.target_height_m,
                    "height_m": float(template.height_m[sample_index]),
                    "soc_pct": float(soc_pct),
                    "pack_voltage_v": voltage_v,
                    "current_a": current_a,
                    "power_w": power_w,
                }
            )

            if sample_index == len(template.time_s) - 1:
                break

            dt_s = float(template.time_s[sample_index + 1] - template.time_s[sample_index])
            if dt_s <= 0.0:
                continue

            soc_drop_pct = power_w * dt_s / 3600.0 / battery_model.capacity_wh * 100.0
            next_soc_pct = soc_pct - soc_drop_pct
            if next_soc_pct <= END_SOC_PCT:
                fraction = (soc_pct - END_SOC_PCT) / max(soc_drop_pct, 1e-12)
                final_time_s = cycle_start_time_s + float(template.time_s[sample_index]) + dt_s * fraction
                final_height_m = float(template.height_m[sample_index] + (template.height_m[sample_index + 1] - template.height_m[sample_index]) * fraction)
                final_voltage_v = battery_model.voltage_from_soc(END_SOC_PCT)
                final_current_a = power_w / final_voltage_v if final_voltage_v > 0.0 else 0.0
                rows.append(
                    {
                        "time_s": final_time_s,
                        "cycle_index": cycle_index,
                        "sample_index_in_cycle": sample_index,
                        "flight_phase": str(template.phase[sample_index]),
                        "target_height_m": template.target_height_m,
                        "height_m": final_height_m,
                        "soc_pct": END_SOC_PCT,
                        "pack_voltage_v": final_voltage_v,
                        "current_a": final_current_a,
                        "power_w": power_w,
                    }
                )
                absolute_time_s = final_time_s
                result_df = pd.DataFrame(rows)
                output_csv = output_dir / f"{template.label}_same_soc_95_to_10.csv"
                result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
                return SimulationResult(
                    label=template.label,
                    target_height_m=template.target_height_m,
                    output_csv=output_csv,
                    total_time_s=float(result_df["time_s"].iloc[-1]),
                    completed_cycles=cycle_index,
                    average_power_w=float(result_df["power_w"].mean()),
                    average_current_a=float(result_df["current_a"].mean()),
                    start_voltage_v=float(result_df["pack_voltage_v"].iloc[0]),
                    end_voltage_v=float(result_df["pack_voltage_v"].iloc[-1]),
                )

            soc_pct = next_soc_pct

        absolute_time_s = cycle_start_time_s + float(template.time_s[-1])

    result_df = pd.DataFrame(rows)
    output_csv = output_dir / f"{template.label}_same_soc_95_to_10.csv"
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return SimulationResult(
        label=template.label,
        target_height_m=template.target_height_m,
        output_csv=output_csv,
        total_time_s=float(result_df["time_s"].iloc[-1]),
        completed_cycles=cycle_index,
        average_power_w=float(result_df["power_w"].mean()),
        average_current_a=float(result_df["current_a"].mean()),
        start_voltage_v=float(result_df["pack_voltage_v"].iloc[0]),
        end_voltage_v=float(result_df["pack_voltage_v"].iloc[-1]),
    )


def plot_soc_vs_time(output_dir: Path, result_files: Iterable[Path]) -> None:
    plt.figure(figsize=(12, 5))
    for path in result_files:
        df = pd.read_csv(path)
        label = path.stem.split("_")[0]
        plt.plot(df["time_s"] / 60.0, df["soc_pct"], linewidth=2.0, label=label)
    plt.xlabel("Mission Time (min)")
    plt.ylabel("SoC (%)")
    plt.title("Three Identical Drones: SoC from 95% to 10%")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "soc_vs_time_95_to_10.png", dpi=300)
    plt.close()


def plot_electrical_vs_time(output_dir: Path, result_files: Iterable[Path]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    metric_specs = [
        ("pack_voltage_v", "Voltage (V)", "Pack Voltage"),
        ("current_a", "Current (A)", "Pack Current"),
        ("power_w", "Power (W)", "Electrical Power"),
    ]
    for path in result_files:
        df = pd.read_csv(path)
        label = path.stem.split("_")[0]
        for ax, (column, ylabel, title) in zip(axes, metric_specs):
            ax.plot(df["time_s"] / 60.0, df[column], linewidth=1.8, label=label)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.45)
    axes[-1].set_xlabel("Mission Time (min)")
    for ax in axes:
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "electrical_vs_time_95_to_10.png", dpi=300)
    plt.close(fig)


def plot_height_first_cycles(output_dir: Path, result_files: Iterable[Path], cycles_to_show: int = 3) -> None:
    plt.figure(figsize=(12, 5))
    for path in result_files:
        df = pd.read_csv(path)
        label = path.stem.split("_")[0]
        subset = df[df["cycle_index"] <= cycles_to_show]
        plt.plot(subset["time_s"] / 60.0, subset["height_m"], linewidth=2.0, label=label)
    plt.xlabel("Mission Time (min)")
    plt.ylabel("Height (m)")
    plt.title(f"Repeated Takeoff-Cruise-Land Pattern (First {cycles_to_show} Cycles)")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "height_first_cycles.png", dpi=300)
    plt.close()


def plot_electrical_vs_soc(output_dir: Path, result_files: Iterable[Path]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    metric_specs = [
        ("pack_voltage_v", "Voltage (V)", "Voltage vs SoC"),
        ("current_a", "Current (A)", "Current vs SoC"),
        ("power_w", "Power (W)", "Power vs SoC"),
    ]
    for path in result_files:
        df = pd.read_csv(path)
        label = path.stem.split("_")[0]
        for ax, (column, ylabel, title) in zip(axes, metric_specs):
            ax.plot(df["soc_pct"], df[column], linewidth=1.8, label=label)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.45)
    axes[-1].set_xlabel("SoC (%)")
    axes[-1].invert_xaxis()
    for ax in axes:
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "electrical_vs_soc_95_to_10.png", dpi=300)
    plt.close(fig)


def save_summary(output_dir: Path, battery_model: BatteryModel, results: Iterable[SimulationResult]) -> None:
    result_rows = [
        {
            "label": result.label,
            "target_height_m": result.target_height_m,
            "output_csv": result.output_csv.name,
            "total_time_min": round(result.total_time_s / 60.0, 3),
            "completed_cycles": result.completed_cycles,
            "average_power_w": round(result.average_power_w, 3),
            "average_current_a": round(result.average_current_a, 3),
            "start_voltage_v": round(result.start_voltage_v, 3),
            "end_voltage_v": round(result.end_voltage_v, 3),
            "battery_capacity_wh": round(battery_model.capacity_wh, 3),
            "voltage_slope": round(battery_model.voltage_slope, 6),
            "voltage_intercept": round(battery_model.voltage_intercept, 6),
        }
        for result in results
    ]
    pd.DataFrame(result_rows).to_csv(output_dir / "simulation_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Simulate equal-initial-charge drone cycles from processed flight profiles.")
    parser.add_argument('--processed-dir', type=Path, default=root / 'runs/flight/processed')
    parser.add_argument('--output-dir', type=Path, default=root / 'runs/flight/processed/same_soc_three_drones')
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    idealized_files = sorted(args.processed_dir.glob('*-idealized.csv'))
    if not idealized_files:
        raise FileNotFoundError(f'No idealized CSV files in {args.processed_dir}')
    battery_model = compute_battery_model(idealized_files)
    templates = [extract_cycle_template(path) for path in idealized_files]
    templates.sort(key=lambda item: item.target_height_m)

    results = [simulate_template(template, battery_model, output_dir) for template in templates]
    result_files = [result.output_csv for result in results]

    save_summary(output_dir, battery_model, results)
    plot_soc_vs_time(output_dir, result_files)
    plot_electrical_vs_time(output_dir, result_files)
    plot_height_first_cycles(output_dir, result_files)
    plot_electrical_vs_soc(output_dir, result_files)

    print("Generated files:")
    for result in results:
        print(f"- {result.output_csv.name}")
    print("- simulation_summary.csv")
    print("- soc_vs_time_95_to_10.png")
    print("- electrical_vs_time_95_to_10.png")
    print("- height_first_cycles.png")
    print("- electrical_vs_soc_95_to_10.png")


if __name__ == "__main__":
    main()
