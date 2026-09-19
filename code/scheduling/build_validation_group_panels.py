import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotting import _scene_bounds
from scenarios import build_manual_mixed_scenario
from scheduling_paths import ROOT


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results_reference/scheduling/manual_first_frame"
DEFAULT_OUTPUT_DIR = ROOT / "runs/figures/grouped"
FIG_LEFT = 0.11
FIG_RIGHT = 0.98
SUPYLABEL_X = 0.008
TRAJECTORY_PANEL_BOTTOM = 0.12
TRAJECTORY_SUPXLABEL_Y = 0.028
LINE_PANEL_BOTTOM = 0.145
LINE_SUPXLABEL_Y = 0.026
TICK_LABEL_SIZE = 17.5
LEGEND_FONT_SIZE = 18

PERIOD_META = {
    "AM": {"label": "Morning (AM)", "folder": "AM_morning"},
    "NN": {"label": "Noon (NN)", "folder": "NN_noon"},
    "PM": {"label": "Afternoon (PM)", "folder": "PM_afternoon"},
}
CASE_META = {
    "actual_generation": {"label": "Actual Generation", "stem": "actual"},
    "predicted_generation": {"label": "Predicted Generation", "stem": "predicted"},
}
INIT_ORDER = ["H", "M", "L", "X"]
INIT_LABELS = {"H": "High", "M": "Medium", "L": "Low", "X": "Mixed"}
INIT_GROUP_ORDER = {
    "AM": {"H": "G1", "M": "G2", "L": "G3", "X": "G4"},
    "NN": {"H": "G5", "M": "G6", "L": "G7", "X": "G8"},
    "PM": {"H": "G9", "M": "G10", "L": "G11", "X": "G12"},
}

UGV_NAMES = ["Explore-Inner", "Explore-Mid", "Explore-Outer"]
UAV_NAMES = ["Drone-3m", "Drone-6m", "Drone-9m"]
UGV_COLORS = {
    "Explore-Inner": "#117733",
    "Explore-Mid": "#dd7722",
    "Explore-Outer": "#4455aa",
}
UAV_COLORS = {
    "Drone-3m": "#1f77b4",
    "Drone-6m": "#ff7f0e",
    "Drone-9m": "#2ca02c",
}
STATION_COLORS = {
    "S0": "#1f77b4",
    "S1": "#d62728",
    "S2": "#2ca02c",
}
# Display labels: internal CSV keys → paper-friendly names
DISPLAY_NAME = {
    "Explore-Inner": "Explore-1",
    "Explore-Mid":   "Explore-2",
    "Explore-Outer": "Explore-3",
    "Drone-3m":      "Drone-1",
    "Drone-6m":      "Drone-2",
    "Drone-9m":      "Drone-3",
    "S0": "S1",
    "S1": "S2",
    "S2": "S3",
}

PANEL_SPECS = [
    {
        "key": "ugv_trajectory",
        "kind": "trajectory",
        "title": "UGV Trajectories",
        "file_stem": "ugv_trajectory_grid",
        "entity_names": UGV_NAMES,
        "colors": UGV_COLORS,
    },
    {
        "key": "ugv_soc",
        "kind": "soc",
        "title": "UGV State of Charge",
        "file_stem": "ugv_soc_grid",
        "entity_names": UGV_NAMES,
        "colors": UGV_COLORS,
        "ylabel": "UGV SoC",
        "ylim": (0.0, 1.0),
    },
    {
        "key": "uav_trajectory",
        "kind": "trajectory",
        "title": "UAV Trajectories",
        "file_stem": "uav_trajectory_grid",
        "entity_names": UAV_NAMES,
        "colors": UAV_COLORS,
    },
    {
        "key": "uav_soc",
        "kind": "soc",
        "title": "UAV State of Charge",
        "file_stem": "uav_soc_grid",
        "entity_names": UAV_NAMES,
        "colors": UAV_COLORS,
        "ylabel": "UAV SoC",
        "ylim": (0.0, 1.0),
    },
    {
        "key": "station_soc",
        "kind": "station_soc",
        "title": "Charging Station State of Charge",
        "file_stem": "station_soc_grid",
        "entity_names": ["S0", "S1", "S2"],
        "colors": STATION_COLORS,
        "ylabel": "Station SoC",
        "ylim": (0.0, 1.0),
    },
    {
        "key": "station_queue",
        "kind": "station_queue",
        "title": "Charging Station Queue Occupancy",
        "file_stem": "station_queue_grid",
        "entity_names": ["S0", "S1", "S2"],
        "colors": STATION_COLORS,
        "ylabel": "Queue Occupancy",
    },
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Build publication-style grouped validation figures from the 12 manual-scene experiment folders."
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Root directory containing G1-G12 validation folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where grouped figures will be written.",
    )
    return parser.parse_args()


def _configure_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman No9 L",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.size": 17,
            "font.weight": "bold",
            "figure.dpi": 300,
            "axes.titlesize": 17,
            "axes.labelsize": 17,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_FONT_SIZE + 1,
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.pad_inches": 0.05,
            "grid.alpha": 0.28,
        }
    )


def _discover_groups(results_dir: Path):
    groups = {period: {} for period in PERIOD_META}
    pattern = re.compile(r"^(G\d+)_(AM|NN|PM)_([HMLX])$")
    for child in results_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        group_id, period, init_tag = match.groups()
        groups[period][init_tag] = {"group_id": group_id, "path": child}

    missing = []
    for period in PERIOD_META:
        for init_tag in INIT_ORDER:
            if init_tag not in groups[period]:
                missing.append(f"{period}-{init_tag}")
    if missing:
        raise FileNotFoundError(f"Missing validation folders for: {', '.join(missing)}")
    return groups


def _load_case_tables(groups):
    bundle = {period: {} for period in PERIOD_META}
    for period, init_map in groups.items():
        for init_tag, item in init_map.items():
            case_tables = {}
            for case_folder in CASE_META:
                case_dir = item["path"] / case_folder
                case_tables[case_folder] = {
                    "vehicle_soc": pd.read_csv(case_dir / "vehicle_soc_history.csv"),
                    "station_history": pd.read_csv(case_dir / "station_history.csv"),
                    "positions": pd.read_csv(case_dir / "mix_rover_positions.positions.csv"),
                }
            bundle[period][init_tag] = {
                "group_id": item["group_id"],
                "path": item["path"],
                "cases": case_tables,
            }
    return bundle


def _init_title(period, init_tag, group_id):
    label = INIT_LABELS.get(init_tag, init_tag)
    expected = INIT_GROUP_ORDER.get(period, {}).get(init_tag)
    if expected is not None and expected != group_id:
        return f"{label} ({group_id})"
    return f"{label}"


def _legend_handles_for_trajectory(entity_names, palette, scene):
    handles = [Line2D([0], [0], color=palette[name], linewidth=2.4, label=DISPLAY_NAME.get(name, name)) for name in entity_names]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markersize=12,
                markerfacecolor="#e67e22",
                markeredgecolor="#8a4700",
                label="Station",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="#222222",
                label="Start",
            ),
            Line2D([0], [0], color="black", linewidth=1.8, label="Terrain boundary"),
        ]
    )
    if len(scene.get("green_regions", [])) > 0:
        handles.append(Patch(facecolor="#9ccf7b", edgecolor="#5b8a44", alpha=0.55, label="Grass"))
    if len(scene.get("no_go_regions", [])) > 0:
        handles.append(Patch(facecolor="#d08873", edgecolor="#8f4a3b", alpha=0.50, label="No-go region"))
    if len(scene.get("drone_takeoff_landing_zones", [])) > 0:
        handles.append(Line2D([0], [0], color="#2c5282", linestyle="--", linewidth=1.4, label="Drone zone"))
    return handles


def _legend_handles_for_lines(entity_names, palette):
    return [Line2D([0], [0], color=palette[name], linewidth=2.4, label=DISPLAY_NAME.get(name, name)) for name in entity_names]


def _place_top_legend(fig, handles, ncol, anchor_y):
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower left",
        bbox_to_anchor=(FIG_LEFT, anchor_y, FIG_RIGHT - FIG_LEFT, 0.001),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        mode="expand",
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.2,
        handlelength=2.4,
        handletextpad=0.6,
        labelspacing=0.8,
        prop={"weight": "bold", "size": LEGEND_FONT_SIZE},
    )


def _draw_clean_scene(ax, scene):
    terrain = np.asarray(scene.get("terrain_boundary", []), dtype=float)
    if len(terrain) >= 3:
        ax.fill(terrain[:, 0], terrain[:, 1], color="#f0ece3", alpha=0.60, zorder=0)
    if len(terrain) >= 2:
        closed = np.vstack([terrain, terrain[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1.8, zorder=2)

    for region in scene.get("green_regions", []):
        poly = np.asarray(region, dtype=float)
        if len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], color="#9ccf7b", alpha=0.55, edgecolor="#5b8a44", linewidth=1.0, zorder=1)

    for region in scene.get("no_go_regions", []):
        poly = np.asarray(region, dtype=float)
        if len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], color="#d08873", alpha=0.50, edgecolor="#8f4a3b", linewidth=1.1, zorder=1)

    for zone in scene.get("drone_takeoff_landing_zones", []):
        poly = np.asarray(zone.get("points", []), dtype=float)
        if len(poly) >= 3:
            closed = np.vstack([poly, poly[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="#2c5282", linewidth=1.1, linestyle="--", zorder=3)


def _prepare_trajectory_axis(ax, scene, stations):
    _draw_clean_scene(ax, scene)
    for station in stations:
        x_pos, y_pos = station["pos"]
        ax.scatter([x_pos], [y_pos], marker="*", s=120, color="#e67e22", edgecolor="#8a4700", zorder=5)
        ax.text(x_pos + 0.18, y_pos + 0.18, f"S{int(station['id']) + 1}", fontsize=7.5, color="#7a3d00", zorder=6)

    x_min, x_max, y_min, y_max = _scene_bounds(scene)
    pad_x = max((x_max - x_min) * 0.04, 0.5)
    pad_y = max((y_max - y_min) * 0.04, 0.5)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.22)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.0, length=5, pad=3.5)


def _plot_trajectory_panel(out_path, period, case_folder, spec, bundle, scene, stations):
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.9), sharex=True, sharey=True)
    axes = axes.flatten()
    palette = spec["colors"]

    for idx, init_tag in enumerate(INIT_ORDER):
        ax = axes[idx]
        item = bundle[period][init_tag]
        positions = item["cases"][case_folder]["positions"]
        _prepare_trajectory_axis(ax, scene, stations)

        for name in spec["entity_names"]:
            x_key = f"{name}_x"
            y_key = f"{name}_y"
            ax.plot(
                positions[x_key].to_numpy(dtype=float),
                positions[y_key].to_numpy(dtype=float),
                color=palette[name],
                linewidth=1.9,
                alpha=0.96,
                solid_capstyle="round",
            )
            ax.scatter(
                [float(positions[x_key].iloc[0])],
                [float(positions[y_key].iloc[0])],
                s=18,
                marker="o",
                facecolor="white",
                edgecolor=palette[name],
                linewidth=0.9,
                zorder=7,
            )

        ax.set_title(_init_title(period, init_tag, item["group_id"]))
        ax.label_outer()

    handles = _legend_handles_for_trajectory(spec["entity_names"], palette, scene)
    fig.suptitle(
        f"{PERIOD_META[period]['label']} | {CASE_META[case_folder]['label']} | {spec['title']}",
        y=0.985,
        fontsize=20,
        fontweight="bold",
    )
    fig.supxlabel("X (m)", y=TRAJECTORY_SUPXLABEL_Y, fontsize=LEGEND_FONT_SIZE, fontweight="bold")
    fig.supylabel("Y (m)", x=SUPYLABEL_X, fontsize=LEGEND_FONT_SIZE, fontweight="bold")
    _place_top_legend(fig, handles, ncol=min(5, len(handles)), anchor_y=0.845)
    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=TRAJECTORY_PANEL_BOTTOM, top=0.76, wspace=0.10, hspace=0.12)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _time_minutes(frame: pd.DataFrame):
    if "time_minutes" in frame.columns:
        return frame["time_minutes"].to_numpy(dtype=float)
    if "time_hours" in frame.columns:
        return frame["time_hours"].to_numpy(dtype=float) * 60.0
    raise KeyError("Expected time_minutes or time_hours column in input frame.")


def _plot_line_panel(out_path, period, case_folder, spec, bundle):
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.9), sharex=True, sharey=True)
    axes = axes.flatten()
    palette = spec["colors"]
    queue_max = 0.0

    if spec["kind"] == "station_queue":
        for init_tag in INIT_ORDER:
            station_history = bundle[period][init_tag]["cases"][case_folder]["station_history"]
            for station_name in spec["entity_names"]:
                queue_max = max(queue_max, float(station_history[f"{station_name}_queue"].max()))

    for idx, init_tag in enumerate(INIT_ORDER):
        ax = axes[idx]
        item = bundle[period][init_tag]

        if spec["kind"] == "soc":
            frame = item["cases"][case_folder]["vehicle_soc"]
            t_vals = _time_minutes(frame)
            for name in spec["entity_names"]:
                ax.plot(t_vals, frame[f"{name}_soc"].to_numpy(dtype=float), color=palette[name], linewidth=2.0)
        elif spec["kind"] == "station_soc":
            frame = item["cases"][case_folder]["station_history"]
            t_vals = _time_minutes(frame)
            for name in spec["entity_names"]:
                ax.plot(t_vals, frame[f"{name}_soc"].to_numpy(dtype=float), color=palette[name], linewidth=2.0)
        elif spec["kind"] == "station_queue":
            frame = item["cases"][case_folder]["station_history"]
            t_vals = _time_minutes(frame)
            for name in spec["entity_names"]:
                ax.step(t_vals, frame[f"{name}_queue"].to_numpy(dtype=float), where="post", color=palette[name], linewidth=2.0)
        else:
            raise ValueError(f"Unsupported line panel kind: {spec['kind']}")

        ax.set_title(_init_title(period, init_tag, item["group_id"]))
        ax.grid(True, linestyle="--", alpha=0.30)
        ax.margins(x=0.02)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.0, length=5, pad=3.5)
        ax.label_outer()

    if "ylim" in spec:
        axes[0].set_ylim(*spec["ylim"])
    elif spec["kind"] == "station_queue":
        queue_ceiling = max(1, int(np.ceil(queue_max)))
        axes[0].set_ylim(-0.02, queue_ceiling + 0.15)
        for ax in axes:
            ax.set_yticks(np.arange(0, queue_ceiling + 1, 1))

    handles = _legend_handles_for_lines(spec["entity_names"], palette)
    fig.suptitle(
        f"{PERIOD_META[period]['label']} | {CASE_META[case_folder]['label']} | {spec['title']}",
        y=0.985,
        fontsize=20,
        fontweight="bold",
    )
    fig.supxlabel("Time (min)", y=LINE_SUPXLABEL_Y, fontsize=LEGEND_FONT_SIZE, fontweight="bold")
    fig.supylabel(spec["ylabel"], x=SUPYLABEL_X, fontsize=LEGEND_FONT_SIZE, fontweight="bold")
    _place_top_legend(fig, handles, ncol=len(handles), anchor_y=0.87)
    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=LINE_PANEL_BOTTOM, top=0.80, wspace=0.10, hspace=0.16)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_panels(results_dir: Path, output_dir: Path):
    groups = _discover_groups(results_dir)
    bundle = _load_case_tables(groups)
    stations, _explore_rovers, _transport_rovers, scene = build_manual_mixed_scenario()

    manifest = {"output_dir": str(output_dir.resolve()), "figures": []}
    for period, period_meta in PERIOD_META.items():
        period_dir = output_dir / period_meta["folder"]
        period_dir.mkdir(parents=True, exist_ok=True)
        for case_folder, case_meta in CASE_META.items():
            for spec in PANEL_SPECS:
                out_path = period_dir / f"{case_meta['stem']}_{spec['file_stem']}.png"
                if spec["kind"] == "trajectory":
                    _plot_trajectory_panel(out_path, period, case_folder, spec, bundle, scene, stations)
                else:
                    _plot_line_panel(out_path, period, case_folder, spec, bundle)
                manifest["figures"].append(
                    {
                        "period": period,
                        "period_label": period_meta["label"],
                        "case": case_folder,
                        "case_label": case_meta["label"],
                        "metric": spec["key"],
                        "title": spec["title"],
                        "path": str(out_path.resolve()),
                    }
                )

    manifest_path = output_dir / "grouped_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path, len(manifest["figures"])


def main():
    args = _parse_args()
    _configure_style()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path, figure_count = _build_panels(results_dir, output_dir)
    print(f"Saved {figure_count} grouped figures under {output_dir}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
