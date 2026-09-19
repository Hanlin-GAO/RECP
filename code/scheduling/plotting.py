import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle

FIELD_X_MIN = 0.0
FIELD_Y_MIN = 0.0
FIELD_X_MAX = 105.0
FIELD_Y_MAX = 68.0


def plot_soc(out_path, t, soc_mat, names, soc_min=0.2, soc_max=0.9, title="SOC", x_label="Time"):
    plt.figure(figsize=(10, 4))
    for k, nm in enumerate(names):
        plt.plot(t, soc_mat[:, k], label=f"{nm}")
    plt.axhline(soc_min, linestyle="--", linewidth=1.2, label="SOC_min")
    plt.axhline(soc_max, linestyle="--", linewidth=1.2, label="SOC_max")
    plt.xlabel(x_label)
    plt.ylabel("SOC")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_power_breakdown_by_station(out_path, t, P_load, P_pv, P_station_used, charge_station_id, names, stations,
                                   title="Power Breakdown (Station used colored by station)", x_label="Time"):
    station_ids = [int(st["id"]) for st in stations]
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 9.0), sharex=True)
    if len(names) == 1:
        axes = [axes]

    for k, nm in enumerate(names):
        ax = axes[k]
        ax.plot(t, P_load[:, k], label="Load (kW)")
        if np.max(np.abs(P_pv[:, k])) > 1e-6:
            ax.plot(t, P_pv[:, k], label="PV (kW)")

        for sid in station_ids:
            mask = (charge_station_id[:, k] == sid).astype(float)
            y = P_station_used[:, k] * mask
            if np.max(y) > 1e-6:
                ax.plot(t, y, label=f"S{sid} used (kW)")

        ax.set_ylabel("kW")
        ax.set_title(nm)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")

    axes[-1].set_xlabel(x_label)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_station_soc_and_queue(out_soc, out_queue, t, station_soc, station_queue, stations, x_label="Time"):
    soc_min = min(float(st.get("soc_min_station", 0.15)) for st in stations)
    soc_max = max(float(st.get("soc_max_station", 0.95)) for st in stations)

    # SoC
    plt.figure(figsize=(10, 4))
    for i, st in enumerate(stations):
        plt.plot(t, station_soc[:, i], label=f"S{st['id']} SoC")
    plt.axhline(soc_min, linestyle="--", linewidth=1.2, label="SoC_min")
    plt.axhline(soc_max, linestyle="--", linewidth=1.2, label="SoC_max")
    plt.xlabel(x_label)
    plt.ylabel("Station SoC")
    plt.title("Charging Station Battery SoC")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_soc, dpi=300)
    plt.close()

    # Queue (3D lines, NO legend)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    t = np.asarray(t).reshape(-1)
    station_queue = np.asarray(station_queue, dtype=float)
    station_ids = [int(st["id"]) for st in stations]

    max_q = float(np.max(station_queue)) if station_queue.size > 0 else 0.0
    zmax = max(1.0, np.ceil(max_q + 0.5))

    fig = plt.figure(figsize=(11.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    for i, sid in enumerate(station_ids):
        y = np.full_like(t, float(sid), dtype=float)
        z = station_queue[:, i]
        ax.plot(t, y, z, linewidth=2.2)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Station ID")
    ax.set_zlabel("Occupancy (#rovers incl. charging)")
    ax.set_title("Charging Station Occupancy (3D)")
    ax.set_yticks(station_ids)
    ax.set_zlim(0.0, zmax)
    ax.set_zticks(np.arange(0, int(np.ceil(zmax)) + 1, 1))
    ax.view_init(elev=22, azim=-55)

    plt.tight_layout()
    plt.savefig(out_queue, dpi=300)
    plt.close(fig)


def _draw_soccer_field(ax):
    rect = Rectangle(
        (FIELD_X_MIN, FIELD_Y_MIN),
        FIELD_X_MAX - FIELD_X_MIN,
        FIELD_Y_MAX - FIELD_Y_MIN,
        fill=False,
        linewidth=2.0
    )
    ax.add_patch(rect)


def _scene_bounds(scene):
    if scene is None:
        return FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX

    bounds = scene.get("field_bounds", {})
    return (
        float(bounds.get("x_min", FIELD_X_MIN)),
        float(bounds.get("x_max", FIELD_X_MAX)),
        float(bounds.get("y_min", FIELD_Y_MIN)),
        float(bounds.get("y_max", FIELD_Y_MAX)),
    )


def _draw_scene(ax, scene=None):
    if scene is None:
        _draw_soccer_field(ax)
        return

    terrain = np.asarray(scene.get("terrain_boundary", []), dtype=float)

    # Fill terrain with a neutral field colour (beige)
    if len(terrain) >= 3:
        ax.fill(terrain[:, 0], terrain[:, 1], color="#f0ece3", alpha=0.60, zorder=0)

    # Terrain boundary outline
    if len(terrain) >= 2:
        closed = np.vstack([terrain, terrain[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", linewidth=2.0, zorder=2)

    # Detected green grass patches (from HSV segmentation)
    for region in scene.get("green_regions", []):
        poly = np.asarray(region, dtype=float)
        if len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], color="#9ccf7b", alpha=0.55, edgecolor="#5b8a44", linewidth=1.0, zorder=1)

    # No-go obstacles (drawn on top of green fill)
    for region in scene.get("no_go_regions", []):
        poly = np.asarray(region, dtype=float)
        if len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], color="#d08873", alpha=0.50, edgecolor="#8f4a3b", linewidth=1.2, zorder=1)

    for zone in scene.get("drone_takeoff_landing_zones", []):
        poly = np.asarray(zone.get("points", []), dtype=float)
        if len(poly) < 3:
            continue
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#2c5282", linewidth=1.4, linestyle="--", zorder=3)
        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        label = zone.get("label")
        if label:
            ax.text(cx, cy, label, fontsize=7.5, color="#1f3c5b", ha="center", va="center", zorder=4)


def plot_traj(out_path, xs, ys, rover_names, stations, *,
             rover_start_pos=None,
             cargo_pairs=None,
             scene=None,
             title="Trajectories (105m × 68m soccer field)"):
    fig = plt.figure(figsize=(10.2, 6.6))
    ax = plt.gca()

    _draw_scene(ax, scene)

    for st in stations:
        x0, y0 = st["pos"]
        ax.scatter([x0], [y0], marker="*", s=240)
        station_radius = float(st.get("approach_radius_m", 0.0))
        if station_radius > 0.0:
            ax.add_patch(plt.Circle((x0, y0), station_radius, fill=False, linestyle="--", linewidth=1.0, color="#4d4d4d", zorder=3))
        ax.text(x0 + 1.0, y0 + 1.0, f"S{st['id']}", fontsize=10)

    if cargo_pairs is not None:
        cmap = plt.get_cmap("tab10")
        for cp in cargo_pairs:
            cid = int(cp["cargo_id"])
            col = cmap(cid % 10)
            px, py = cp["pickup"]
            dx, dy = cp["dropoff"]
            ax.scatter([px], [py], marker="^", s=80, color=col)
            ax.scatter([dx], [dy], marker="s", s=80, color=col)

    for k, nm in enumerate(rover_names):
        ax.plot(xs[:, k], ys[:, k], linewidth=2.0, label=nm)

    if rover_start_pos is not None:
        for sp in rover_start_pos:
            sp = np.asarray(sp, dtype=float)
            ax.scatter([sp[0]], [sp[1]], marker="o", s=45)

    x_min, x_max, y_min, y_max = _scene_bounds(scene)
    pad_x = max((x_max - x_min) * 0.04, 0.5)
    pad_y = max((y_max - y_min) * 0.04, 0.5)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)

    handles = [
        Line2D([0], [0], marker="*", linestyle="None", markersize=12, label="Station"),
        Line2D([0], [0], color="#4d4d4d", linestyle="--", linewidth=1.0, label="Station boundary"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=8, label="Vehicle start"),
    ]
    if scene is not None:
        handles.append(Line2D([0], [0], color="black", linewidth=2.0, label="Terrain boundary"))
        if len(scene.get("green_regions", [])) > 0:
            handles.append(Patch(facecolor="#9ccf7b", edgecolor="#5b8a44", alpha=0.20, label="Grass (passable)"))
        if len(scene.get("no_go_regions", [])) > 0:
            handles.append(Patch(facecolor="#d08873", edgecolor="#8f4a3b", alpha=0.40, label="Manual no-go region"))
    if cargo_pairs is not None:
        handles += [
            Line2D([0], [0], marker="^", linestyle="None", markersize=8, label="Pickup"),
            Line2D([0], [0], marker="s", linestyle="None", markersize=8, label="Dropoff"),
        ]

    h2, l2 = ax.get_legend_handles_labels()
    ax.legend(handles=handles + h2, labels=[h.get_label() for h in handles] + l2, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_scene_boundary(out_path, stations, *, scene=None, title=None):
    fig = plt.figure(figsize=(10.2, 6.6))
    ax = plt.gca()

    _draw_scene(ax, scene)

    for st in stations:
        x0, y0 = st["pos"]
        ax.scatter([x0], [y0], marker="*", s=240, color="#e67e22", zorder=5)
        ax.text(x0 + 0.15, y0 + 0.15, f"S{st['id']}", fontsize=9, color="#9c4f0d", zorder=6)

    x_min, x_max, y_min, y_max = _scene_bounds(scene)
    pad_x = max((x_max - x_min) * 0.04, 0.5)
    pad_y = max((y_max - y_min) * 0.04, 0.5)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_transport_tasks_colored(out_path, x, y, job_id, cargo_id, cargo_pairs, stations, *, scene=None,
                                 rover_name="Transport"):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    job_id = np.asarray(job_id).reshape(-1)
    cargo_id = np.asarray(cargo_id).reshape(-1)

    pts = np.stack([x, y], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    cmap_job = plt.get_cmap("tab20")
    seg_colors = [cmap_job(int(j) % 20) for j in job_id[:-1]]
    lc = LineCollection(segs, colors=seg_colors, linewidths=2.5)

    fig = plt.figure(figsize=(10.2, 6.6))
    ax = plt.gca()
    _draw_scene(ax, scene)
    ax.add_collection(lc)

    for st in stations:
        x0, y0 = st["pos"]
        ax.scatter([x0], [y0], marker="*", s=240)
        ax.text(x0 + 1.0, y0 + 1.0, f"S{st['id']}", fontsize=10)

    cmap_cargo = plt.get_cmap("tab10")
    for cp in cargo_pairs:
        cid = int(cp["cargo_id"])
        col = cmap_cargo(cid % 10)
        px, py = cp["pickup"]
        dx, dy = cp["dropoff"]
        ax.scatter([px], [py], marker="^", s=95, color=col)
        ax.scatter([dx], [dy], marker="s", s=95, color=col)

    ax.scatter([x[0]], [y[0]], marker="o", s=60)

    x_min, x_max, y_min, y_max = _scene_bounds(scene)
    pad_x = max((x_max - x_min) * 0.04, 0.5)
    pad_y = max((y_max - y_min) * 0.04, 0.5)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{rover_name} mission segments (colored by job_id)")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)