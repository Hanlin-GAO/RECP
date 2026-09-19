"""
Publication-quality visualisation — Nature Communications style.

Generates three figures per station:
  1. comparison_normal.png   – Normal scenario dashboard
  2. comparison_extreme.png  – Extreme scenario dashboard
  3. miss_ratio_sweep.png    – Robustness curve under sensor data loss
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════
# Global rc – Nature Communications house style
# ═══════════════════════════════════════════════════════════════════════════
def _setup_rc():
    """Apply publication rcParams once."""
    mpl.rcParams.update({
        # Font – Times New Roman (serif) for publication
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          22,
        "axes.titlesize":     26,
        "axes.labelsize":     24,
        "xtick.labelsize":    20,
        "ytick.labelsize":    20,
        "legend.fontsize":    20,
        "legend.title_fontsize": 22,
        # Axes – clean, minimal
        "axes.linewidth":     1.0,
        "axes.unicode_minus": False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # Ticks
        "xtick.major.width":  1.0,
        "ytick.major.width":  1.0,
        "xtick.minor.width":  0.6,
        "ytick.minor.width":  0.6,
        "xtick.major.size":   5,
        "ytick.major.size":   5,
        "xtick.minor.size":   2.5,
        "ytick.minor.size":   2.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Lines
        "lines.linewidth":    1.4,
        "lines.markersize":   6,
        # Grid off by default; added explicitly where needed
        "axes.grid":          False,
        # Save
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        # Figure
        "figure.dpi":         150,
        "figure.facecolor":   "white",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Palette — muted, colour-blind safe, print-friendly
# ═══════════════════════════════════════════════════════════════════════════
from forecast_paths import PV_OUTPUT
OUTPUT_DIR = PV_OUTPUT
STATIONS = ["Inverter_1", "Inverter_2", "Inverter_3"]

MODELS = [
    {"tag": "dnn",   "label": "DNN",         "color": "#4daf4a", "marker": "o"},
    {"tag": "lstm",  "label": "LSTM",        "color": "#377eb8", "marker": "s"},
    {"tag": "gru",   "label": "GRU",         "color": "#984ea3", "marker": "D"},
    {"tag": "tcn",   "label": "TCN",         "color": "#e41a1c", "marker": "^"},
    {"tag": "trans", "label": "Transformer", "color": "#17becf", "marker": "v"},
    {"tag": "pinn",  "label": "PINN",        "color": "#ff7f00", "marker": "P"},
]
N = len(MODELS)
_TAGS   = [m["tag"]   for m in MODELS]
_LABELS = [m["label"] for m in MODELS]
_COLORS = [m["color"] for m in MODELS]


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════
def _panel_label(ax, label, x=-0.08, y=1.10):
    """Bold a/b/c panel label — Nature convention."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=28, fontweight="bold", va="top", ha="right")


def _light_grid(ax, axis="y"):
    ax.grid(True, axis=axis, linewidth=0.3, alpha=0.35, color="#cccccc")
    ax.set_axisbelow(True)


def _minor_ticks(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def _bar_val(ax, bars, fmt="{:.0f}", fs=12, pad=0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + pad,
                fmt.format(h), ha="center", va="bottom", fontsize=fs)


# ═══════════════════════════════════════════════════════════════════════════
# Panel drawing functions
# ═══════════════════════════════════════════════════════════════════════════

# ---- Training curves ----
def _panel_loss(ax, hists):
    for m in MODELS:
        h = hists[m["tag"]]
        ax.plot(h["epoch"], h["train_data"], color=m["color"],
                lw=1.0, alpha=0.85, label=f'{m["label"]} train')
        ax.plot(h["epoch"], h["val_data"], color=m["color"],
                lw=1.0, ls="--", alpha=0.85, label=f'{m["label"]} val')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalised)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    _light_grid(ax); _minor_ticks(ax)
    ax.legend(ncol=2, frameon=False, loc="upper right",
              borderaxespad=0.3, handlelength=1.5, columnspacing=0.8)


def _panel_mae_curve(ax, hists):
    for m in MODELS:
        h = hists[m["tag"]]
        ax.plot(h["epoch"], h["train_mae"], color=m["color"],
                lw=1.0, alpha=0.85, label=f'{m["label"]} train')
        ax.plot(h["epoch"], h["val_mae"], color=m["color"],
                lw=1.0, ls="--", alpha=0.85, label=f'{m["label"]} val')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (W)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    _light_grid(ax); _minor_ticks(ax)
    ax.legend(ncol=2, frameon=False, loc="upper right",
              borderaxespad=0.3, handlelength=1.5, columnspacing=0.8)


# ---- Prediction time-series ----
def _panel_pred(ax, preds, n_show, subtitle=""):
    ref = preds[MODELS[0]["tag"]]
    n = min(n_show, len(ref))
    t = ref["Time"][:n]
    ax.plot(t, ref["True_Power"][:n], color="#222222", lw=0.8,
            alpha=0.85, label="Ground truth", zorder=10)
    for m in MODELS:
        p = preds[m["tag"]]
        ax.plot(t, p["Pred_Power"][:n], color=m["color"],
                lw=0.6, alpha=0.7, label=m["label"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Power (W)")
    if subtitle:
        ax.set_title(subtitle, fontsize=16, pad=4)
    _light_grid(ax)
    ax.legend(ncol=N + 1, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=1.0)
    ax.tick_params(axis="x", rotation=15)


# ---- Scatter (true vs pred) ----
def _panel_scatter(ax, preds, tag, label, color):
    p = preds[tag]
    t_arr = p["True_Power"].values
    p_arr = p["Pred_Power"].values
    ax.scatter(t_arr, p_arr, s=2, alpha=0.18, color=color,
               edgecolors="none", rasterized=True)
    lo = min(t_arr.min(), p_arr.min(), 0)
    hi = max(t_arr.max(), p_arr.max())
    ax.plot([lo, hi], [lo, hi], color="#444444", lw=0.6,
            ls="--", alpha=0.5, zorder=5)
    ss_res = np.sum((t_arr - p_arr) ** 2)
    ss_tot = np.sum((t_arr - t_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0
    ax.text(0.05, 0.93, f"$R^2 = {r2:.3f}$", transform=ax.transAxes,
            fontsize=14, va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("True power (W)")
    ax.set_ylabel("Predicted power (W)")
    ax.set_title(label, fontsize=16, pad=3)
    ax.set_aspect("equal", adjustable="datalim")
    _minor_ticks(ax)


# ---- Error distribution ----
def _panel_error_kde(ax, preds):
    for m in MODELS:
        err = preds[m["tag"]]["Pred_Power"].values - preds[m["tag"]]["True_Power"].values
        try:
            kde = stats.gaussian_kde(err, bw_method=0.15)
            xs = np.linspace(err.min(), err.max(), 300)
            ax.plot(xs, kde(xs), color=m["color"], lw=1.0, label=m["label"])
            ax.fill_between(xs, kde(xs), alpha=0.07, color=m["color"])
        except Exception:
            ax.hist(err, bins=60, density=True, alpha=0.35,
                    color=m["color"], label=m["label"], edgecolor="none")
    ax.axvline(0, color="#444444", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Prediction error (W)")
    ax.set_ylabel("Density")
    _light_grid(ax)
    ax.legend(frameon=False, handlelength=1.5)


# ---- Grouped bar MAE/RMSE ----
def _panel_bar_mae_rmse(ax, met):
    x = np.arange(2)
    w = 0.14
    for i, m in enumerate(MODELS):
        vals = [met[m["tag"]]["mae"], met[m["tag"]]["rmse"]]
        bars = ax.bar(x + (i - (N - 1) / 2) * w, vals, w, label=m["label"],
                      color=m["color"], edgecolor="white", linewidth=0.3)
        _bar_val(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAE", "RMSE"])
    ax.set_ylabel("Value (W)")
    _light_grid(ax)
    ax.legend(frameon=False, ncol=N, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), handlelength=1.2, columnspacing=0.6)


# ---- R² bar ----
def _panel_bar_r2(ax, met, ylim_lo=None):
    r2v = [met[m["tag"]]["r2"] for m in MODELS]
    bars = ax.bar(_LABELS, r2v, width=0.52,
                  color=_COLORS, edgecolor="white", linewidth=0.4)
    _bar_val(ax, bars, fmt="{:.3f}", fs=13)
    ax.set_ylabel("$R^2$")
    lo = min(0, min(r2v) - 0.05) if ylim_lo is None else ylim_lo
    ax.set_ylim(lo, 1.02)
    _light_grid(ax)


# ---- Residual ----
def _panel_residual(ax, preds, tag, color, n_show=1000):
    p = preds[tag]
    err = p["Pred_Power"].values - p["True_Power"].values
    n = min(n_show, len(err))
    ax.fill_between(range(n), err[:n], 0, color=color, alpha=0.22, linewidth=0)
    ax.plot(range(n), err[:n], color=color, lw=0.35, alpha=0.7)
    ax.axhline(0, color="#444444", lw=0.5, ls="-", alpha=0.4)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Error (W)")
    _light_grid(ax)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Normal Scenario
# ═══════════════════════════════════════════════════════════════════════════
def plot_normal(metrics, hists, preds, imp, fig_path):
    fig = plt.figure(figsize=(24, 24))
    gs = gridspec.GridSpec(5, N, figure=fig, hspace=0.65, wspace=0.55,
                           height_ratios=[1, 1.2, 0.9, 1, 0.9])

    # Row 0 — training curves
    ax0 = fig.add_subplot(gs[0, :3])
    _panel_loss(ax0, hists);  _panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 3:])
    _panel_mae_curve(ax1, hists);  _panel_label(ax1, "b")

    # Row 1 — prediction curve
    ax2 = fig.add_subplot(gs[1, :])
    _panel_pred(ax2, preds, 1000);  _panel_label(ax2, "c")

    # Row 2 — scatter
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[2, i])
        _panel_scatter(ax, preds, m["tag"], m["label"], m["color"])
        if i == 0: _panel_label(ax, "d")

    # Row 3 — error dist + bars
    ax3 = fig.add_subplot(gs[3, :2])
    _panel_error_kde(ax3, preds);  _panel_label(ax3, "e")

    ax4 = fig.add_subplot(gs[3, 2:N-1])
    _panel_bar_mae_rmse(ax4, metrics);  _panel_label(ax4, "f")

    ax5 = fig.add_subplot(gs[3, N-1])
    _panel_bar_r2(ax5, metrics);  _panel_label(ax5, "g")

    # Row 4 — residuals
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[4, i])
        _panel_residual(ax, preds, m["tag"], m["color"])
        ax.set_title(m["label"], fontsize=14, pad=3)
        if i == 0: _panel_label(ax, "h")

    pv_d = imp.get("pinn_vs_dnn_mae_pct", 0)
    pv_l = imp.get("pinn_vs_lstm_mae_pct", 0)
    fig.suptitle(
        f"Normal scenario  |  PINN vs DNN  MAE {pv_d:+.1f}%   "
        f"PINN vs LSTM  MAE {pv_l:+.1f}%",
        fontsize=20, fontweight="bold", y=1.01)

    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Extreme Scenario
# ═══════════════════════════════════════════════════════════════════════════
def plot_extreme(metrics, ext_met, preds_ext, ext_imp, hists, fig_path):
    fig = plt.figure(figsize=(24, 23))
    gs = gridspec.GridSpec(5, N, figure=fig, hspace=0.65, wspace=0.55,
                           height_ratios=[1, 1.2, 0.9, 1, 0.9])

    ax0 = fig.add_subplot(gs[0, :3])
    _panel_loss(ax0, hists);  _panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 3:])
    _panel_mae_curve(ax1, hists);  _panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[1, :])
    _panel_pred(ax2, preds_ext, 800,
                "Extreme weather + 30% sensor data missing")
    _panel_label(ax2, "c")

    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[2, i])
        _panel_scatter(ax, preds_ext, m["tag"], m["label"], m["color"])
        if i == 0: _panel_label(ax, "d")

    ax3 = fig.add_subplot(gs[3, :2])
    _panel_error_kde(ax3, preds_ext);  _panel_label(ax3, "e")

    ax4 = fig.add_subplot(gs[3, 2:N-1])
    _panel_bar_mae_rmse(ax4, ext_met);  _panel_label(ax4, "f")

    ax5 = fig.add_subplot(gs[3, N-1])
    _panel_bar_r2(ax5, ext_met);  _panel_label(ax5, "g")

    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[4, i])
        _panel_residual(ax, preds_ext, m["tag"], m["color"], n_show=800)
        ax.set_title(m["label"], fontsize=14, pad=3)
        if i == 0: _panel_label(ax, "h")

    pv_d = ext_imp.get("pinn_vs_dnn_mae_pct", 0)
    pv_l = ext_imp.get("pinn_vs_lstm_mae_pct", 0)
    fig.suptitle(
        f"Extreme scenario (30% missing)  |  PINN vs DNN  MAE {pv_d:+.1f}%   "
        f"PINN vs LSTM  MAE {pv_l:+.1f}%",
        fontsize=20, fontweight="bold", y=1.01)

    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Robustness sweep
# ═══════════════════════════════════════════════════════════════════════════
def plot_sweep(metrics, fig_path):
    sweep = metrics.get("miss_sweep", {})
    if not sweep:
        print("  No miss_sweep data — skipped")
        return

    ratios = sorted(int(k) for k in sweep.keys())
    data = {k: {m["tag"]: [] for m in MODELS} for k in ("mae", "rmse", "r2")}
    for r in ratios:
        s = sweep[str(r)]
        for m in MODELS:
            for k in ("mae", "rmse", "r2"):
                data[k][m["tag"]].append(s[m["tag"]][k])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    plt.subplots_adjust(wspace=0.45)

    info = [
        ("mae",  "MAE (W)",  "MAE (W)"),
        ("rmse", "RMSE (W)", "RMSE (W)"),
        ("r2",   "$R^2$",    "$R^2$"),
    ]
    panel_ids = ["a", "b", "c"]

    for idx, (ax, (key, title, ylabel)) in enumerate(zip(axes, info)):
        for m in MODELS:
            ax.plot(ratios, data[key][m["tag"]],
                    marker=m["marker"], ms=4, lw=1.2,
                    color=m["color"], label=m["label"],
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xlabel("Missing data ratio (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=16, pad=4)
        ax.set_xticks(ratios)
        ax.set_xticklabels([str(r) for r in ratios], fontsize=12)
        _light_grid(ax); _minor_ticks(ax)
        _panel_label(ax, panel_ids[idx])
        if key == "r2":
            ax.set_ylim(None, 1.02)
        if idx == 1:
            ax.legend(frameon=False, ncol=N, loc="upper center",
                      bbox_to_anchor=(0.5, 1.24), handlelength=1.5,
                      columnspacing=1.0)

    fig.suptitle(
        "Model robustness under increasing sensor data loss (extreme weather)",
        fontsize=20, fontweight="bold", y=1.08)

    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4a/4b — Standalone scatter plot (5 models in one figure)
# ═══════════════════════════════════════════════════════════════════════════
def plot_scatter_standalone(preds, fig_path, title=""):
    """
    Standalone 1×5 scatter plot (true vs predicted) for all models.
    Units in kW. Only first subplot gets y-label; shared x-label at bottom.
    """
    fig, axes = plt.subplots(1, N, figsize=(5.2 * N, 5.5),
                             constrained_layout=True)

    for i, (ax, m) in enumerate(zip(axes, MODELS)):
        p = preds[m["tag"]]
        t_arr = p["True_Power"].values / 1000.0
        p_arr = p["Pred_Power"].values / 1000.0

        ax.scatter(t_arr, p_arr, s=3, alpha=0.20, color=m["color"],
                   edgecolors="none", rasterized=True)
        lo = min(t_arr.min(), p_arr.min(), 0)
        hi = max(t_arr.max(), p_arr.max())
        ax.plot([lo, hi], [lo, hi], color="#444444", lw=0.8,
                ls="--", alpha=0.6, zorder=5)

        ss_res = np.sum((t_arr - p_arr) ** 2)
        ss_tot = np.sum((t_arr - t_arr.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0
        rmse = np.sqrt(np.mean((t_arr - p_arr) ** 2))
        mae = np.mean(np.abs(t_arr - p_arr))

        ax.text(0.05, 0.95,
                f"$R^2 = {r2:.3f}$\nRMSE = {rmse:.2f} kW\nMAE = {mae:.2f} kW",
                transform=ax.transAxes, fontsize=20, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", alpha=0.9))

        ax.set_title(m["label"], pad=6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("True power (kW)")
        if i == 0:
            ax.set_ylabel("Predicted power (kW)")
        _light_grid(ax, axis="both")
        _minor_ticks(ax)

    if title:
        fig.suptitle(title, fontsize=28, fontweight="bold", y=1.04)

    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5a/5b — Standalone error distribution (5 models in one figure)
# ═══════════════════════════════════════════════════════════════════════════
def plot_error_dist_standalone(preds, fig_path, title=""):
    """
    Single-panel KDE of prediction error for all 5 models.  Units in kW.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)

    for m in MODELS:
        err = (preds[m["tag"]]["Pred_Power"].values
               - preds[m["tag"]]["True_Power"].values) / 1000.0
        try:
            kde = stats.gaussian_kde(err, bw_method=0.15)
            xs = np.linspace(err.min(), err.max(), 400)
            ax.plot(xs, kde(xs), color=m["color"], lw=1.4, label=m["label"])
            ax.fill_between(xs, kde(xs), alpha=0.10, color=m["color"])
        except Exception:
            ax.hist(err, bins=60, density=True, alpha=0.35,
                    color=m["color"], label=m["label"], edgecolor="none")

    ax.axvline(0, color="#444444", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("Prediction error (kW)")
    ax.set_ylabel("Density")
    _light_grid(ax)
    ax.legend(frameon=False, handlelength=1.8)

    if title:
        ax.set_title(title, fontsize=28, fontweight="bold", pad=10)

    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Cross-station box plot (R², RMSE, MAE × Train/Val/Test)
# ═══════════════════════════════════════════════════════════════════════════
def plot_metrics_boxplot(all_metrics, fig_path):
    """
    Box plot of R², RMSE, MAE across stations for each model,
    grouped by Train / Validation / Test split.
    Each box contains one data point per station.
    """
    from matplotlib.patches import Patch

    stations = list(all_metrics.keys())
    if len(stations) < 2:
        print("  Need ≥ 2 stations for boxplot — skipped")
        return

    # Check that train/val keys exist
    sample = all_metrics[stations[0]]
    if "train" not in sample or "val" not in sample:
        print("  No train/val metrics in JSON — skipped boxplot")
        return

    split_info = [
        ("train", "Train",      "#66c2a5"),
        ("val",   "Validation", "#fc8d62"),
        ("test",  "Test",       "#8da0cb"),
    ]
    metric_info = [
        ("r2",   "$R^2$"),
        ("rmse", "RMSE (W)"),
        ("mae",  "MAE (W)"),
    ]
    panel_ids = ["a", "b", "c"]
    n_splits = len(split_info)
    group_width = n_splits + 1.2          # 3 boxes + gap between groups

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))
    plt.subplots_adjust(wspace=0.36)

    for ax_idx, (ax, (mkey, mlabel)) in enumerate(zip(axes, metric_info)):
        box_data = []
        positions = []
        face_colors = []
        group_centers = []

        for i, m in enumerate(MODELS):
            base = i * group_width
            group_centers.append(base + (n_splits - 1) / 2)

            for j, (split_key, _, color) in enumerate(split_info):
                vals = []
                for st in stations:
                    met = all_metrics[st]
                    if split_key == "test":
                        vals.append(met[m["tag"]][mkey])
                    else:
                        vals.append(met[split_key][m["tag"]][mkey])
                positions.append(base + j)
                box_data.append(vals)
                face_colors.append(color)

        bp = ax.boxplot(
            box_data, positions=positions, widths=0.65,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="#333333", linewidth=1.2),
            whiskerprops=dict(linewidth=0.6),
            capprops=dict(linewidth=0.6),
            boxprops=dict(linewidth=0.5),
        )
        for patch, fc in zip(bp["boxes"], face_colors):
            patch.set_facecolor(fc)
            patch.set_alpha(0.55)

        # Overlay individual station points
        rng = np.random.default_rng(42)
        for pos, vals in zip(positions, box_data):
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                pos + jitter, vals, s=14, color="#333333",
                alpha=0.75, zorder=5, edgecolors="white", linewidths=0.3,
            )

        ax.set_xticks(group_centers)
        ax.set_xticklabels([m["label"] for m in MODELS], rotation=30, ha="right")
        ax.set_ylabel(mlabel)
        _light_grid(ax)
        _panel_label(ax, panel_ids[ax_idx])

        if mkey == "r2":
            all_r2 = [v for vlist in box_data for v in vlist]
            lo = min(all_r2) - 0.05
            ax.set_ylim(max(0, lo), 1.02)

    # Shared legend
    legend_patches = [
        Patch(facecolor=c, alpha=0.55, edgecolor="#444444", label=lbl)
        for _, lbl, c in split_info
    ]
    axes[1].legend(
        handles=legend_patches, frameon=False, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, 1.20),
        handlelength=1.5, columnspacing=1.2,
    )

    fig.suptitle(
        "Model performance across stations: Train / Validation / Test",
        fontsize=20, fontweight="bold", y=1.06,
    )

    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    _setup_rc()

    for station_name in STATIONS:
        station_dir = OUTPUT_DIR / station_name
        metrics_file = station_dir / "test_metrics.json"
        if not metrics_file.exists():
            print(f"Skipped {station_name}: {metrics_file} not found")
            continue

        print(f"\n=== {station_name} ===")
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Normal
        hists, preds = {}, {}
        for m in MODELS:
            hists[m["tag"]] = pd.read_csv(station_dir / f"history_{m['tag']}.csv")
            p = pd.read_csv(station_dir / f"test_predictions_{m['tag']}.csv")
            p["Time"] = pd.to_datetime(p["Time"])
            preds[m["tag"]] = p

        plot_normal(metrics, hists, preds,
                    metrics.get("improvement", {}),
                    station_dir / "comparison_normal.png")

        # Standalone normal scatter & error distribution
        plot_scatter_standalone(
            preds, station_dir / "scatter_normal.png",
            title=f"{station_name} — Normal scenario")
        plot_error_dist_standalone(
            preds, station_dir / "error_dist_normal.png",
            title=f"{station_name} — Normal scenario")

        # Extreme
        ext_met = metrics.get("extreme", {})
        if not ext_met:
            print("  No extreme metrics — skipped")
        else:
            preds_ext = {}
            ok = True
            for m in MODELS:
                fp = station_dir / f"test_predictions_extreme_{m['tag']}.csv"
                if not fp.exists():
                    print(f"  Missing {fp.name} — skipped extreme plot")
                    ok = False; break
                p = pd.read_csv(fp)
                p["Time"] = pd.to_datetime(p["Time"])
                preds_ext[m["tag"]] = p
            if ok:
                plot_extreme(metrics, ext_met, preds_ext,
                             metrics.get("extreme_improvement", {}),
                             hists,
                             station_dir / "comparison_extreme.png")
                # Standalone extreme scatter & error distribution
                plot_scatter_standalone(
                    preds_ext, station_dir / "scatter_extreme.png",
                    title=f"{station_name} — Extreme scenario (30% missing)")
                plot_error_dist_standalone(
                    preds_ext, station_dir / "error_dist_extreme.png",
                    title=f"{station_name} — Extreme scenario (30% missing)")

        # Sweep
        plot_sweep(metrics, station_dir / "miss_ratio_sweep.png")

    # ── Cross-station box plot ──────────────────────────────────────────
    all_metrics = {}
    for station_name in STATIONS:
        mf = OUTPUT_DIR / station_name / "test_metrics.json"
        if mf.exists():
            with open(mf, "r", encoding="utf-8") as f:
                all_metrics[station_name] = json.load(f)
    if len(all_metrics) >= 2:
        plot_metrics_boxplot(all_metrics, OUTPUT_DIR / "metrics_boxplot.png")


if __name__ == "__main__":
    main()
