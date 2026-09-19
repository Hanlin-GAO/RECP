"""
Wind power prediction — publication-quality visualisation (Times New Roman, MW units).

Generates:
  Per turbine:
    comparison_normal.png       – Normal scenario dashboard
    comparison_extreme.png      – Extreme scenario dashboard
    miss_ratio_sweep.png        – Robustness curve
    train_val_test_bar.png      – Train/Val/Test bar chart
    train_boxplot.png           – Prediction error boxplot
  Cross-turbine (outputs_wind/):
    scatter_normal.png          – 3×5 scatter (normal)
    scatter_extreme.png         – 3×5 scatter (extreme)
    error_dist_normal.png       – 1×3 error KDE (normal)
    error_dist_extreme.png      – 1×3 error KDE (extreme)
    miss_ratio_sweep_all.png    – 3×3 sweep
    metrics_boxplot.png         – box plot across turbines
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════
# Global rc
# ═══════════════════════════════════════════════════════════════════════════
def _setup_rc():
    mpl.rcParams.update({
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
        "axes.linewidth":     1.0,
        "axes.unicode_minus": False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.major.width":  1.0,  "ytick.major.width":  1.0,
        "xtick.minor.width":  0.6,  "ytick.minor.width":  0.6,
        "xtick.major.size":   5,    "ytick.major.size":   5,
        "xtick.minor.size":   2.5,  "ytick.minor.size":   2.5,
        "xtick.direction":    "out", "ytick.direction":   "out",
        "lines.linewidth":    1.4,
        "lines.markersize":   6,
        "axes.grid":          False,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "figure.dpi":         150,
        "figure.facecolor":   "white",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Palette
# ═══════════════════════════════════════════════════════════════════════════
from forecast_paths import WIND_OUTPUT
OUTPUT_DIR = WIND_OUTPUT
STATIONS = ["Turbine_1", "Turbine_2", "Turbine_3"]
STATION_LABELS = ["Turbine 1", "Turbine 2", "Turbine 3"]

MODELS = [
    {"tag": "dnn",   "label": "DNN",         "color": "#4daf4a", "marker": "o"},
    {"tag": "lstm",  "label": "LSTM",        "color": "#377eb8", "marker": "s"},
    {"tag": "gru",   "label": "GRU",         "color": "#984ea3", "marker": "D"},
    {"tag": "tcn",   "label": "TCN",         "color": "#e41a1c", "marker": "^"},
    {"tag": "trans", "label": "Transformer", "color": "#17becf", "marker": "v"},
    {"tag": "pinn",  "label": "PINN",        "color": "#ff7f00", "marker": "P"},
]
N = len(MODELS)
_LABELS = [m["label"] for m in MODELS]
_COLORS = [m["color"] for m in MODELS]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _panel_label(ax, label, x=-0.10, y=1.12):
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
def _panel_loss(ax, hists):
    for m in MODELS:
        h = hists[m["tag"]]
        ax.plot(h["epoch"], h["train_data"], color=m["color"],
                lw=1.0, alpha=0.85, label=f'{m["label"]} train')
        ax.plot(h["epoch"], h["val_data"], color=m["color"],
                lw=1.0, ls="--", alpha=0.85, label=f'{m["label"]} val')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (normalised)")
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
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (MW)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    _light_grid(ax); _minor_ticks(ax)
    ax.legend(ncol=2, frameon=False, loc="upper right",
              borderaxespad=0.3, handlelength=1.5, columnspacing=0.8)

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
    ax.set_xlabel("Time"); ax.set_ylabel("Power (MW)")
    if subtitle:
        ax.set_title(subtitle, fontsize=22, pad=8)
    _light_grid(ax)
    ax.legend(ncol=N + 1, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=1.0)
    ax.tick_params(axis="x", rotation=15)

def _panel_scatter(ax, preds, tag, label, color):
    p = preds[tag]
    t_arr = p["True_Power"].values
    p_arr = p["Pred_Power"].values
    ax.scatter(t_arr, p_arr, s=5, alpha=0.45, color=color,
               edgecolors="none", rasterized=True)
    hi = max(t_arr.max(), p_arr.max()) * 1.02
    ax.plot([0, hi], [0, hi], color="#444444", lw=0.6,
            ls="--", alpha=0.5, zorder=5)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ss_res = np.sum((t_arr - p_arr) ** 2)
    ss_tot = np.sum((t_arr - t_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0
    ax.text(0.05, 0.93, f"$R^2 = {r2:.2f}$", transform=ax.transAxes,
            fontsize=18, va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("True power (MW)"); ax.set_ylabel("Predicted power (MW)")
    ax.set_title(label, fontsize=22, pad=6)
    ax.set_aspect("equal", adjustable="datalim")
    _minor_ticks(ax)

def _panel_error_kde(ax, preds):
    for m in MODELS:
        err = (preds[m["tag"]]["Pred_Power"].values
               - preds[m["tag"]]["True_Power"].values)
        try:
            kde = stats.gaussian_kde(err, bw_method=0.15)
            xs = np.linspace(err.min(), err.max(), 300)
            ax.plot(xs, kde(xs), color=m["color"], lw=1.0, label=m["label"])
            ax.fill_between(xs, kde(xs), alpha=0.07, color=m["color"])
        except Exception:
            ax.hist(err, bins=60, density=True, alpha=0.35,
                    color=m["color"], label=m["label"], edgecolor="none")
    ax.axvline(0, color="#444444", lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("Prediction error (MW)"); ax.set_ylabel("Density")
    _light_grid(ax)
    ax.legend(frameon=False, handlelength=1.5)

def _panel_bar_mae_rmse(ax, met):
    x = np.arange(2); w = 0.14
    for i, m in enumerate(MODELS):
        vals = [met[m["tag"]]["mae"], met[m["tag"]]["rmse"]]
        bars = ax.bar(x + (i - (N - 1) / 2) * w, vals, w, label=m["label"],
                      color=m["color"], edgecolor="white", linewidth=0.3)
        _bar_val(ax, bars, fmt="{:.2f}", fs=14)
    ax.set_xticks(x); ax.set_xticklabels(["MAE", "RMSE"])
    ax.set_ylabel("Value (MW)")
    _light_grid(ax)
    ax.legend(frameon=False, ncol=N, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), handlelength=1.2, columnspacing=0.6)

def _panel_bar_r2(ax, met, ylim_lo=None):
    r2v = [met[m["tag"]]["r2"] for m in MODELS]
    bars = ax.bar(_LABELS, r2v, width=0.52,
                  color=_COLORS, edgecolor="white", linewidth=0.4)
    _bar_val(ax, bars, fmt="{:.2f}", fs=16)
    ax.set_ylabel("$R^2$")
    lo = min(0, min(r2v) - 0.05) if ylim_lo is None else ylim_lo
    ax.set_ylim(lo, 1.02)
    _light_grid(ax)

def _panel_residual(ax, preds, tag, color, n_show=1000):
    p = preds[tag]
    err = p["Pred_Power"].values - p["True_Power"].values
    n = min(n_show, len(err))
    ax.fill_between(range(n), err[:n], 0, color=color, alpha=0.22, linewidth=0)
    ax.plot(range(n), err[:n], color=color, lw=0.35, alpha=0.7)
    ax.axhline(0, color="#444444", lw=0.5, ls="-", alpha=0.4)
    ax.set_xlabel("Sample index"); ax.set_ylabel("Error (MW)")
    _light_grid(ax)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Normal Scenario (per turbine)
# ═══════════════════════════════════════════════════════════════════════════
def plot_normal(metrics, hists, preds, imp, fig_path):
    fig = plt.figure(figsize=(24, 24))
    gs = gridspec.GridSpec(5, N, figure=fig, hspace=0.70, wspace=0.55,
                           height_ratios=[1, 1.2, 0.9, 1, 0.9])
    ax0 = fig.add_subplot(gs[0, :3])
    _panel_loss(ax0, hists);  _panel_label(ax0, "a")
    ax1 = fig.add_subplot(gs[0, 3:])
    _panel_mae_curve(ax1, hists);  _panel_label(ax1, "b")
    ax2 = fig.add_subplot(gs[1, :])
    _panel_pred(ax2, preds, 1000);  _panel_label(ax2, "c")
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[2, i])
        _panel_scatter(ax, preds, m["tag"], m["label"], m["color"])
        if i == 0: _panel_label(ax, "d")
    ax3 = fig.add_subplot(gs[3, :2])
    _panel_error_kde(ax3, preds);  _panel_label(ax3, "e")
    ax4 = fig.add_subplot(gs[3, 2:N-1])
    _panel_bar_mae_rmse(ax4, metrics);  _panel_label(ax4, "f")
    ax5 = fig.add_subplot(gs[3, N-1])
    _panel_bar_r2(ax5, metrics);  _panel_label(ax5, "g")
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[4, i])
        _panel_residual(ax, preds, m["tag"], m["color"])
        ax.set_title(m["label"], fontsize=18, pad=6)
        if i == 0: _panel_label(ax, "h")

    pv_d = imp.get("pinn_vs_dnn_mae_pct", 0)
    pv_l = imp.get("pinn_vs_lstm_mae_pct", 0)
    fig.suptitle(
        f"Normal scenario  |  PINN vs DNN  MAE {pv_d:+.1f}%   "
        f"PINN vs LSTM  MAE {pv_l:+.1f}%",
        fontsize=24, fontweight="bold", y=1.02)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Extreme Scenario (per turbine)
# ═══════════════════════════════════════════════════════════════════════════
def plot_extreme(metrics, ext_met, preds_ext, ext_imp, hists, fig_path):
    fig = plt.figure(figsize=(24, 23))
    gs = gridspec.GridSpec(5, N, figure=fig, hspace=0.70, wspace=0.55,
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
        ax.set_title(m["label"], fontsize=18, pad=6)
        if i == 0: _panel_label(ax, "h")

    pv_d = ext_imp.get("pinn_vs_dnn_mae_pct", 0)
    pv_l = ext_imp.get("pinn_vs_lstm_mae_pct", 0)
    fig.suptitle(
        f"Extreme scenario (30% missing)  |  PINN vs DNN  MAE {pv_d:+.1f}%   "
        f"PINN vs LSTM  MAE {pv_l:+.1f}%",
        fontsize=24, fontweight="bold", y=1.02)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Robustness sweep (per turbine) — line chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_sweep(metrics, fig_path):
    sweep = metrics.get("miss_sweep", {})
    if not sweep:
        print("  No miss_sweep data — skipped"); return

    ratios = sorted(int(k) for k in sweep.keys())
    data = {k: {m["tag"]: [] for m in MODELS} for k in ("mae", "rmse", "r2")}
    for r in ratios:
        s = sweep[str(r)]
        for m in MODELS:
            for k in ("mae", "rmse", "r2"):
                data[k][m["tag"]].append(s[m["tag"]][k])

    metric_info = [("mae", "MAE (MW)"), ("rmse", "RMSE (MW)"), ("r2", "$R^2$")]
    xpos = np.arange(len(ratios))

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.32, top=0.82)

    for idx, (key, ylabel) in enumerate(metric_info):
        ax = axes[idx]
        for m in MODELS:
            ax.plot(xpos, data[key][m["tag"]], color=m["color"],
                    marker=m["marker"], lw=1.8, ms=7, label=m["label"])
        ax.set_xlabel("Missing ratio (%)", fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_title(ylabel, fontsize=24, fontweight="bold", pad=10)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{r}" for r in ratios], fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        _light_grid(ax)
        _minor_ticks(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=N,
               loc="upper center", bbox_to_anchor=(0.5, 1.00),
               handlelength=1.5, columnspacing=1.0, fontsize=18)
    fig.suptitle(
        "Model robustness under increasing sensor data loss (extreme weather)",
        fontsize=22, fontweight="bold", y=1.10)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Combined 3-turbine scatter  (3 rows × 5 cols)
# ═══════════════════════════════════════════════════════════════════════════
def plot_scatter_combined(all_preds, fig_path, title=""):
    nrows = len(all_preds)
    fig, axes = plt.subplots(nrows, N, figsize=(5 * N, 5.2 * nrows),
                             constrained_layout=True)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for r, (station_label, preds) in enumerate(all_preds):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]
            p = preds[m["tag"]]
            t_arr = p["True_Power"].values
            p_arr = p["Pred_Power"].values

            # subsample for performance if too many points
            if len(t_arr) > 3000:
                idx_s = np.random.default_rng(42).choice(len(t_arr), 3000, replace=False)
                t_plot, p_plot = t_arr[idx_s], p_arr[idx_s]
            else:
                t_plot, p_plot = t_arr, p_arr
            ax.scatter(t_plot, p_plot, s=5, alpha=0.45, color=m["color"],
                       edgecolors="none", rasterized=True)
            hi = max(t_arr.max(), p_arr.max()) * 1.02
            ax.plot([0, hi], [0, hi], color="#444444", lw=0.7,
                    ls="--", alpha=0.55, zorder=5)
            ax.set_xlim(0, hi); ax.set_ylim(0, hi)
            ax.set_aspect("equal", adjustable="datalim")

            ss_res = np.sum((t_arr - p_arr) ** 2)
            ss_tot = np.sum((t_arr - t_arr.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0
            rmse = np.sqrt(np.mean((t_arr - p_arr) ** 2))
            mae = np.mean(np.abs(t_arr - p_arr))

            ax.text(0.05, 0.95,
                    f"$R^2$={r2:.2f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}",
                    transform=ax.transAxes, fontsize=24, va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#cccccc", alpha=0.9, linewidth=0.5))

            _light_grid(ax, axis="both"); _minor_ticks(ax)
            ax.tick_params(axis="both", labelsize=22)

            if r == 0:
                ax.set_title(m["label"], fontsize=34, pad=22)
            if c == 0:
                ax.set_ylabel("Predicted (MW)", fontsize=26)
            else:
                ax.set_yticklabels([])
            if r == nrows - 1:
                ax.set_xlabel("True (MW)", fontsize=26)
            else:
                ax.set_xticklabels([])

        axes[r, -1].annotate(
            station_label, xy=(1.10, 0.5), xycoords="axes fraction",
            fontsize=28, fontweight="bold", rotation=-90,
            ha="left", va="center")

    if title:
        fig.suptitle(title, fontsize=38, fontweight="bold", y=1.07)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Combined 3-turbine error distribution  (1 row × 3 cols)
# ═══════════════════════════════════════════════════════════════════════════
def plot_error_dist_combined(all_preds, fig_path, title=""):
    ncols = len(all_preds)
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 8.0),
                             constrained_layout=False)
    if ncols == 1:
        axes = [axes]
    letters = "abc"

    for idx, (ax, (station_label, preds)) in enumerate(zip(axes, all_preds)):
        for m in MODELS:
            err = (preds[m["tag"]]["Pred_Power"].values
                   - preds[m["tag"]]["True_Power"].values)
            try:
                kde = stats.gaussian_kde(err, bw_method=0.15)
                xs = np.linspace(err.min(), err.max(), 400)
                ax.plot(xs, kde(xs), color=m["color"], lw=1.3, label=m["label"])
                ax.fill_between(xs, kde(xs), alpha=0.08, color=m["color"])
            except Exception:
                ax.hist(err, bins=60, density=True, alpha=0.35,
                        color=m["color"], label=m["label"], edgecolor="none")

        ax.axvline(0, color="#444444", lw=0.6, ls=":", alpha=0.5)
        ax.set_xlabel("Prediction error (MW)", fontsize=26)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=26)
        ax.set_title(station_label, fontsize=28, pad=10)
        ax.tick_params(axis="both", labelsize=20)
        _light_grid(ax)
        _panel_label(ax, letters[idx])

    # Reserve top margin so legend and suptitle don't overlap axes titles
    top = 0.55 if title else 0.68
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12,
                        top=top, wspace=0.28)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_y = top + 0.20   # well above subplot titles
    fig.legend(handles, labels, frameon=False, ncol=N,
               loc="upper center", bbox_to_anchor=(0.5, legend_y),
               handlelength=1.5, columnspacing=1.0, fontsize=22)

    if title:
        fig.suptitle(title, fontsize=30, fontweight="bold", y=top + 0.32)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Combined 3-turbine miss-ratio sweep  (3 rows × 3 cols) — line chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_sweep_combined(all_metrics, fig_path):
    station_items = [(sl, all_metrics[sn])
                     for sn, sl in zip(STATIONS, STATION_LABELS)
                     if sn in all_metrics and "miss_sweep" in all_metrics[sn]]
    if not station_items:
        print("  No sweep data — skipped"); return

    nrows = len(station_items)
    metric_info = [("mae", "MAE (MW)"), ("rmse", "RMSE (MW)"), ("r2", "$R^2$")]

    fig, axes = plt.subplots(nrows, 3, figsize=(24, 7.5 * nrows))
    fig.subplots_adjust(left=0.06, right=0.92, hspace=0.45, wspace=0.32, top=0.90)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for r, (station_label, met) in enumerate(station_items):
        sweep = met["miss_sweep"]
        ratios = sorted(int(k) for k in sweep.keys())
        xpos = np.arange(len(ratios))

        data = {k: {m["tag"]: [] for m in MODELS} for k in ("mae", "rmse", "r2")}
        for ratio in ratios:
            s = sweep[str(ratio)]
            for m in MODELS:
                for k in ("mae", "rmse", "r2"):
                    data[k][m["tag"]].append(s[m["tag"]][k])

        for c, (key, ylabel) in enumerate(metric_info):
            ax = axes[r, c]
            for m in MODELS:
                ax.plot(xpos, data[key][m["tag"]], color=m["color"],
                        marker=m["marker"], lw=1.8, ms=7, label=m["label"])
            ax.set_xlabel("Missing ratio (%)", fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_xticks(xpos)
            ax.set_xticklabels([f"{rt}" for rt in ratios], fontsize=16)
            ax.tick_params(axis="y", labelsize=16)
            _light_grid(ax)
            _minor_ticks(ax)
            if r == 0:
                ax.set_title(ylabel, fontsize=22, pad=10)
            if c == 2:
                ax.annotate(station_label, xy=(1.08, 0.5), xycoords="axes fraction",
                            fontsize=20, fontweight="bold", rotation=-90,
                            ha="left", va="center")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=N,
               loc="upper center", bbox_to_anchor=(0.5, 0.97),
               handlelength=1.5, columnspacing=1.0, fontsize=24)
    fig.suptitle(
        "Model robustness under increasing sensor data loss",
        fontsize=26, fontweight="bold", y=1.03)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Per-turbine train/val/test bar chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_station_train_bar(metrics, fig_path, station_label=""):
    if "train" not in metrics or "val" not in metrics:
        print("  No train/val metrics — skipped station bar chart"); return

    split_info = [
        ("train", "Train",      "#66c2a5"),
        ("val",   "Validation", "#fc8d62"),
        ("test",  "Test",       "#8da0cb"),
    ]
    metric_info = [("r2", "$R^2$"), ("rmse", "RMSE (MW)"), ("mae", "MAE (MW)")]
    panel_ids = "abc"

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    plt.subplots_adjust(wspace=0.38)

    x = np.arange(N)
    w = 0.22

    for ax_idx, (ax, (mkey, mlabel)) in enumerate(zip(axes, metric_info)):
        for j, (split_key, split_label, color) in enumerate(split_info):
            vals = []
            for m in MODELS:
                if split_key == "test":
                    raw = metrics[m["tag"]][mkey]
                else:
                    raw = metrics[split_key][m["tag"]][mkey]
                vals.append(raw)
            bars = ax.bar(x + (j - 1) * w, vals, w, label=split_label,
                          color=color, alpha=0.7, edgecolor="white", linewidth=0.4)
            fmt = "{:.2f}" if mkey == "r2" else "{:.2f}"
            _bar_val(ax, bars, fmt=fmt, fs=14, pad=0.002 if mkey == "r2" else 0)

        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in MODELS], rotation=30, ha="right")
        ax.set_ylabel(mlabel)
        _light_grid(ax)
        _panel_label(ax, panel_ids[ax_idx])
        if mkey == "r2":
            all_v = []
            for sk, _, _ in split_info:
                for m in MODELS:
                    all_v.append(metrics[m["tag"]]["r2"] if sk == "test"
                                 else metrics[sk][m["tag"]]["r2"])
            ax.set_ylim(max(0, min(all_v) - 0.05), 1.02)

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, alpha=0.7, edgecolor="#444444", label=lbl)
                      for _, lbl, c in split_info]
    axes[1].legend(handles=legend_patches, frameon=False, ncol=3,
                   loc="upper center", bbox_to_anchor=(0.5, 1.28),
                   handlelength=1.5, columnspacing=1.2)
    title = f"{station_label} \u2014 Train / Validation / Test" if station_label else \
            "Train / Validation / Test"
    fig.suptitle(title, fontsize=24, fontweight="bold", y=1.12)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Per-turbine training boxplot
# ═══════════════════════════════════════════════════════════════════════════
def plot_train_boxplot_station(preds, fig_path, station_label=""):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    plt.subplots_adjust(wspace=0.35)

    errors = []
    for m in MODELS:
        err = (preds[m["tag"]]["Pred_Power"].values
               - preds[m["tag"]]["True_Power"].values)
        errors.append(err)

    bp1 = axes[0].boxplot(
        errors, tick_labels=[m["label"] for m in MODELS],
        patch_artist=True, showfliers=False,
        medianprops=dict(color="#333333", linewidth=1.2),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        boxprops=dict(linewidth=0.5))
    for patch, m in zip(bp1["boxes"], MODELS):
        patch.set_facecolor(m["color"]); patch.set_alpha(0.6)
    axes[0].axhline(0, color="#444444", lw=0.5, ls=":", alpha=0.5)
    axes[0].set_ylabel("Prediction error (MW)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
    _light_grid(axes[0]); _panel_label(axes[0], "a")

    abs_errors = [np.abs(e) for e in errors]
    bp2 = axes[1].boxplot(
        abs_errors, tick_labels=[m["label"] for m in MODELS],
        patch_artist=True, showfliers=False,
        medianprops=dict(color="#333333", linewidth=1.2),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        boxprops=dict(linewidth=0.5))
    for patch, m in zip(bp2["boxes"], MODELS):
        patch.set_facecolor(m["color"]); patch.set_alpha(0.6)
    axes[1].set_ylabel("Absolute error (MW)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
    _light_grid(axes[1]); _panel_label(axes[1], "b")

    title = f"{station_label} \u2014 Normal conditions" if station_label \
            else "Normal conditions"
    fig.suptitle(title, fontsize=24, fontweight="bold", y=1.04)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Cross-turbine box plot
# ═══════════════════════════════════════════════════════════════════════════
def plot_metrics_boxplot(all_metrics, fig_path):
    """Grouped bar chart (mean ± range across stations) with journal-quality colors."""
    from matplotlib.patches import Patch

    stations = list(all_metrics.keys())
    if len(stations) < 1:
        print("  No station data — skipped"); return
    sample = all_metrics[stations[0]]
    if "train" not in sample or "val" not in sample:
        print("  No train/val metrics in JSON — skipped"); return

    # Wong (2011) colorblind-safe palette — Nature Methods recommended
    split_info = [
        ("train", "Train",      "#0072B2"),   # Wong blue
        ("val",   "Validation", "#E69F00"),   # Wong orange
        ("test",  "Test",       "#009E73"),   # Wong bluish-green
    ]
    metric_info = [("r2", "$R^2$"), ("rmse", "RMSE (MW)"), ("mae", "MAE (MW)")]
    panel_ids = "abc"
    bar_w = 0.24
    offsets = [-bar_w, 0.0, bar_w]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    plt.subplots_adjust(wspace=0.36)

    x = np.arange(N)

    for ax_idx, (ax, (mkey, mlabel)) in enumerate(zip(axes, metric_info)):
        for j, (split_key, split_label, color) in enumerate(split_info):
            means, lo_errs, hi_errs = [], [], []
            for m in MODELS:
                vals = [
                    (all_metrics[st][m["tag"]][mkey] if split_key == "test"
                     else all_metrics[st][split_key][m["tag"]][mkey])
                    for st in stations
                ]
                mn = float(np.mean(vals))
                means.append(mn)
                lo_errs.append(mn - min(vals))
                hi_errs.append(max(vals) - mn)

            ax.bar(x + offsets[j], means, bar_w,
                   label=split_label, color=color,
                   edgecolor="white", linewidth=0.6, zorder=3)
            ax.errorbar(x + offsets[j], means,
                        yerr=[lo_errs, hi_errs],
                        fmt="none", color="#222222",
                        elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([m["label"] for m in MODELS], rotation=30, ha="right")
        ax.set_ylabel(mlabel)
        _light_grid(ax)
        _panel_label(ax, panel_ids[ax_idx])
        if mkey == "r2":
            all_vals = [
                all_metrics[st][m["tag"]]["r2"] if sk == "test"
                else all_metrics[st][sk][m["tag"]]["r2"]
                for sk, _, _ in split_info
                for m in MODELS
                for st in stations
            ]
            ax.set_ylim(max(0, min(all_vals) - 0.03), 1.01)

    legend_patches = [Patch(facecolor=c, edgecolor="white", label=lbl)
                      for _, lbl, c in split_info]
    axes[1].legend(handles=legend_patches, frameon=False, ncol=3,
                   loc="upper center", bbox_to_anchor=(0.5, 1.18),
                   handlelength=1.5, columnspacing=1.2, fontsize=14)
    fig.suptitle("Model performance across turbines: Train / Validation / Test",
                 fontsize=24, fontweight="bold", y=1.06)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [saved] {fig_path}"); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    _setup_rc()

    all_metrics = {}
    all_preds_normal = []
    all_preds_extreme = []

    for sname, slabel in zip(STATIONS, STATION_LABELS):
        station_dir = OUTPUT_DIR / sname
        mf = station_dir / "test_metrics.json"
        if not mf.exists():
            print(f"Skipped {sname}: {mf} not found"); continue

        print(f"\n=== {sname} ===")
        with open(mf, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        all_metrics[sname] = metrics

        hists, preds = {}, {}
        for m in MODELS:
            hists[m["tag"]] = pd.read_csv(station_dir / f"history_{m['tag']}.csv")
            p = pd.read_csv(station_dir / f"test_predictions_{m['tag']}.csv")
            p["Time"] = pd.to_datetime(p["Time"])
            preds[m["tag"]] = p
        all_preds_normal.append((slabel, preds))

        plot_normal(metrics, hists, preds,
                    metrics.get("improvement", {}),
                    station_dir / "comparison_normal.png")

        plot_station_train_bar(metrics,
                               station_dir / "train_val_test_bar.png",
                               station_label=slabel)

        plot_train_boxplot_station(preds,
                                   station_dir / "train_boxplot.png",
                                   station_label=slabel)

        ext_met = metrics.get("extreme", {})
        if ext_met:
            preds_ext_30 = {}; ok_30 = True
            for m in MODELS:
                fp = station_dir / f"test_predictions_extreme_{m['tag']}.csv"
                if not fp.exists():
                    print(f"  Missing {fp.name}"); ok_30 = False; break
                p = pd.read_csv(fp); p["Time"] = pd.to_datetime(p["Time"])
                preds_ext_30[m["tag"]] = p
            if ok_30:
                plot_extreme(metrics, ext_met, preds_ext_30,
                             metrics.get("extreme_improvement", {}),
                             hists, station_dir / "comparison_extreme.png")

        preds_ext_0 = {}; ok_0 = True
        for m in MODELS:
            fp = station_dir / f"test_predictions_extreme_0miss_{m['tag']}.csv"
            if not fp.exists():
                print(f"  Missing {fp.name}"); ok_0 = False; break
            p = pd.read_csv(fp); p["Time"] = pd.to_datetime(p["Time"])
            preds_ext_0[m["tag"]] = p
        if ok_0:
            all_preds_extreme.append((slabel, preds_ext_0))

        plot_sweep(metrics, station_dir / "miss_ratio_sweep.png")

    # Cross-turbine combined figures
    if all_preds_normal:
        plot_scatter_combined(all_preds_normal,
                              OUTPUT_DIR / "scatter_normal.png",
                              title="Normal scenario — True vs. Predicted")
        plot_error_dist_combined(all_preds_normal,
                                 OUTPUT_DIR / "error_dist_normal.png",
                                 title="Normal scenario — Error distribution")
    if all_preds_extreme:
        plot_scatter_combined(all_preds_extreme,
                              OUTPUT_DIR / "scatter_extreme.png",
                              title="Extreme weather — True vs. Predicted")
        plot_error_dist_combined(all_preds_extreme,
                                 OUTPUT_DIR / "error_dist_extreme.png",
                                 title="Extreme weather — Error distribution")
    if len(all_metrics) >= 2:
        plot_sweep_combined(all_metrics, OUTPUT_DIR / "miss_ratio_sweep_all.png")
        plot_metrics_boxplot(all_metrics, OUTPUT_DIR / "metrics_boxplot.png")


if __name__ == "__main__":
    main()
