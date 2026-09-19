"""
Generate miss_sweep data for Inverter_3 using already-trained models.
Saves results directly into outputs/Inverter_3/test_metrics.json.
"""

raise RuntimeError('Inspection-only historical code. Execution is disabled; use scripts/reproduce.py train with the reviewed protocol.')

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── reuse everything from main.py ──────────────────────────────────────────
from main import (
    DATA_DIR, OUTPUT_DIR,
    FEATURE_COLS, WINDOW_SIZE, BATCH_SIZE, DEVICE, RANDOM_SEED,
    HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_DROPOUT,
    EXTREME_MISS_RATIOS,
    LOW_IRR_THRESHOLD,
    merge_all_data, add_time_features, build_sequences,
    split_data_stratified, fit_scalers,
    PVDataset, postprocess_predictions,
    select_extreme_samples, inject_missing, evaluate_on_loader,
    set_seed,
)
from model import DNNModel, BaselineModel, GRUModel, TCNModel, PINNModel, TransformerModel

STATION_NAME  = "Inverter_3"
INVERTER_FILE = DATA_DIR / "Library_Inverter_3.csv"
STATION_DIR   = OUTPUT_DIR / STATION_NAME


def load_model(tag, model_obj, station_dir, device):
    ckpt = torch.load(station_dir / f"best_model_{tag}.pth",
                      map_location=device, weights_only=False)
    model_obj.load_state_dict(ckpt["model_state_dict"])
    model_obj.to(device).eval()
    return model_obj


def main():
    set_seed(RANDOM_SEED)

    # ── 1. Load & preprocess data (identical to main.py) ──────────────────
    print("Loading data...")
    df = merge_all_data(INVERTER_FILE)
    df = add_time_features(df)

    X, y, irr, temp, times = build_sequences(df, FEATURE_COLS, window_size=WINDOW_SIZE)
    sp = split_data_stratified(X, y, irr, temp, times)
    Xtr, ytr, irr_tr, tmp_tr, ttr = sp["train"]
    Xte, yte, irr_te, tmp_te, tte = sp["test"]

    xm, xs, ymin, ymax = fit_scalers(Xtr, ytr)
    Xte_s = (Xte - xm) / xs
    yte_s  = (yte - ymin) / (ymax - ymin)
    ym   = ymin
    ystd = ymax - ymin

    rated_power = float(np.max(ytr) * 1.05)
    print(f"rated_power = {rated_power:.0f} W")

    # ── 2. Instantiate models ──────────────────────────────────────────────
    ndim = len(FEATURE_COLS)
    dnn_model  = DNNModel(ndim, HIDDEN_DIM, WINDOW_SIZE, DROPOUT).to(DEVICE)
    bl_model   = BaselineModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    gru_model  = GRUModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    tcn_model  = TCNModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    tr_model   = TransformerModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    # Inv3 PINN was retrained with hidden=192 (retrain_pinn_multiseed.py)
    pinn_model = PINNModel(ndim, 192, PINN_NUM_LAYERS, PINN_DROPOUT).to(DEVICE)
    pinn_model.set_output_scale(ym, ystd)

    tag_map = [
        ("dnn",   dnn_model,  False),
        ("lstm",  bl_model,   False),
        ("gru",   gru_model,  False),
        ("tcn",   tcn_model,  False),
        ("trans", tr_model,   False),
        ("pinn",  pinn_model, True),
    ]
    name_map = {"dnn": "DNN", "lstm": "LSTM", "gru": "GRU",
                "tcn": "TCN", "trans": "Transformer", "pinn": "PINN"}

    for tag, mdl, _ in tag_map:
        load_model(tag, mdl, STATION_DIR, DEVICE)
        print(f"  Loaded {tag}")

    # ── 3. Select extreme test samples ────────────────────────────────────
    ext_idx = select_extreme_samples(
        Xte_s, yte_s, yte, irr_te, tmp_te, tte,
        irr_all=irr_tr, temp_all=tmp_tr)

    X_ext_raw = Xte_s[ext_idx]
    y_ext_s   = yte_s[ext_idx]
    y_ext     = yte[ext_idx]
    irr_ext   = irr_te[ext_idx]
    tmp_ext   = tmp_te[ext_idx]
    t_ext     = tte[ext_idx]

    # ── 4. Sweep miss ratios ───────────────────────────────────────────────
    sweep_results = {}
    for mr in EXTREME_MISS_RATIOS:
        pct_label = int(round(mr * 100))
        X_miss, irr_miss, tmp_miss = inject_missing(
            X_ext_raw, irr_ext, tmp_ext, mr, seed=RANDOM_SEED)
        loader = DataLoader(
            PVDataset(X_miss, y_ext_s, y_ext, irr_miss, tmp_miss, t_ext),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        row = {}
        for tag, mdl, is_p in tag_map:
            res = evaluate_on_loader(mdl, loader, ym, ystd, is_p, DEVICE,
                                     rated_power=rated_power)
            row[name_map[tag]] = res

        sweep_results[pct_label] = row
        line = f"  miss={pct_label:2d}%"
        for tag, _, _ in tag_map:
            line += f"  {name_map[tag]} R2={row[name_map[tag]]['r2']:.4f}"
        print(line)

    # ── 5. Update test_metrics.json ───────────────────────────────────────
    metrics_path = STATION_DIR / "test_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    ALL_TAGS = {"DNN": "dnn", "LSTM": "lstm", "GRU": "gru",
                "TCN": "tcn", "Transformer": "trans", "PINN": "pinn"}
    metrics["miss_sweep"] = {
        str(pct): {
            tag: {k: sweep_results[pct][name][k]
                  for k in ["mae", "mse", "rmse", "r2"]}
            for name, tag in ALL_TAGS.items()
        }
        for pct, row in sweep_results.items()
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"\n[saved] miss_sweep ({len(sweep_results)} ratios) → {metrics_path}")


if __name__ == "__main__":
    main()
