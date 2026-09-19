"""
Generate extreme-weather predictions with 0% missing data (pure extreme weather).
Uses saved model weights — no retraining needed.
Also saves to main.py format for future use.
"""

raise RuntimeError('Inspection-only historical code. Execution is disabled; use scripts/reproduce.py train with the reviewed protocol.')

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from main import (
    merge_all_data, add_time_features, build_sequences, split_data_stratified,
    fit_scalers, PVDataset, evaluate_on_loader, select_extreme_samples,
    inject_missing, set_seed,
    FEATURE_COLS, WINDOW_SIZE, BATCH_SIZE, RANDOM_SEED, DEVICE,
    RATED_POWER, INVERTER_FILES, OUTPUT_DIR,
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_DROPOUT,
)
from model import DNNModel, BaselineModel, GRUModel, TCNModel, PINNModel, TransformerModel, TransformerModel


def gen_station(station_name, inverter_file):
    station_dir = OUTPUT_DIR / station_name
    set_seed(RANDOM_SEED)

    df = merge_all_data(inverter_file)
    df = add_time_features(df)
    X, y, irr, temp, times = build_sequences(df, FEATURE_COLS, window_size=WINDOW_SIZE)
    sp = split_data_stratified(X, y, irr, temp, times)
    Xtr, ytr, irr_tr, tmp_tr, ttr = sp["train"]
    Xte, yte, irr_te, tmp_te, tte = sp["test"]

    xm, xs, ymin, ymax = fit_scalers(Xtr, ytr)
    Xte_s = (Xte - xm) / xs
    yte_s = (yte - ymin) / (ymax - ymin)
    ym = ymin
    ystd = ymax - ymin
    rated_power = float(RATED_POWER or np.max(ytr) * 1.05)
    ndim = len(FEATURE_COLS)

    # Select extreme samples
    ext_idx = select_extreme_samples(Xte_s, yte_s, yte, irr_te, tmp_te, tte,
                                     irr_all=irr_tr, temp_all=tmp_tr)
    X_ext_raw = Xte_s[ext_idx]
    y_ext_s = yte_s[ext_idx]
    y_ext = yte[ext_idx]
    irr_ext = irr_te[ext_idx]
    tmp_ext = tmp_te[ext_idx]
    t_ext = tte[ext_idx]

    # 0% missing — no injection
    X_0miss, irr_0miss, tmp_0miss = inject_missing(X_ext_raw, irr_ext, tmp_ext, 0.0, seed=RANDOM_SEED)
    loader = DataLoader(
        PVDataset(X_0miss, y_ext_s, y_ext, irr_0miss, tmp_0miss, t_ext),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model_configs = [
        ("dnn",   DNNModel(ndim, HIDDEN_DIM, WINDOW_SIZE, DROPOUT), False),
        ("lstm",  BaselineModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT), False),
        ("gru",   GRUModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT), False),
        ("tcn",   TCNModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT), False),
        ("trans", TransformerModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT), False),
        ("pinn",  PINNModel(ndim, PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_DROPOUT), True),
    ]

    for tag, model, is_pinn in model_configs:
        ckpt = torch.load(station_dir / f"best_model_{tag}.pth",
                          map_location=DEVICE, weights_only=False)
        if is_pinn:
            model.set_output_scale(ym, ystd)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)

        res = evaluate_on_loader(model, loader, ym, ystd, is_pinn, DEVICE,
                                 rated_power=rated_power)
        pd.DataFrame({
            "Time": pd.to_datetime(res["times"]),
            "True_Power": res["trues"],
            "Pred_Power": res["preds"],
        }).to_csv(station_dir / f"test_predictions_extreme_0miss_{tag}.csv",
                  index=False, encoding="utf-8-sig")
        print(f"  {tag}: R2={res['r2']:.3f}  MAE={res['mae']:.0f}")

    # Also save 0miss metrics to JSON
    json_path = station_dir / "test_metrics.json"
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics["extreme_0miss"] = {}
    for tag, model, is_pinn in model_configs:
        ckpt = torch.load(station_dir / f"best_model_{tag}.pth",
                          map_location=DEVICE, weights_only=False)
        if is_pinn:
            model.set_output_scale(ym, ystd)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        res = evaluate_on_loader(model, loader, ym, ystd, is_pinn, DEVICE,
                                 rated_power=rated_power)
        metrics["extreme_0miss"][tag] = {k: res[k] for k in ["mae", "mse", "rmse", "r2"]}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"  -> saved CSV & JSON for {station_name}")


def main():
    print(f"Device: {DEVICE}")
    for station_name, inverter_file in INVERTER_FILES:
        print(f"\n=== {station_name} ===")
        gen_station(station_name, inverter_file)
    print("\nDone! 0% missing extreme predictions generated.")


if __name__ == "__main__":
    main()
