"""
Quick evaluation: load saved model weights, compute train/val/test metrics,
update test_metrics.json with train & val sections for box-plot visualisation.
No re-training needed.
"""

raise RuntimeError('Inspection-only historical code. Execution is disabled; use scripts/reproduce.py train with the reviewed protocol.')

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from main import (
    merge_all_data, add_time_features, build_sequences, split_data_stratified,
    fit_scalers, PVDataset, evaluate_on_loader, set_seed,
    FEATURE_COLS, WINDOW_SIZE, BATCH_SIZE, RANDOM_SEED, DEVICE,
    RATED_POWER, INVERTER_FILES, OUTPUT_DIR,
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, PINN_HIDDEN_DIM,
)
from model import DNNModel, BaselineModel, GRUModel, TCNModel, PINNModel


def eval_station(station_name, inverter_file):
    station_dir = OUTPUT_DIR / station_name
    set_seed(RANDOM_SEED)

    # Load data using the main training pipeline
    df = merge_all_data(inverter_file)
    df = add_time_features(df)
    X, y, irr, temp, times = build_sequences(df, FEATURE_COLS, window_size=WINDOW_SIZE)
    sp = split_data_stratified(X, y, irr, temp, times)
    Xtr, ytr, irr_tr, tmp_tr, ttr = sp["train"]
    Xva, yva, irr_va, tmp_va, tva = sp["val"]
    Xte, yte, irr_te, tmp_te, tte = sp["test"]

    xm, xs, ymin, ymax = fit_scalers(Xtr, ytr)
    Xtr_s = (Xtr - xm) / xs
    Xva_s = (Xva - xm) / xs
    Xte_s = (Xte - xm) / xs
    ytr_s = (ytr - ymin) / (ymax - ymin)
    yva_s = (yva - ymin) / (ymax - ymin)
    yte_s = (yte - ymin) / (ymax - ymin)
    ym = ymin
    ystd = ymax - ymin

    rated_power = float(RATED_POWER or np.max(ytr) * 1.05)
    ndim = len(FEATURE_COLS)

    trn = DataLoader(PVDataset(Xtr_s, ytr_s, ytr, irr_tr, tmp_tr, ttr),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(PVDataset(Xva_s, yva_s, yva, irr_va, tmp_va, tva),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    tst = DataLoader(PVDataset(Xte_s, yte_s, yte, irr_te, tmp_te, tte),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model_configs = [
        ("dnn",  DNNModel(ndim, HIDDEN_DIM, WINDOW_SIZE, DROPOUT), False),
        ("lstm", BaselineModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT), False),
        ("gru",  GRUModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT), False),
        ("tcn",  TCNModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT), False),
        ("pinn", PINNModel(ndim, PINN_HIDDEN_DIM, NUM_LAYERS, DROPOUT), True),
    ]

    # Read the saved metrics JSON
    json_path = station_dir / "test_metrics.json"
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics["train"] = {}
    metrics["val"] = {}

    print(f"  {'model':<6} {'split':<6} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("  " + "-" * 40)

    for tag, model, is_pinn in model_configs:
        ckpt = torch.load(station_dir / f"best_model_{tag}.pth",
                          map_location=DEVICE, weights_only=False)
        if is_pinn:
            model.set_output_scale(ym, ystd)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)

        for split_name, loader in [("train", trn), ("val", val_loader)]:
            res = evaluate_on_loader(model, loader, ym, ystd, is_pinn, DEVICE,
                                     rated_power=rated_power)
            m_dict = {k: res[k] for k in ["mae", "mse", "rmse", "r2"]}
            metrics[split_name][tag] = m_dict
            print(f"  {tag:<6} {split_name:<6} {res['mae']:>8.0f} "
                  f"{res['rmse']:>8.0f} {res['r2']:>8.4f}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"  ✓ Updated {json_path}")


def main():
    print(f"Device: {DEVICE}")
    for station_name, inverter_file in INVERTER_FILES:
        print(f"\n=== {station_name} ===")
        eval_station(station_name, inverter_file)
    print("\nDone! Now run:  python plot.py")


if __name__ == "__main__":
    main()
