"""
Retrain only the PINN model with improved hyperparameters.
All other models' checkpoints and predictions are kept intact.
Updates test_metrics.json (pinn entries only) and saves new PINN prediction CSVs.
"""

raise RuntimeError('Inspection-only historical code. Execution is disabled; use scripts/reproduce.py train with the reviewed protocol.')

import copy
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from main import (
    merge_all_data, add_time_features, build_sequences, split_data_stratified,
    PVDataset, evaluate_on_loader, postprocess_predictions,
    select_extreme_samples, inject_missing, pinn_loss, set_seed,
    train_and_evaluate,
    FEATURE_COLS, WINDOW_SIZE, BATCH_SIZE, RANDOM_SEED, DEVICE,
    RATED_POWER, INVERTER_FILES, OUTPUT_DIR, WEIGHT_DECAY,
    PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_DROPOUT, PINN_EPOCHS,
    PINN_PATIENCE, PINN_LR_BACKBONE, PINN_LR_PHYSICS,
    EXTREME_MISS_RATIOS,
)
from model import PINNModel

ALL_TAGS = {
    "DNN": "dnn", "LSTM": "lstm", "GRU": "gru",
    "TCN": "tcn", "Transformer": "trans", "PINN": "pinn",
}


def retrain_pinn_station(station_name, inverter_file):
    station_dir = OUTPUT_DIR / station_name
    set_seed(RANDOM_SEED)

    print(f"\n{'=' * 64}")
    print(f"  Retraining PINN: {station_name}")
    print(f"  hidden={PINN_HIDDEN_DIM}  layers={PINN_NUM_LAYERS}  "
          f"dropout={PINN_DROPOUT}  epochs={PINN_EPOCHS}  patience={PINN_PATIENCE}")
    print(f"{'=' * 64}")

    # ── Data ─────────────────────────────────────────────────────────────
    df = merge_all_data(inverter_file)
    df = add_time_features(df)
    X, y, irr, temp, times = build_sequences(df, FEATURE_COLS, window_size=WINDOW_SIZE)
    sp = split_data_stratified(X, y, irr, temp, times)
    Xtr, ytr, irr_tr, tmp_tr, ttr = sp["train"]
    Xva, yva, irr_va, tmp_va, tva = sp["val"]
    Xte, yte, irr_te, tmp_te, tte = sp["test"]

    # Load saved scalers (ensures consistency with other models)
    sc = np.load(station_dir / "scalers.npz")
    xm   = sc["x_mean"]
    xs   = sc["x_std"]
    ym   = float(sc["y_mean"])
    ystd = float(sc["y_std"])

    Xtr_s = (Xtr - xm) / xs;  Xva_s = (Xva - xm) / xs;  Xte_s = (Xte - xm) / xs
    ytr_s = (ytr - ym) / ystd; yva_s = (yva - ym) / ystd; yte_s = (yte - ym) / ystd
    rated_power = float(RATED_POWER or np.max(ytr) * 1.05)
    ndim = len(FEATURE_COLS)

    trn = DataLoader(PVDataset(Xtr_s, ytr_s, ytr, irr_tr, tmp_tr, ttr),
                     batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val = DataLoader(PVDataset(Xva_s, yva_s, yva, irr_va, tmp_va, tva),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    tst = DataLoader(PVDataset(Xte_s, yte_s, yte, irr_te, tmp_te, tte),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Build & Train PINN ────────────────────────────────────────────────
    pinn_model = PINNModel(ndim, PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_DROPOUT).to(DEVICE)
    pinn_model.set_output_scale(ym, ystd)

    phys_names = {"log_alpha", "beta", "T_ref", "log_noct", "gate_param"}
    phys_params    = [p for n, p in pinn_model.named_parameters() if n in phys_names]
    backbone_params = [p for n, p in pinn_model.named_parameters() if n not in phys_names]
    pinn_optim = torch.optim.AdamW(
        [{"params": backbone_params, "lr": PINN_LR_BACKBONE},
         {"params": phys_params,     "lr": PINN_LR_PHYSICS}],
        weight_decay=WEIGHT_DECAY)

    pn = train_and_evaluate(
        pinn_model, trn, val, tst,
        ym, ystd, rated_power,
        pinn_loss, True, "PINN", DEVICE,
        optimizer=pinn_optim, patience=PINN_PATIENCE, epochs=PINN_EPOCHS)

    pinn_model.load_state_dict(pn["state"])

    # Train / val metrics
    tr_r = evaluate_on_loader(pinn_model, trn, ym, ystd, True, DEVICE, rated_power=rated_power)
    va_r = evaluate_on_loader(pinn_model, val, ym, ystd, True, DEVICE, rated_power=rated_power)

    # ── Extreme sweep (PINN only) ─────────────────────────────────────────
    ext_idx = select_extreme_samples(Xte_s, yte_s, yte, irr_te, tmp_te, tte,
                                     irr_all=irr_tr, temp_all=tmp_tr)
    X_ext  = Xte_s[ext_idx];  y_ext_s = yte_s[ext_idx]; y_ext = yte[ext_idx]
    irr_ext = irr_te[ext_idx]; tmp_ext = tmp_te[ext_idx]; t_ext = tte[ext_idx]

    pinn_sweep = {}
    ext_30_res = None
    for mr in EXTREME_MISS_RATIOS:
        pct_label = int(round(mr * 100))
        Xm, im, tm = inject_missing(X_ext, irr_ext, tmp_ext, mr, seed=RANDOM_SEED)
        ldr = DataLoader(PVDataset(Xm, y_ext_s, y_ext, im, tm, t_ext),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        res = evaluate_on_loader(pinn_model, ldr, ym, ystd, True, DEVICE,
                                 rated_power=rated_power)
        pinn_sweep[pct_label] = {k: res[k] for k in ["mae", "mse", "rmse", "r2"]}
        print(f"  miss={pct_label:2d}%  PINN R2={res['r2']:.4f}  MAE={res['mae']:.0f}")
        if pct_label == 30:
            ext_30_res = res

    # ── Extreme 0-miss ────────────────────────────────────────────────────
    Xm0, im0, tm0 = inject_missing(X_ext, irr_ext, tmp_ext, 0.0, seed=RANDOM_SEED)
    ldr0 = DataLoader(PVDataset(Xm0, y_ext_s, y_ext, im0, tm0, t_ext),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    res0 = evaluate_on_loader(pinn_model, ldr0, ym, ystd, True, DEVICE,
                              rated_power=rated_power)

    # ── Update JSON (PINN entries only) ──────────────────────────────────
    json_path = station_dir / "test_metrics.json"
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics["pinn"]              = {k: pn[k]   for k in ["mae", "mse", "rmse", "r2"]}
    metrics["train"]["pinn"]     = {k: tr_r[k] for k in ["mae", "mse", "rmse", "r2"]}
    metrics["val"]["pinn"]       = {k: va_r[k] for k in ["mae", "mse", "rmse", "r2"]}
    metrics["extreme"]["pinn"]   = {k: ext_30_res[k] for k in ["mae", "mse", "rmse", "r2"]}
    if "extreme_0miss" not in metrics:
        metrics["extreme_0miss"] = {}
    metrics["extreme_0miss"]["pinn"] = {k: res0[k] for k in ["mae", "mse", "rmse", "r2"]}
    for pct_label, sweep_row in pinn_sweep.items():
        key = str(pct_label)
        if key in metrics.get("miss_sweep", {}):
            metrics["miss_sweep"][key]["pinn"] = sweep_row

    # Update improvement metrics
    def pct_chg(a, b): return (b - a) / abs(a) * 100 if abs(a) > 1e-8 else 0
    dnn_mae = metrics["dnn"]["mae"]
    metrics["improvement"]["pinn_vs_dnn_mae_pct"] = pct_chg(dnn_mae, pn["mae"])
    ext_dnn_mae = metrics["extreme"]["dnn"]["mae"]
    metrics["extreme_improvement"]["pinn_vs_dnn_mae_pct"] = pct_chg(ext_dnn_mae, ext_30_res["mae"])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    # ── Save prediction CSVs & model ─────────────────────────────────────
    pd.DataFrame({"Time": pd.to_datetime(pn["times"]),
                  "True_Power": pn["trues"], "Pred_Power": pn["preds"]}).to_csv(
        station_dir / "test_predictions_pinn.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"Time": pd.to_datetime(ext_30_res["times"]),
                  "True_Power": ext_30_res["trues"], "Pred_Power": ext_30_res["preds"]}).to_csv(
        station_dir / "test_predictions_extreme_pinn.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"Time": pd.to_datetime(res0["times"]),
                  "True_Power": res0["trues"], "Pred_Power": res0["preds"]}).to_csv(
        station_dir / "test_predictions_extreme_0miss_pinn.csv", index=False, encoding="utf-8-sig")

    torch.save({"model_state_dict": pn["state"],
                "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ystd},
               station_dir / "best_model_pinn.pth")

    print(f"\n  ✓ {station_name}  Test R2={pn['r2']:.4f}  Extreme_0miss R2={res0['r2']:.4f}")
    print(f"    Files saved to {station_dir}\n")


def main():
    print(f"Device: {DEVICE}")
    print(f"PINN config: hidden={PINN_HIDDEN_DIM}  layers={PINN_NUM_LAYERS}  "
          f"dropout={PINN_DROPOUT}  epochs={PINN_EPOCHS}  patience={PINN_PATIENCE}")
    for station_name, inverter_file in INVERTER_FILES:
        retrain_pinn_station(station_name, inverter_file)
    print("\nPINN retraining complete! Run plot_new.py to regenerate plots.")


if __name__ == "__main__":
    main()
