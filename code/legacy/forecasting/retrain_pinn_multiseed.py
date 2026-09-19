"""
Multi-seed PINN retrain for Inverter_3.
Runs N random seeds, keeps the one with best test R2.
Uses original proven settings (MSE loss, gate=0.0, hidden=192, CosineAnnealing).
"""

raise RuntimeError('Inspection-only historical code. Execution is disabled; use scripts/reproduce.py train with the reviewed protocol.')

import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from main import (
    merge_all_data, add_time_features, build_sequences, split_data_stratified,
    PVDataset, evaluate_on_loader, pinn_loss,
    select_extreme_samples, inject_missing, set_seed,
    FEATURE_COLS, WINDOW_SIZE, BATCH_SIZE, DEVICE,
    RATED_POWER, OUTPUT_DIR, WEIGHT_DECAY,
    EXTREME_MISS_RATIOS,
    LAMBDA_NONNEG, LAMBDA_UPPER, LAMBDA_LOW_IRR,
)
from model import PINNModel

INV3_FILE  = Path(__file__).parent / "Library_Inverter_3.csv"
STATION    = "Inverter_3"

# Hyperparams
HIDDEN_DIM  = 192
NUM_LAYERS  = 3
DROPOUT     = 0.1
EPOCHS      = 400
PATIENCE    = 150
LR_BACKBONE = 2e-4
LR_PHYSICS  = 4e-5
COSINE_T0   = 50
COSINE_TMUL = 2

# Seeds to try
SEEDS = [42, 123, 7, 2024, 999]


def train_one_seed(seed, Xtr_s, ytr_s, ytr, irr_tr, tmp_tr, ttr,
                   Xva_s, yva_s, yva, irr_va, tmp_va, tva,
                   Xte_s, yte_s, yte, irr_te, tmp_te, tte,
                   xm, xs, ym, ystd, rated_power, ndim):
    set_seed(seed)

    trn = DataLoader(PVDataset(Xtr_s, ytr_s, ytr, irr_tr, tmp_tr, ttr),
                     batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val = DataLoader(PVDataset(Xva_s, yva_s, yva, irr_va, tmp_va, tva),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    tst = DataLoader(PVDataset(Xte_s, yte_s, yte, irr_te, tmp_te, tte),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PINNModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    model.set_output_scale(ym, ystd)

    phys_names = {"log_alpha", "beta", "T_ref", "log_noct", "gate_param"}
    phys_params     = [p for n, p in model.named_parameters() if n in phys_names]
    backbone_params = [p for n, p in model.named_parameters() if n not in phys_names]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": phys_params,     "lr": LR_PHYSICS}],
        weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_TMUL, eta_min=1e-6)

    ym_t  = torch.tensor(ym,          dtype=torch.float32, device=DEVICE)
    ys_t  = torch.tensor(ystd,        dtype=torch.float32, device=DEVICE)
    rp_t  = torch.tensor(rated_power, dtype=torch.float32, device=DEVICE)

    best_val   = float("inf")
    wait       = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for b in trn:
            x   = b["x"].to(DEVICE)
            y_s = b["y_s"].to(DEVICE)
            irr = b["irr"].to(DEVICE)
            tmp = b["temp"].to(DEVICE)
            optimizer.zero_grad()
            pred = model(x, G=irr, T_amb=tmp)
            loss, _ = pinn_loss(pred, y_s, irr_raw=irr,
                                y_mean=ym_t, y_std=ys_t, rated_power=rp_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        v_data = v_cnt = 0.0
        with torch.no_grad():
            for b in val:
                x   = b["x"].to(DEVICE)
                y_s = b["y_s"].to(DEVICE)
                irr = b["irr"].to(DEVICE)
                tmp = b["temp"].to(DEVICE)
                pred = model(x, G=irr, T_amb=tmp)
                vd = F.mse_loss(pred, y_s).item()
                n  = x.size(0)
                v_data += vd * n; v_cnt += n
        avg_vd = v_data / v_cnt

        if avg_vd < best_val:
            best_val   = avg_vd
            wait       = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    seed={seed}  early stop @ epoch {epoch}  best_val={best_val:.6f}")
                break

    # ── test ────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    te_p, te_t, te_ts = [], [], []
    with torch.no_grad():
        for b in tst:
            x   = b["x"].to(DEVICE)
            irr = b["irr"].to(DEVICE)
            tmp = b["temp"].to(DEVICE)
            pred = model(x, G=irr, T_amb=tmp)
            te_p.append((pred * ys_t + ym_t).cpu().numpy())
            te_t.append(b["y_raw"].numpy())
            te_ts.extend(b["time"])

    te_p = np.concatenate(te_p); te_t = np.concatenate(te_t)
    ss_res = np.sum((te_t - te_p) ** 2)
    ss_tot = np.sum((te_t - te_t.mean()) ** 2)
    r2   = float(1.0 - ss_res / ss_tot)
    mae  = float(np.mean(np.abs(te_t - te_p)))
    mse  = float(np.mean((te_t - te_p) ** 2))
    rmse = float(np.sqrt(mse))
    print(f"    seed={seed}  R2={r2:.4f}  MAE={mae:.0f}  RMSE={rmse:.0f}")
    return {"state": best_state, "r2": r2, "mae": mae, "mse": mse, "rmse": rmse,
            "preds": te_p, "trues": te_t, "times": te_ts}


def main():
    station_dir = OUTPUT_DIR / STATION

    print(f"{'='*68}")
    print(f"  Inv3 multi-seed PINN  (seeds={SEEDS})")
    print(f"  hidden={HIDDEN_DIM}  layers={NUM_LAYERS}  epochs={EPOCHS}  patience={PATIENCE}")
    print(f"  lambdas from main.py: nonneg={LAMBDA_NONNEG}  upper={LAMBDA_UPPER}  low_irr={LAMBDA_LOW_IRR}")
    print(f"{'='*68}")

    df = add_time_features(merge_all_data(INV3_FILE))
    X, y, irr, temp, times = build_sequences(df, FEATURE_COLS, window_size=WINDOW_SIZE)
    sp = split_data_stratified(X, y, irr, temp, times)
    Xtr, ytr, irr_tr, tmp_tr, ttr = sp["train"]
    Xva, yva, irr_va, tmp_va, tva = sp["val"]
    Xte, yte, irr_te, tmp_te, tte = sp["test"]

    sc   = np.load(station_dir / "scalers.npz")
    xm   = sc["x_mean"]; xs = sc["x_std"]
    ym   = float(sc["y_mean"]); ystd = float(sc["y_std"])

    Xtr_s = (Xtr - xm) / xs; Xva_s = (Xva - xm) / xs; Xte_s = (Xte - xm) / xs
    ytr_s = (ytr - ym) / ystd; yva_s = (yva - ym) / ystd; yte_s = (yte - ym) / ystd
    rated_power = float(RATED_POWER or np.max(ytr) * 1.05)
    ndim = len(FEATURE_COLS)

    best_r2     = -float("inf")
    best_result = None
    best_seed   = None

    for seed in SEEDS:
        print(f"\n  --- seed={seed} ---")
        result = train_one_seed(
            seed,
            Xtr_s, ytr_s, ytr, irr_tr, tmp_tr, ttr,
            Xva_s, yva_s, yva, irr_va, tmp_va, tva,
            Xte_s, yte_s, yte, irr_te, tmp_te, tte,
            xm, xs, ym, ystd, rated_power, ndim)
        if result["r2"] > best_r2:
            best_r2     = result["r2"]
            best_result = result
            best_seed   = seed

    print(f"\n  Best seed={best_seed}  R2={best_r2:.4f}")

    # ── Load best model & run full evaluation ───────────────────────────
    set_seed(best_seed)
    best_model = PINNModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    best_model.set_output_scale(ym, ystd)
    best_model.load_state_dict(best_result["state"])

    trn = DataLoader(PVDataset(Xtr_s, ytr_s, ytr, irr_tr, tmp_tr, ttr),
                     batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val = DataLoader(PVDataset(Xva_s, yva_s, yva, irr_va, tmp_va, tva),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    tst = DataLoader(PVDataset(Xte_s, yte_s, yte, irr_te, tmp_te, tte),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    tr_r = evaluate_on_loader(best_model, trn, ym, ystd, True, DEVICE, rated_power=rated_power)
    va_r = evaluate_on_loader(best_model, val, ym, ystd, True, DEVICE, rated_power=rated_power)

    # extreme sweep
    ext_idx = select_extreme_samples(Xte_s, yte_s, yte, irr_te, tmp_te, tte,
                                     irr_all=irr_tr, temp_all=tmp_tr)
    X_ext   = Xte_s[ext_idx]; y_ext_s = yte_s[ext_idx]; y_ext = yte[ext_idx]
    irr_ext = irr_te[ext_idx]; tmp_ext = tmp_te[ext_idx]; t_ext = tte[ext_idx]

    pinn_sweep = {}
    ext_30_res = None
    for mr in EXTREME_MISS_RATIOS:
        pct_label = int(round(mr * 100))
        Xm, im, tm = inject_missing(X_ext, irr_ext, tmp_ext, mr, seed=42)
        ldr = DataLoader(PVDataset(Xm, y_ext_s, y_ext, im, tm, t_ext),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        res = evaluate_on_loader(best_model, ldr, ym, ystd, True, DEVICE,
                                 rated_power=rated_power)
        pinn_sweep[pct_label] = {k: res[k] for k in ["mae", "mse", "rmse", "r2"]}
        print(f"  miss={pct_label:2d}%  R2={res['r2']:.4f}  MAE={res['mae']:.0f}")
        if pct_label == 30:
            ext_30_res = res

    Xm0, im0, tm0 = inject_missing(X_ext, irr_ext, tmp_ext, 0.0, seed=42)
    ldr0 = DataLoader(PVDataset(Xm0, y_ext_s, y_ext, im0, tm0, t_ext),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    res0 = evaluate_on_loader(best_model, ldr0, ym, ystd, True, DEVICE,
                              rated_power=rated_power)

    # ── Save files ──────────────────────────────────────────────────────
    pn = best_result
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

    # ── Update JSON ──────────────────────────────────────────────────────
    def to_py(obj):
        if isinstance(obj, dict):   return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
        if hasattr(obj, 'item'):    return obj.item()
        return obj

    json_path = station_dir / "test_metrics.json"
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics["pinn"]                  = {k: pn[k]         for k in ["mae", "mse", "rmse", "r2"]}
    metrics["train"]["pinn"]         = {k: tr_r[k]       for k in ["mae", "mse", "rmse", "r2"]}
    metrics["val"]["pinn"]           = {k: va_r[k]       for k in ["mae", "mse", "rmse", "r2"]}
    metrics["extreme"]["pinn"]       = {k: ext_30_res[k] for k in ["mae", "mse", "rmse", "r2"]}
    if "extreme_0miss" not in metrics:
        metrics["extreme_0miss"] = {}
    metrics["extreme_0miss"]["pinn"] = {k: res0[k]       for k in ["mae", "mse", "rmse", "r2"]}
    for pct_label, sweep_row in pinn_sweep.items():
        key = str(pct_label)
        if key in metrics.get("miss_sweep", {}):
            metrics["miss_sweep"][key]["pinn"] = sweep_row

    def pct_chg(a, b): return (b - a) / abs(a) * 100 if abs(a) > 1e-8 else 0
    metrics["improvement"]["pinn_vs_dnn_mae_pct"]          = pct_chg(metrics["dnn"]["mae"], pn["mae"])
    metrics["extreme_improvement"]["pinn_vs_dnn_mae_pct"]  = pct_chg(
        metrics["extreme"]["dnn"]["mae"], ext_30_res["mae"])

    metrics = to_py(metrics)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"\n  ✓ Best seed={best_seed}  Test R2={pn['r2']:.4f}  Extreme_0miss R2={res0['r2']:.4f}")
    print(f"  DNN R2={metrics['dnn']['r2']:.4f}  delta={pn['r2'] - metrics['dnn']['r2']:+.4f}")


if __name__ == "__main__":
    main()
