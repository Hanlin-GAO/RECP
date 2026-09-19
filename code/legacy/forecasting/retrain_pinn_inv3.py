"""
Targeted PINN retrain for Inverter_3 only.

Changes vs retrain_pinn_only.py:
- Reduced physics lambda weights (0.05/0.03/0.05) → MSE dominates, reduces large-error outliers
- CosineAnnealingWarmRestarts instead of ReduceLROnPlateau → multiple escape opportunities
- Larger backbone (hidden_dim=192) and more epochs (500, patience=200)
- Lower backbone LR (1e-4) for finer convergence
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
    PVDataset, evaluate_on_loader,
    select_extreme_samples, inject_missing, set_seed,
    train_and_evaluate,
    FEATURE_COLS, WINDOW_SIZE, BATCH_SIZE, RANDOM_SEED, DEVICE,
    RATED_POWER, OUTPUT_DIR, WEIGHT_DECAY,
    EXTREME_MISS_RATIOS, LOW_IRR_THRESHOLD,
)
from model import PINNModel

# ── Inv3-specific hyperparams ────────────────────────────────────────────────
INV3_FILE = Path(__file__).parent / "Library_Inverter_3.csv"
STATION   = "Inverter_3"

INV3_HIDDEN_DIM   = 192      # larger backbone vs 128
INV3_NUM_LAYERS   = 3
INV3_DROPOUT      = 0.1
INV3_EPOCHS       = 500
INV3_PATIENCE     = 200
INV3_LR_BACKBONE  = 1e-4    # lower than default 3e-4
INV3_LR_PHYSICS   = 3e-5

# Reduced physics constraints → let MSE dominate (reduces large-error outliers → better R2)
INV3_LAMBDA_NONNEG   = 0.05
INV3_LAMBDA_UPPER    = 0.03
INV3_LAMBDA_LOW_IRR  = 0.05

# CosineAnnealing params
COSINE_T0   = 50
COSINE_TMUL = 2


def pinn_loss_inv3(pred_s, y_s, irr_raw, y_mean, y_std, rated_power, **_kw):
    # Huber (smooth_l1) data loss → less sensitivity to large outliers → better R2
    data_loss = F.smooth_l1_loss(pred_s, y_s, beta=0.1)

    pred_raw = pred_s * y_std + y_mean
    rp2 = rated_power ** 2

    L_nn = torch.mean(torch.relu(-pred_raw) ** 2) / rp2
    L_ub = torch.mean(torch.relu(pred_raw - rated_power) ** 2) / rp2

    mask  = (irr_raw < LOW_IRR_THRESHOLD).float()
    n_low = torch.sum(mask) + 1e-8
    L_li  = torch.sum(pred_raw ** 2 * mask) / (n_low * rp2)

    total = (data_loss
             + INV3_LAMBDA_NONNEG  * L_nn
             + INV3_LAMBDA_UPPER   * L_ub
             + INV3_LAMBDA_LOW_IRR * L_li)
    return total, {"total": total.item(), "data": data_loss.item(),
                   "nn": L_nn.item(), "ub": L_ub.item(), "li": L_li.item()}


def train_pinn_cosine(model, train_loader, val_loader, test_loader,
                      y_mean, y_std, rated_power, optimizer,
                      epochs, patience, device):
    """Like train_and_evaluate but uses CosineAnnealingWarmRestarts."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=COSINE_T0, T_mult=COSINE_TMUL, eta_min=1e-6)

    ym  = torch.tensor(y_mean,     dtype=torch.float32, device=device)
    ys  = torch.tensor(y_std,      dtype=torch.float32, device=device)
    rp  = torch.tensor(rated_power, dtype=torch.float32, device=device)

    best_val   = float("inf")
    wait       = 0
    best_state = None
    history    = []

    for epoch in range(1, epochs + 1):
        # ── train ───────────────────────────────────────────────────────
        model.train()
        t_loss = t_data = t_cnt = 0.0
        t_pred, t_true = [], []

        for b in train_loader:
            x    = b["x"].to(device)
            y_s  = b["y_s"].to(device)
            irr  = b["irr"].to(device)
            tmp  = b["temp"].to(device)

            optimizer.zero_grad()
            pred = model(x, G=irr, T_amb=tmp)
            loss, items = pinn_loss_inv3(pred, y_s, irr_raw=irr,
                                         y_mean=ym, y_std=ys, rated_power=rp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = x.size(0)
            t_loss += items["total"] * n
            t_data += items["data"]  * n
            t_cnt  += n
            t_pred.append((pred.detach() * ys + ym).cpu().numpy())
            t_true.append(b["y_raw"].numpy())

        scheduler.step()

        # ── val ─────────────────────────────────────────────────────────
        model.eval()
        v_data = v_cnt = 0.0
        v_pred, v_true = [], []
        with torch.no_grad():
            for b in val_loader:
                x   = b["x"].to(device)
                y_s = b["y_s"].to(device)
                irr = b["irr"].to(device)
                tmp = b["temp"].to(device)
                pred = model(x, G=irr, T_amb=tmp)
                vdl = F.mse_loss(pred, y_s).item()
                n   = x.size(0)
                v_data += vdl * n
                v_cnt  += n
                v_pred.append((pred * ys + ym).cpu().numpy())
                v_true.append(b["y_raw"].numpy())

        t_pred = np.concatenate(t_pred); t_true = np.concatenate(t_true)
        v_pred = np.concatenate(v_pred); v_true = np.concatenate(v_true)
        t_mae = float(np.mean(np.abs(t_true - t_pred)))
        v_mae = float(np.mean(np.abs(v_true - v_pred)))
        t_mse = float(np.mean((t_true - t_pred) ** 2))
        v_mse = float(np.mean((v_true - v_pred) ** 2))
        avg_vd = v_data / v_cnt

        history.append({"epoch": epoch, "train_total": t_loss / t_cnt,
                         "train_data": t_data / t_cnt, "train_mae": t_mae,
                         "train_mse": t_mse, "val_data": avg_vd,
                         "val_mae": v_mae, "val_mse": v_mse})

        if epoch == 1 or epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [PINN-Inv3] {epoch:03d}/{epochs} | "
                  f"val_mse={avg_vd:.6f} | t_MAE={t_mae:.0f} | v_MAE={v_mae:.0f} | lr={lr_now:.2e}")

        if avg_vd < best_val:
            best_val   = avg_vd
            wait       = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"  [PINN-Inv3] Early stop @ epoch {epoch}  (best={best_val:.6f})")
                break

    # ── test ────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    te_p, te_t, te_ts, te_irr = [], [], [], []
    with torch.no_grad():
        for b in test_loader:
            x   = b["x"].to(device)
            irr = b["irr"].to(device)
            tmp = b["temp"].to(device)
            pred = model(x, G=irr, T_amb=tmp)
            te_p.append((pred * ys + ym).cpu().numpy())
            te_t.append(b["y_raw"].numpy())
            te_irr.append(b["irr"].numpy())
            te_ts.extend(b["time"])

    te_p = np.concatenate(te_p); te_t = np.concatenate(te_t)
    te_irr = np.concatenate(te_irr)

    ss_res = np.sum((te_t - te_p) ** 2)
    ss_tot = np.sum((te_t - te_t.mean()) ** 2)
    r2   = 1.0 - ss_res / ss_tot
    mae  = float(np.mean(np.abs(te_t - te_p)))
    mse  = float(np.mean((te_t - te_p) ** 2))
    rmse = float(np.sqrt(mse))

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / STATION / "history_pinn.csv", index=False)

    return {"state": best_state, "r2": r2, "mae": mae, "mse": mse, "rmse": rmse,
            "preds": te_p, "trues": te_t, "times": te_ts}


def retrain_pinn_inv3():
    station_dir = OUTPUT_DIR / STATION
    set_seed(RANDOM_SEED)

    print(f"\n{'=' * 68}")
    print(f"  Inv3 PINN targeted retrain")
    print(f"  hidden={INV3_HIDDEN_DIM}  layers={INV3_NUM_LAYERS}  epochs={INV3_EPOCHS}  patience={INV3_PATIENCE}")
    print(f"  lambdas: nonneg={INV3_LAMBDA_NONNEG}  upper={INV3_LAMBDA_UPPER}  low_irr={INV3_LAMBDA_LOW_IRR}")
    print(f"  scheduler: CosineAnnealingWarmRestarts  T0={COSINE_T0}  Tmul={COSINE_TMUL}")
    print(f"{'=' * 68}")

    # ── Data ──────────────────────────────────────────────────────────────
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

    trn = DataLoader(PVDataset(Xtr_s, ytr_s, ytr, irr_tr, tmp_tr, ttr),
                     batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val = DataLoader(PVDataset(Xva_s, yva_s, yva, irr_va, tmp_va, tva),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    tst = DataLoader(PVDataset(Xte_s, yte_s, yte, irr_te, tmp_te, tte),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Build model ────────────────────────────────────────────────────────
    pinn_model = PINNModel(ndim, INV3_HIDDEN_DIM, INV3_NUM_LAYERS, INV3_DROPOUT).to(DEVICE)
    pinn_model.set_output_scale(ym, ystd)
    # Start with near-zero physics gate (gate≈0.018): let LSTM backbone dominate first
    with torch.no_grad():
        pinn_model.gate_param.fill_(-4.0)

    phys_names      = {"log_alpha", "beta", "T_ref", "log_noct", "gate_param"}
    phys_params     = [p for n, p in pinn_model.named_parameters() if n in phys_names]
    backbone_params = [p for n, p in pinn_model.named_parameters() if n not in phys_names]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": INV3_LR_BACKBONE},
         {"params": phys_params,     "lr": INV3_LR_PHYSICS}],
        weight_decay=WEIGHT_DECAY)

    # ── Train ──────────────────────────────────────────────────────────────
    pn = train_pinn_cosine(pinn_model, trn, val, tst,
                           ym, ystd, rated_power, optimizer,
                           INV3_EPOCHS, INV3_PATIENCE, DEVICE)
    pinn_model.load_state_dict(pn["state"])
    print(f"\n  Test → R2={pn['r2']:.4f}  MAE={pn['mae']:.0f}  RMSE={pn['rmse']:.0f}")

    # ── Train / val metrics ────────────────────────────────────────────────
    tr_r = evaluate_on_loader(pinn_model, trn, ym, ystd, True, DEVICE, rated_power=rated_power)
    va_r = evaluate_on_loader(pinn_model, val, ym, ystd, True, DEVICE, rated_power=rated_power)

    # ── Extreme sweep ──────────────────────────────────────────────────────
    ext_idx = select_extreme_samples(Xte_s, yte_s, yte, irr_te, tmp_te, tte,
                                     irr_all=irr_tr, temp_all=tmp_tr)
    X_ext  = Xte_s[ext_idx]; y_ext_s = yte_s[ext_idx]; y_ext = yte[ext_idx]
    irr_ext = irr_te[ext_idx]; tmp_ext = tmp_te[ext_idx]; t_ext = tte[ext_idx]

    pinn_sweep  = {}
    ext_30_res  = None
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

    # extreme_0miss
    Xm0, im0, tm0 = inject_missing(X_ext, irr_ext, tmp_ext, 0.0, seed=RANDOM_SEED)
    ldr0 = DataLoader(PVDataset(Xm0, y_ext_s, y_ext, im0, tm0, t_ext),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    res0 = evaluate_on_loader(pinn_model, ldr0, ym, ystd, True, DEVICE,
                              rated_power=rated_power)

    # ── Update JSON ────────────────────────────────────────────────────────
    json_path = station_dir / "test_metrics.json"
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metrics["pinn"]                  = {k: pn[k]        for k in ["mae", "mse", "rmse", "r2"]}
    metrics["train"]["pinn"]         = {k: tr_r[k]      for k in ["mae", "mse", "rmse", "r2"]}
    metrics["val"]["pinn"]           = {k: va_r[k]      for k in ["mae", "mse", "rmse", "r2"]}
    metrics["extreme"]["pinn"]       = {k: ext_30_res[k] for k in ["mae", "mse", "rmse", "r2"]}
    if "extreme_0miss" not in metrics:
        metrics["extreme_0miss"] = {}
    metrics["extreme_0miss"]["pinn"] = {k: res0[k]      for k in ["mae", "mse", "rmse", "r2"]}
    for pct_label, sweep_row in pinn_sweep.items():
        key = str(pct_label)
        if key in metrics.get("miss_sweep", {}):
            metrics["miss_sweep"][key]["pinn"] = sweep_row

    def pct_chg(a, b): return (b - a) / abs(a) * 100 if abs(a) > 1e-8 else 0
    metrics["improvement"]["pinn_vs_dnn_mae_pct"] = pct_chg(metrics["dnn"]["mae"], pn["mae"])
    metrics["extreme_improvement"]["pinn_vs_dnn_mae_pct"] = pct_chg(
        metrics["extreme"]["dnn"]["mae"], ext_30_res["mae"])

    # Convert numpy types to native Python for JSON serialization
    def to_py(obj):
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_py(v) for v in obj]
        if hasattr(obj, 'item'):
            return obj.item()
        return obj
    metrics = to_py(metrics)

    # ── Save files first (before JSON, so they're never lost on crash) ─────
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

    # ── Write JSON ─────────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"\n  ✓ Inv3  Test R2={pn['r2']:.4f}  Extreme_0miss R2={res0['r2']:.4f}")
    dnn_r2 = metrics["dnn"]["r2"]
    print(f"  DNN  R2={dnn_r2:.4f}  PINN vs DNN: {pn['r2'] - dnn_r2:+.4f}")


if __name__ == "__main__":
    retrain_pinn_inv3()
