"""
Wind power prediction — training & evaluation pipeline.
Structure mirrors main.py (photovoltaic) but adapted for wind turbine data.
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from wind_model import DNNModel, BaselineModel, GRUModel, TCNModel, TransformerModel, WindPINNModel


# =========================
# Basic configuration
# =========================
from forecast_paths import WIND_DATA, WIND_OUTPUT
DATA_DIR = WIND_DATA
OUTPUT_DIR = WIND_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TURBINE_FILES = [
    ("Turbine_1", DATA_DIR / "Wind 1.xlsx"),
    ("Turbine_2", DATA_DIR / "Wind 2.xlsx"),
    ("Turbine_3", DATA_DIR / "Wind 3.xlsx"),
    ("Turbine_4", DATA_DIR / "Wind 4.xlsx"),
    ("Turbine_5", DATA_DIR / "Wind 5.xlsx"),
    ("Turbine_6", DATA_DIR / "Wind 6.xlsx"),
]

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 24          # Six hours of context: 24 samples at 15-minute intervals
HORIZON = 1
BATCH_SIZE = 256
EPOCHS = 150
LR = 5e-4
PATIENCE = 30
SCHED_PATIENCE = 10
SCHED_FACTOR = 0.5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4

# Physics-constraint weights used in the historical wind experiment
LAMBDA_NONNEG = 0.5
LAMBDA_UPPER = 0.2
LAMBDA_LOW_WIND = 0.5

LOW_WIND_THRESHOLD = 3.0   # m/s — typical cut-in speed
RATED_POWER = None          # Infer the bound from data when no rated power is supplied

# PINN settings; historical transfer learning uses the shared GRU backbone
PINN_PATIENCE = 80
PINN_WARMUP_EPOCHS = 25    # Phase 1: freeze the backbone and fit physical parameters
PINN_LR_BACKBONE = 5e-5    # Phase 2: fine-tune the backbone at a lower learning rate
PINN_LR_PHYSICS = 1e-3     # Learning rate for physical parameters and the gate
PINN_PHASE2_LR_WARMUP = 10 # Number of backbone learning-rate warmup epochs in phase 2
PINN_SCHED_PATIENCE = 20   # PINN scheduler patience

# Extreme-condition evaluation settings
EXTREME_WS_HIGH = 0.90      # Wind speed above the 90th percentile
EXTREME_WS_LOW = 0.10       # Wind speed below the 10th percentile
EXTREME_TEMP_LOW = 0.10
EXTREME_TEMP_HIGH = 0.90
EXTREME_MISS_RATIOS = [r / 100 for r in range(0, 55, 5)]


# =========================
# Utility functions
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_time_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.to_datetime(s, errors="coerce")


def find_col(df_cols, keywords):
    """Match column names by keyword after normalizing whitespace."""
    norm = {c: " ".join(c.split()) for c in df_cols}
    for kw in keywords:
        kw_norm = " ".join(kw.split())
        for orig, n in norm.items():
            if kw_norm.lower() in n.lower():
                return orig
    return None


# =========================
# Data loading
# =========================
def load_wind_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    # Match column names across the original workbook formats
    time_col = find_col(df.columns, ["Time(year-month-day"])
    ws_col = find_col(df.columns, [
        "Wind speed - at the height of wheel hub (m/s)",
        "Wind speed - at the height of wheel hub  (m/s)",
    ])
    wd_col = find_col(df.columns, [
        "Wind direction - at the height of wheel hub",
        # Wind 6: direction column is mislabelled with "Wind speed" but has ˚ unit
        "Wind speed - at the height of wheel hub  (˚)",
        "Wind speed - at the height of wheel hub (˚)",
    ])
    temp_col = find_col(df.columns, ["Air temperature"])
    power_col = find_col(df.columns, ["Power (MW)"])

    if not all([time_col, ws_col, wd_col, temp_col, power_col]):
        missing = []
        for name, col in [("Time", time_col), ("WindSpeed", ws_col),
                           ("WindDir", wd_col), ("Temp", temp_col),
                           ("Power", power_col)]:
            if col is None:
                missing.append(name)
        raise ValueError(f"缺少列: {missing}, 可用列: {list(df.columns)}")

    df = df.rename(columns={
        time_col: "Time",
        ws_col: "WindSpeed",
        wd_col: "WindDirection",
        temp_col: "Temperature",
        power_col: "Power",
    })

    df["Time"] = parse_time_series(df["Time"])
    for c in ["WindSpeed", "WindDirection", "Temperature", "Power"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["Time", "WindSpeed", "WindDirection", "Temperature", "Power"]].copy()
    df = df.dropna().sort_values("Time").reset_index(drop=True)

    # Historical experiment retains only the 2020 records
    df = df[df["Time"].dt.year == 2020].reset_index(drop=True)

    # Average duplicate timestamps
    df = df.groupby("Time", as_index=False).mean(numeric_only=True)
    return df


# =========================
# Feature construction
# =========================
def add_time_features(df):
    t = pd.to_datetime(df["Time"])
    hour = t.dt.hour + t.dt.minute / 60.0
    doy = t.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

    # Represent circular wind direction using sine and cosine
    wd_rad = np.deg2rad(df["WindDirection"])
    df["WindDir_sin"] = np.sin(wd_rad)
    df["WindDir_cos"] = np.cos(wd_rad)

    # First difference of wind speed
    df["WindSpeed_diff"] = df["WindSpeed"].diff().fillna(0.0)

    return df


FEATURE_COLS = [
    "WindSpeed", "WindDir_sin", "WindDir_cos", "Temperature",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "WindSpeed_diff",
]


# =========================
# Sequence construction
# =========================
def build_sequences(df, feature_cols, target_col="Power",
                    window_size=24, horizon=1):
    feat = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    ws = df["WindSpeed"].values.astype(np.float32)
    temp = df["Temperature"].values.astype(np.float32)
    tv = df["Time"].values

    X, y, wv, tp, ts = [], [], [], [], []
    for i in range(window_size, len(df) - horizon + 1):
        ti = i + horizon - 1
        X.append(feat[i - window_size: i])
        y.append(target[ti])
        wv.append(ws[ti])
        tp.append(temp[ti])
        ts.append(tv[ti])

    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(wv, dtype=np.float32), np.array(tp, dtype=np.float32),
            np.array(ts))


def split_data_stratified(X, y, ws, temp, times):
    """Historical month-wise chronological train/validation/test split."""
    months = pd.to_datetime(times).month
    train_idx, val_idx, test_idx = [], [], []
    for m in sorted(np.unique(months)):
        idx = np.where(months == m)[0]
        n = len(idx)
        i1 = int(n * TRAIN_RATIO)
        i2 = int(n * (TRAIN_RATIO + VAL_RATIO))
        train_idx.extend(idx[:i1])
        val_idx.extend(idx[i1:i2])
        test_idx.extend(idx[i2:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    return {
        "train": (X[train_idx], y[train_idx], ws[train_idx], temp[train_idx], times[train_idx]),
        "val":   (X[val_idx],   y[val_idx],   ws[val_idx],   temp[val_idx],   times[val_idx]),
        "test":  (X[test_idx],  y[test_idx],  ws[test_idx],  temp[test_idx],  times[test_idx]),
    }


def fit_scalers(X_train, y_train):
    xm = X_train.reshape(-1, X_train.shape[-1]).mean(0)
    xs = X_train.reshape(-1, X_train.shape[-1]).std(0)
    xs[xs < 1e-8] = 1.0
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    if y_max - y_min < 1e-8:
        y_max = y_min + 1.0
    return xm, xs, y_min, y_max


# =========================
# Dataset
# =========================
class WindDataset(Dataset):
    def __init__(self, X_s, y_s, y_raw, ws, temp, times):
        self.X = torch.tensor(X_s, dtype=torch.float32)
        self.y_s = torch.tensor(y_s, dtype=torch.float32)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32)
        self.ws = torch.tensor(ws, dtype=torch.float32)
        self.temp = torch.tensor(temp, dtype=torch.float32)
        self.times = pd.to_datetime(times).strftime("%Y-%m-%d %H:%M:%S").tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y_s": self.y_s[idx], "y_raw": self.y_raw[idx],
                "ws": self.ws[idx], "temp": self.temp[idx], "time": self.times[idx]}


# =========================
# Loss functions
# =========================
def postprocess_predictions(preds, ws_raw, rated_power):
    """Apply the historical low-wind zeroing rule and clip predictions to physical bounds."""
    preds = np.clip(preds, 0.0, rated_power)
    preds[ws_raw < LOW_WIND_THRESHOLD] = 0.0
    return preds


def baseline_loss(pred_s, y_s, **_kw):
    loss = F.mse_loss(pred_s, y_s)
    return loss, {"total": loss.item(), "data": loss.item()}


def pinn_loss(pred_s, y_s, ws_raw, y_mean, y_std, rated_power, **_kw):
    data_loss = F.mse_loss(pred_s, y_s)

    pred_raw = pred_s * y_std + y_mean
    rp2 = rated_power ** 2

    # Non-negativity penalty
    L_nn = torch.mean(torch.relu(-pred_raw) ** 2) / rp2

    # Upper-bound penalty
    L_ub = torch.mean(torch.relu(pred_raw - rated_power) ** 2) / rp2

    # Penalize positive power below the cut-in wind speed
    mask = (ws_raw < LOW_WIND_THRESHOLD).float()
    n_low = torch.sum(mask) + 1e-8
    L_lw = torch.sum(pred_raw ** 2 * mask) / (n_low * rp2)

    total = data_loss + LAMBDA_NONNEG * L_nn + LAMBDA_UPPER * L_ub + LAMBDA_LOW_WIND * L_lw
    return total, {"total": total.item(), "data": data_loss.item(),
                    "nn": L_nn.item(), "ub": L_ub.item(), "lw": L_lw.item()}


# =========================
# Training and evaluation
# =========================
def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       y_mean, y_std, rated_power,
                       loss_fn, is_pinn, label, device,
                       optimizer=None, patience=None, epochs=None,
                       sched_patience=None, backbone_lr_warmup=0):

    patience = patience or PATIENCE
    epochs = epochs or EPOCHS
    sched_patience = sched_patience or SCHED_PATIENCE
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR, patience=sched_patience,
        min_lr=1e-6)

    # Store the target backbone learning rate for phase-2 warmup
    target_backbone_lr = (optimizer.param_groups[0]['lr']
                          if backbone_lr_warmup > 0 else None)

    ym = torch.tensor(y_mean, dtype=torch.float32, device=device)
    ys = torch.tensor(y_std, dtype=torch.float32, device=device)
    rp = torch.tensor(rated_power, dtype=torch.float32, device=device)

    best_val = float("inf")
    wait = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        # Increase the backbone learning rate from one tenth of its target
        if backbone_lr_warmup > 0 and epoch <= backbone_lr_warmup:
            frac = epoch / backbone_lr_warmup
            optimizer.param_groups[0]['lr'] = target_backbone_lr * (0.1 + 0.9 * frac)

        # ---- train ----
        model.train()
        t_loss, t_data, t_cnt = 0.0, 0.0, 0
        t_pred, t_true = [], []

        for b in train_loader:
            x = b["x"].to(device)
            y_s = b["y_s"].to(device)
            y_raw = b["y_raw"].to(device)
            ws = b["ws"].to(device)
            tmp = b["temp"].to(device)

            optimizer.zero_grad()
            if is_pinn:
                pred = model(x, v=ws, T_amb=tmp)
            else:
                pred = model(x)

            loss, items = loss_fn(pred, y_s, ws_raw=ws,
                                  y_mean=ym, y_std=ys, rated_power=rp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = x.size(0)
            t_loss += items["total"] * n
            t_data += items["data"] * n
            t_cnt += n
            t_pred.append((pred.detach() * ys + ym).cpu().numpy())
            t_true.append(y_raw.cpu().numpy())

        # ---- val ----
        model.eval()
        v_data, v_cnt = 0.0, 0
        v_pred, v_true = [], []
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device)
                y_s = b["y_s"].to(device)
                y_raw = b["y_raw"].to(device)
                ws = b["ws"].to(device)
                tmp = b["temp"].to(device)

                if is_pinn:
                    pred = model(x, v=ws, T_amb=tmp)
                else:
                    pred = model(x)
                vdl = F.mse_loss(pred, y_s).item()
                n = x.size(0)
                v_data += vdl * n
                v_cnt += n
                v_pred.append((pred * ys + ym).cpu().numpy())
                v_true.append(y_raw.cpu().numpy())

        t_pred = np.concatenate(t_pred); t_true = np.concatenate(t_true)
        v_pred = np.concatenate(v_pred); v_true = np.concatenate(v_true)
        t_mae = float(np.mean(np.abs(t_true - t_pred)))
        v_mae = float(np.mean(np.abs(v_true - v_pred)))
        t_mse = float(np.mean((t_true - t_pred) ** 2))
        v_mse = float(np.mean((v_true - v_pred) ** 2))
        avg_vd = v_data / v_cnt
        # Delay ReduceLROnPlateau until backbone warmup finishes
        if epoch > backbone_lr_warmup:
            scheduler.step(avg_vd)

        history.append({"epoch": epoch, "train_total": t_loss / t_cnt,
                         "train_data": t_data / t_cnt, "train_mae": t_mae,
                         "train_mse": t_mse,
                         "val_data": avg_vd, "val_mae": v_mae,
                         "val_mse": v_mse})

        if epoch == 1 or epoch % 5 == 0:
            print(f"  [{label}] {epoch:03d}/{epochs} | "
                  f"val_loss={avg_vd:.6f} | t_MAE={t_mae:.4f} | v_MAE={v_mae:.4f}")

        if avg_vd < best_val:
            best_val = avg_vd
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{label}] Early stop @ epoch {epoch}")
                break

    # ---- test ----
    model.load_state_dict(best_state)
    model.eval()
    te_p, te_t, te_ts, te_ws = [], [], [], []
    with torch.no_grad():
        for b in test_loader:
            x = b["x"].to(device)
            ws = b["ws"].to(device)
            tmp = b["temp"].to(device)
            if is_pinn:
                pred = model(x, v=ws, T_amb=tmp)
            else:
                pred = model(x)
            te_p.append((pred * ys + ym).cpu().numpy())
            te_t.append(b["y_raw"].numpy())
            te_ws.append(b["ws"].numpy())
            te_ts.extend(b["time"])

    te_p = np.concatenate(te_p); te_t = np.concatenate(te_t)
    te_ws = np.concatenate(te_ws)
    te_p = postprocess_predictions(te_p, te_ws, rated_power)
    mae = float(np.mean(np.abs(te_t - te_p)))
    mse = float(np.mean((te_t - te_p) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((te_t - te_p) ** 2)
    ss_tot = np.sum((te_t - np.mean(te_t)) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
            "best_val": float(best_val),
            "history": pd.DataFrame(history),
            "preds": te_p, "trues": te_t, "times": te_ts,
            "state": best_state}


# =========================
# Evaluation under extreme conditions
# =========================
def select_extreme_samples(Xte_s, yte_s, yte, ws_te, tmp_te, tte,
                           ws_all, temp_all):
    """Select samples with high/low wind speed or extreme temperature."""
    ws_q_high = np.percentile(ws_all, EXTREME_WS_HIGH * 100)
    ws_q_low = np.percentile(ws_all, EXTREME_WS_LOW * 100)
    temp_q_low = np.percentile(temp_all, EXTREME_TEMP_LOW * 100)
    temp_q_high = np.percentile(temp_all, EXTREME_TEMP_HIGH * 100)

    mask = ((ws_te > ws_q_high) |
            (ws_te < ws_q_low) |
            (tmp_te < temp_q_low) |
            (tmp_te > temp_q_high))
    idx = np.where(mask)[0]

    if len(idx) < 50:
        ws_dev = np.abs(ws_te - np.median(ws_te))
        tmp_dev = np.abs(tmp_te - np.median(tmp_te))
        combined = ws_dev / (ws_dev.max() + 1e-8) + tmp_dev / (tmp_dev.max() + 1e-8)
        k = max(50, int(len(ws_te) * 0.2))
        idx = np.argsort(combined)[-k:]

    print(f"  极端场景样本数: {len(idx)} / {len(ws_te)} "
          f"(ws>{ws_q_high:.1f} or ws<{ws_q_low:.1f} m/s, "
          f"temp<{temp_q_low:.1f} or >{temp_q_high:.1f}°C)")
    return idx


def inject_missing(X, ws, tmp, miss_ratio, seed=42):
    """Historical masking: zero temporal features while leaving scalar physics inputs unchanged. This is not an equal-input comparison."""
    X_out = X.copy()
    if miss_ratio > 0:
        rng = np.random.RandomState(seed)
        miss_mask = rng.random(X_out.shape) < miss_ratio
        X_out[miss_mask] = 0.0
    return X_out, ws, tmp


def evaluate_on_loader(model, loader, y_mean, y_std, is_pinn, device,
                       rated_power=None):
    ym = torch.tensor(y_mean, dtype=torch.float32, device=device)
    ys = torch.tensor(y_std, dtype=torch.float32, device=device)

    model.eval()
    te_p, te_t, te_ts, te_ws = [], [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            ws = b["ws"].to(device)
            tmp = b["temp"].to(device)
            if is_pinn:
                pred = model(x, v=ws, T_amb=tmp)
            else:
                pred = model(x)
            te_p.append((pred * ys + ym).cpu().numpy())
            te_t.append(b["y_raw"].numpy())
            te_ws.append(b["ws"].numpy())
            te_ts.extend(b["time"])

    te_p = np.concatenate(te_p)
    te_t = np.concatenate(te_t)
    te_ws = np.concatenate(te_ws)
    if rated_power is not None:
        te_p = postprocess_predictions(te_p, te_ws, rated_power)
    mae = float(np.mean(np.abs(te_t - te_p)))
    mse = float(np.mean((te_t - te_p) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((te_t - te_p) ** 2)
    ss_tot = np.sum((te_t - np.mean(te_t)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
            "preds": te_p, "trues": te_t, "times": te_ts}


# =========================
# Entry point
# =========================
def run_turbine(turbine_name, turbine_file):
    """Run the historical training and evaluation pipeline for one wind source."""
    station_dir = OUTPUT_DIR / turbine_name
    station_dir.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_SEED)
    print(f"\n{'#' * 76}")
    print(f"# 机组: {turbine_name} ({turbine_file.name})")
    print(f"{'#' * 76}")

    # Data
    print("加载数据...")
    df = load_wind_data(turbine_file)
    df = add_time_features(df)
    df.to_csv(station_dir / "merged_wind_data.csv", index=False, encoding="utf-8-sig")
    print(f"样本: {len(df)},  特征: {FEATURE_COLS}")

    X, y, ws, temp, times = build_sequences(df, FEATURE_COLS, window_size=WINDOW_SIZE)
    print("  -> 使用按月分层划分")
    sp = split_data_stratified(X, y, ws, temp, times)
    Xtr, ytr, ws_tr, tmp_tr, ttr = sp["train"]
    Xva, yva, ws_va, tmp_va, tva = sp["val"]
    Xte, yte, ws_te, tmp_te, tte = sp["test"]
    print(f"序列: {len(X)}  (train {len(Xtr)} / val {len(Xva)} / test {len(Xte)})")

    xm, xs, ymin, ymax = fit_scalers(Xtr, ytr)
    Xtr_s = (Xtr - xm) / xs; Xva_s = (Xva - xm) / xs; Xte_s = (Xte - xm) / xs
    ytr_s = (ytr - ymin) / (ymax - ymin)
    yva_s = (yva - ymin) / (ymax - ymin)
    yte_s = (yte - ymin) / (ymax - ymin)
    ym = ymin; ystd = ymax - ymin
    np.savez(station_dir / "scalers.npz", x_mean=xm, x_std=xs, y_mean=ym, y_std=ystd)

    rated_power = float(RATED_POWER or np.max(ytr) * 1.05)
    print(f"额定功率 = {rated_power:.4f} MW")

    trn = DataLoader(WindDataset(Xtr_s, ytr_s, ytr, ws_tr, tmp_tr, ttr),
                     batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val = DataLoader(WindDataset(Xva_s, yva_s, yva, ws_va, tmp_va, tva),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    tst = DataLoader(WindDataset(Xte_s, yte_s, yte, ws_te, tmp_te, tte),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ndim = len(FEATURE_COLS)

    # Experiment 1: DNN
    print("\n" + "=" * 60)
    print("实验 1: DNN（纯前馈神经网络）")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    dnn_model = DNNModel(ndim, HIDDEN_DIM, WINDOW_SIZE, DROPOUT).to(DEVICE)
    dnn = train_and_evaluate(dnn_model, trn, val, tst,
                             ym, ystd, rated_power,
                             baseline_loss, False, "DNN", DEVICE)

    # Experiment 2: LSTM
    print("\n" + "=" * 60)
    print("实验 2: LSTM（纯数据驱动）")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    bl_model = BaselineModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    bl = train_and_evaluate(bl_model, trn, val, tst,
                            ym, ystd, rated_power,
                            baseline_loss, False, "LSTM", DEVICE)

    # Experiment 3: GRU
    print("\n" + "=" * 60)
    print("实验 3: GRU（门控循环单元）")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    gru_model = GRUModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    gru = train_and_evaluate(gru_model, trn, val, tst,
                             ym, ystd, rated_power,
                             baseline_loss, False, "GRU", DEVICE)

    # Experiment 4: TCN
    print("\n" + "=" * 60)
    print("实验 4: TCN（时序卷积网络）")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    tcn_model = TCNModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    tcn = train_and_evaluate(tcn_model, trn, val, tst,
                             ym, ystd, rated_power,
                             baseline_loss, False, "TCN", DEVICE)

    # Experiment 5: Transformer
    print("\n" + "=" * 60)
    print("实验 5: Transformer（多头自注意力时序编码器）")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    tr_model = TransformerModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    tr = train_and_evaluate(tr_model, trn, val, tst,
                            ym, ystd, rated_power,
                            baseline_loss, False, "Transformer", DEVICE)

    # Experiment 6: wind PINN with GRU transfer, a physical power curve, and a learned gate
    print("\n" + "=" * 60)
    print("实验 6: Wind PINN（物理功率曲线 + GRU 迁移学习 + 自适应门控）")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    pinn_model = WindPINNModel(ndim, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
                               rated_power_init=rated_power * 0.9).to(DEVICE)
    pinn_model.set_output_scale(ym, ystd)
    # Initialize the physics-model backbone from the trained GRU
    pinn_model.backbone.load_state_dict(gru_model.state_dict())
    print("  -> 已从 GRU 迁移 backbone 权重")

    # Phase 1: fit physical parameters and the gate with a frozen backbone
    print(f"  -> Phase 1: 校准物理分支 ({PINN_WARMUP_EPOCHS} epochs, backbone frozen)")
    for p in pinn_model.backbone.parameters():
        p.requires_grad = False
    warmup_params = [p for p in pinn_model.parameters() if p.requires_grad]
    warmup_optim = torch.optim.AdamW(warmup_params, lr=PINN_LR_PHYSICS, weight_decay=0)
    ym_t = torch.tensor(ym, dtype=torch.float32, device=DEVICE)
    ys_t = torch.tensor(ystd, dtype=torch.float32, device=DEVICE)
    rp_t = torch.tensor(rated_power, dtype=torch.float32, device=DEVICE)
    for wep in range(1, PINN_WARMUP_EPOCHS + 1):
        pinn_model.train()
        w_loss_sum, w_cnt = 0.0, 0
        for b in trn:
            x = b["x"].to(DEVICE); y_s = b["y_s"].to(DEVICE)
            ws = b["ws"].to(DEVICE); tmp = b["temp"].to(DEVICE)
            warmup_optim.zero_grad()
            pred = pinn_model(x, v=ws, T_amb=tmp)
            loss, _ = pinn_loss(pred, y_s, ws_raw=ws, y_mean=ym_t, y_std=ys_t, rated_power=rp_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), 1.0)
            warmup_optim.step()
            w_loss_sum += loss.item() * x.size(0); w_cnt += x.size(0)
        if wep % 5 == 0:
            print(f"  [PINN warmup] {wep:02d}/{PINN_WARMUP_EPOCHS} | loss={w_loss_sum/w_cnt:.6f}")

    # Phase 2: unfreeze all parameters and use separate learning rates
    print("  -> Phase 2: 全参数微调 (backbone unfrozen)")
    for p in pinn_model.backbone.parameters():
        p.requires_grad = True
    backbone_params = [p for n, p in pinn_model.named_parameters()
                       if n.startswith("backbone.")]
    other_params = [p for n, p in pinn_model.named_parameters()
                    if not n.startswith("backbone.")]
    pinn_optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": PINN_LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": other_params, "lr": PINN_LR_PHYSICS, "weight_decay": 0},
    ])
    pn = train_and_evaluate(pinn_model, trn, val, tst,
                            ym, ystd, rated_power,
                            pinn_loss, True, "PINN", DEVICE,
                            optimizer=pinn_optim, patience=PINN_PATIENCE,
                            sched_patience=PINN_SCHED_PATIENCE,
                            backbone_lr_warmup=PINN_PHASE2_LR_WARMUP)

    # Model comparison
    def pct(a, b): return (b - a) / abs(a) * 100 if abs(a) > 1e-8 else 0

    results = {"DNN": dnn, "LSTM": bl, "GRU": gru, "TCN": tcn, "Transformer": tr, "PINN": pn}

    print("\n" + "=" * 88)
    print("       DNN  vs  LSTM  vs  GRU  vs  TCN  vs  Transformer  vs  PINN 对比结果")
    print("=" * 88)
    print(f"{'指标':<12}{'DNN':>12}{'LSTM':>12}{'GRU':>12}{'TCN':>12}{'Trans':>12}{'PINN':>12}")
    print("-" * 84)
    for nm, k in [("MAE(MW)", "mae"), ("RMSE(MW)", "rmse"),
                  ("MSE", "mse"), ("R2", "r2")]:
        vals = [results[n][k] for n in ["DNN", "LSTM", "GRU", "TCN", "Transformer", "PINN"]]
        print(f"{nm:<12}" + "".join(f"{v:>12.4f}" for v in vals))
    print("-" * 84)
    for ref_name, tgt_name in [("DNN", "LSTM"), ("DNN", "GRU"), ("DNN", "TCN"),
                                ("DNN", "Transformer"), ("LSTM", "PINN"), ("DNN", "PINN")]:
        ref, tgt = results[ref_name], results[tgt_name]
        print(f"{'%s vs %s' % (tgt_name, ref_name):>20} | MAE: {pct(ref['mae'], tgt['mae']):+.2f}%  R2: {pct(ref['r2'], tgt['r2']):+.2f}%")
    print("=" * 76)

    # Evaluate on training and validation subsets
    print("\n评估训练集和验证集上的最优模型指标...")
    model_objs = {"DNN": dnn_model, "LSTM": bl_model, "GRU": gru_model,
                  "TCN": tcn_model, "Transformer": tr_model, "PINN": pinn_model}
    for name in model_objs:
        is_p = (name == "PINN")
        model_objs[name].load_state_dict(results[name]["state"])
        tr_r = evaluate_on_loader(model_objs[name], trn, ym, ystd, is_p, DEVICE,
                                  rated_power=rated_power)
        va_r = evaluate_on_loader(model_objs[name], val, ym, ystd, is_p, DEVICE,
                                  rated_power=rated_power)
        results[name]["train_metrics"] = {k: tr_r[k] for k in ["mae", "mse", "rmse", "r2"]}
        results[name]["val_metrics"] = {k: va_r[k] for k in ["mae", "mse", "rmse", "r2"]}

    print(f"\n{'集合':<8}{'指标':<12}{'DNN':>10}{'LSTM':>10}{'GRU':>10}{'TCN':>10}{'Trans':>10}{'PINN':>10}")
    print("-" * 80)
    for split_label, split_key in [("Train", "train_metrics"), ("Val", "val_metrics"),
                                    ("Test", None)]:
        for nm, k in [("MAE(MW)", "mae"), ("RMSE(MW)", "rmse"), ("R2", "r2")]:
            if split_key:
                vals = [results[n][split_key][k] for n in ["DNN", "LSTM", "GRU", "TCN", "Transformer", "PINN"]]
            else:
                vals = [results[n][k] for n in ["DNN", "LSTM", "GRU", "TCN", "Transformer", "PINN"]]
            print(f"{split_label:<8}{nm:<12}" + "".join(f"{v:>10.4f}" for v in vals))
    print("-" * 80)

    # Evaluate extreme conditions
    print("\n" + "=" * 60)
    print("极端场景评估（极端天气 + 传感器缺失扫描 0%~50%）")
    print("=" * 60)

    ext_idx = select_extreme_samples(
        Xte_s, yte_s, yte, ws_te, tmp_te, tte,
        ws_all=ws_tr, temp_all=tmp_tr)

    X_ext_raw = Xte_s[ext_idx]
    y_ext_s = yte_s[ext_idx]
    y_ext = yte[ext_idx]
    ws_ext = ws_te[ext_idx]
    tmp_ext = tmp_te[ext_idx]
    t_ext = tte[ext_idx]

    models_map = [
        ("DNN",         dnn_model,  dnn["state"], False),
        ("LSTM",        bl_model,   bl["state"],  False),
        ("GRU",         gru_model,  gru["state"], False),
        ("TCN",         tcn_model,  tcn["state"], False),
        ("Transformer", tr_model,   tr["state"],  False),
        ("PINN",        pinn_model, pn["state"],  True),
    ]

    sweep_results = {}
    for mr in EXTREME_MISS_RATIOS:
        pct_label = int(round(mr * 100))
        X_miss, ws_miss, tmp_miss = inject_missing(X_ext_raw, ws_ext, tmp_ext, mr, seed=RANDOM_SEED)
        loader = DataLoader(
            WindDataset(X_miss, y_ext_s, y_ext, ws_miss, tmp_miss, t_ext),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        row = {}
        for name, mdl, state, is_p in models_map:
            mdl.load_state_dict(state)
            row[name] = evaluate_on_loader(mdl, loader, ym, ystd, is_p, DEVICE,
                                             rated_power=rated_power)
        sweep_results[pct_label] = row
        line = f"  miss={pct_label:2d}%"
        for name, _, _, _ in models_map:
            line += f"  {name} MAE={row[name]['mae']:.4f}"
        print(line)

    # Unmasked extreme-condition results for scatter and error plots
    ext_results_0 = sweep_results[0]
    # Extreme-condition results with 30% masking
    ext_results = sweep_results[30]
    print(f"\n--- 极端场景 (miss=30%) 详细指标 ---")
    print(f"{'指标':<12}{'DNN':>12}{'LSTM':>12}{'GRU':>12}{'TCN':>12}{'Trans':>12}{'PINN':>12}")
    print("-" * 84)
    for nm, k in [("MAE(MW)", "mae"), ("RMSE(MW)", "rmse"),
                  ("MSE", "mse"), ("R2", "r2")]:
        vals = [ext_results[n][k] for n in ["DNN", "LSTM", "GRU", "TCN", "Transformer", "PINN"]]
        print(f"{nm:<12}" + "".join(f"{v:>12.4f}" for v in vals))
    print("-" * 84)
    ep = ext_results["PINN"]
    for ref in ["DNN", "LSTM", "GRU", "TCN", "Transformer"]:
        er = ext_results[ref]
        print(f"{'PINN vs %s' % ref:>20} | MAE: {pct(er['mae'], ep['mae']):+.2f}%  R2: {pct(er['r2'], ep['r2']):+.2f}%")
    print("=" * 76)

    # Save outputs
    # Ensure the output directory exists before saving
    station_dir.mkdir(parents=True, exist_ok=True)
    ALL_TAGS = {"DNN": "dnn", "LSTM": "lstm", "GRU": "gru", "TCN": "tcn", "Transformer": "trans", "PINN": "pinn"}
    for name, tag in ALL_TAGS.items():
        res = results[name]
        res["history"].to_csv(station_dir / f"history_{tag}.csv",
                              index=False, encoding="utf-8-sig")
        pd.DataFrame({"Time": pd.to_datetime(res["times"]),
                       "True_Power": res["trues"],
                       "Pred_Power": res["preds"]}).to_csv(
            station_dir / f"test_predictions_{tag}.csv",
            index=False, encoding="utf-8-sig")
        # Predictions for extreme conditions with 30% masking
        eres = ext_results[name]
        pd.DataFrame({"Time": pd.to_datetime(eres["times"]),
                       "True_Power": eres["trues"],
                       "Pred_Power": eres["preds"]}).to_csv(
            station_dir / f"test_predictions_extreme_{tag}.csv",
            index=False, encoding="utf-8-sig")
        # Save unmasked extreme-condition predictions
        eres0 = ext_results_0[name]
        pd.DataFrame({"Time": pd.to_datetime(eres0["times"]),
                       "True_Power": eres0["trues"],
                       "Pred_Power": eres0["preds"]}).to_csv(
            station_dir / f"test_predictions_extreme_0miss_{tag}.csv",
            index=False, encoding="utf-8-sig")
        torch.save({"model_state_dict": res["state"],
                     "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ystd},
                    station_dir / f"best_model_{tag}.pth")

    metrics = {
        tag: {k: results[name][k] for k in ["mae", "mse", "rmse", "r2"]}
        for name, tag in ALL_TAGS.items()
    }
    metrics["train"] = {
        tag: results[name]["train_metrics"]
        for name, tag in ALL_TAGS.items()
    }
    metrics["val"] = {
        tag: results[name]["val_metrics"]
        for name, tag in ALL_TAGS.items()
    }
    pinn_res, dnn_res = results["PINN"], results["DNN"]
    metrics["improvement"] = {
        "pinn_vs_dnn_mae_pct": pct(dnn_res["mae"], pinn_res["mae"]),
        "pinn_vs_lstm_mae_pct": pct(results["LSTM"]["mae"], pinn_res["mae"]),
        "pinn_vs_gru_mae_pct": pct(results["GRU"]["mae"], pinn_res["mae"]),
        "pinn_vs_tcn_mae_pct": pct(results["TCN"]["mae"], pinn_res["mae"]),
        "pinn_vs_trans_mae_pct": pct(results["Transformer"]["mae"], pinn_res["mae"]),
    }
    metrics["extreme"] = {
        tag: {k: ext_results[name][k] for k in ["mae", "mse", "rmse", "r2"]}
        for name, tag in ALL_TAGS.items()
    }
    metrics["extreme_improvement"] = {
        "pinn_vs_dnn_mae_pct": pct(ext_results["DNN"]["mae"], ep["mae"]),
        "pinn_vs_lstm_mae_pct": pct(ext_results["LSTM"]["mae"], ep["mae"]),
        "pinn_vs_gru_mae_pct": pct(ext_results["GRU"]["mae"], ep["mae"]),
        "pinn_vs_tcn_mae_pct": pct(ext_results["TCN"]["mae"], ep["mae"]),
        "pinn_vs_trans_mae_pct": pct(ext_results["Transformer"]["mae"], ep["mae"]),
    }
    metrics["miss_sweep"] = {}
    for pct_label, row in sweep_results.items():
        metrics["miss_sweep"][str(pct_label)] = {
            tag: {k: row[name][k] for k in ["mae", "mse", "rmse", "r2"]}
            for name, tag in ALL_TAGS.items()
        }
    metrics["config"] = {"features": FEATURE_COLS, "window": WINDOW_SIZE,
                         "hidden": HIDDEN_DIM, "rated_power": rated_power,
                         "extreme_miss_ratios": [int(r*100) for r in EXTREME_MISS_RATIOS]}
    with open(station_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"\n{turbine_name} 结果已保存到 {station_dir}/。")


def main():
    """Train all original wind sources without selecting, deleting, or renaming them."""
    raise RuntimeError('Historical trainer disabled. Use scripts/reproduce.py train for the reviewed protocol.')
    print(f"Device: {DEVICE}")
    for turbine_name, turbine_file in TURBINE_FILES:
        run_turbine(turbine_name, turbine_file)
    print("Finished training all six original sources. See provenance for archived display IDs.")


if __name__ == "__main__":
    main()
