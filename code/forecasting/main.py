import copy
import json
import random
from pathlib import Path
from datetime import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import DNNModel, BaselineModel, GRUModel, TCNModel, PINNModel, TransformerModel


# =========================
# Basic configuration
# =========================
from forecast_paths import PV_DATA, PV_OUTPUT
DATA_DIR = PV_DATA
OUTPUT_DIR = PV_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INVERTER_FILES = [
    ("Inverter_1", DATA_DIR / "Library_Inverter_1.csv"),
    ("Inverter_2", DATA_DIR / "Library_Inverter_2.csv"),
    ("Inverter_3", DATA_DIR / "Library_Inverter_3.csv"),
]
IRR_FILE = DATA_DIR / "Irradiance_2023.csv"
TEMP_FILE = DATA_DIR / "Temperature_2023.csv"
WIND_FILE = DATA_DIR / "Wind_2023.csv"

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 24          # Two hours of context at five-minute intervals
HORIZON = 1
BATCH_SIZE = 256
EPOCHS = 150
LR = 5e-4
PATIENCE = 30
SCHED_PATIENCE = 10       # ReduceLROnPlateau patience
SCHED_FACTOR = 0.5        # Learning-rate reduction factor

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4       # L2 regularization

# Physics-constraint weights
LAMBDA_NONNEG = 0.2
LAMBDA_UPPER = 0.1
LAMBDA_LOW_IRR = 0.2

LOW_IRR_THRESHOLD = 50.0  # W/m²
RATED_POWER = None

# PINN-specific hyperparameters
PINN_HIDDEN_DIM = 128     # Larger LSTM correction branch
PINN_NUM_LAYERS = 3        # Deeper LSTM
PINN_DROPOUT = 0.1         # Lower dropout rate
PINN_EPOCHS = 250          # Additional training epochs
PINN_PATIENCE = 100        # Longer early-stopping patience
PINN_LR_BACKBONE = 3e-4   # Lower backbone learning rate for the larger model
PINN_LR_PHYSICS = 5e-5    # Lower learning rate for physical parameters

# Extreme-condition evaluation settings
EXTREME_IRR_HIGH = 0.90     # Irradiance above the 90th percentile
EXTREME_TEMP_LOW = 0.10     # Temperature below the 10th percentile
EXTREME_TEMP_HIGH = 0.90    # Temperature above the 90th percentile
EXTREME_MISS_RATIOS = [r / 100 for r in range(0, 55, 5)]  # Missing fractions from 0% to 50%, in steps of 5%


# =========================
# Utility functions
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_auto(file_path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"无法读取 {file_path}")


def parse_time_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.to_datetime(s, errors="coerce")


def safe_to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def find_existing_column(df_cols, aliases):
    for a in aliases:
        if a in df_cols:
            return a
    return None


# =========================
# Data loading
# =========================
def load_inverter_data(file_path):
    df = read_csv_auto(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    time_col = df.columns[0]
    df[time_col] = parse_time_series(df[time_col])
    df = df[df[time_col].dt.year == 2023].copy()
    df = df[(df[time_col].dt.time >= time(6, 0)) &
            (df[time_col].dt.time <= time(17, 55))].copy()

    col_alias = {
        "dcVoltage": ["dcVoltage(V)", "dcVoltage"],
        "totalActivePower": ["totalActivePower(W)", "totalActivePower"],
        "L1_acCurrent": ["L1_acCurrent(A)", "L1_acCurrent"],
        "L1_acVoltage": ["L1_acVoltage(V)", "L1_acVoltage"],
        "L2_acCurrent": ["L2_acCurrent(A)", "L2_acCurrent"],
        "L2_acVoltage": ["L2_acVoltage(V)", "L2_acVoltage"],
        "L3_acCurrent": ["L3_acCurrent(A)", "L3_acCurrent"],
        "L3_acVoltage": ["L3_acVoltage(V)", "L3_acVoltage"],
        "L1_acFrequency": ["L1_acFrequency(Hz)", "L1_acFrequency"],
        "L2_acFrequency": ["L2_acFrequency(Hz)", "L2_acFrequency"],
        "L3_acFrequency": ["L3_acFrequency(Hz)", "L3_acFrequency"],
    }
    rename_map = {time_col: "Time"}
    for std_name, aliases in col_alias.items():
        rc = find_existing_column(df.columns, aliases)
        if rc:
            rename_map[rc] = std_name
    df = df.rename(columns=rename_map)

    for c in ["Time", "totalActivePower"]:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")

    keep = ["Time"] + [c for c in col_alias if c in df.columns]
    df = df[keep].copy()
    safe_to_numeric(df, [c for c in df.columns if c != "Time"])
    df = df.dropna(subset=["Time", "totalActivePower"]).sort_values("Time").reset_index(drop=True)
    df = df.groupby("Time", as_index=False).mean(numeric_only=True)
    return df


def load_weather_data(file_path, value_name):
    df = read_csv_auto(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    tcol, vcol = df.columns[0], df.columns[1]
    df[tcol] = parse_time_series(df[tcol])
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
    df = df[df[tcol].dt.year == 2023].copy()
    df = df[[tcol, vcol]].rename(columns={tcol: "Time", vcol: value_name})
    df = df.dropna(subset=["Time", value_name]).sort_values("Time").reset_index(drop=True)
    df = df.set_index("Time").resample("5min").mean().reset_index()
    df = df.groupby("Time", as_index=False).mean(numeric_only=True)
    return df


def merge_all_data(inverter_file):
    inv = load_inverter_data(inverter_file)
    irr = load_weather_data(IRR_FILE, "Irradiance")
    temp = load_weather_data(TEMP_FILE, "Temperature")
    wind = load_weather_data(WIND_FILE, "WindSpeed")
    m = inv.sort_values("Time").copy()
    for w in [irr, temp, wind]:
        m = pd.merge_asof(m.sort_values("Time"), w.sort_values("Time"),
                          on="Time", direction="nearest",
                          tolerance=pd.Timedelta("2min30s"))
    return m.dropna().sort_values("Time").reset_index(drop=True)


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

    # Irradiance first difference; initialize the first row to zero
    df["Irr_diff"] = df["Irradiance"].diff().fillna(0.0)

    # Approximate clear-sky index: observed irradiance / nominal maximum
    # Approximate maximum irradiance using 1000 times the hour-sine term
    solar_elev = np.sin(np.pi * (hour - 6.0) / 12.0).clip(0.01, 1.0)
    clear_sky_irr = 1000.0 * solar_elev
    df["ClearSkyIndex"] = (df["Irradiance"] / clear_sky_irr).clip(0.0, 1.5)

    return df


FEATURE_COLS = ["Irradiance", "Temperature", "WindSpeed",
                "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


# =========================
# Sequence construction
# =========================
def build_sequences(df, feature_cols, target_col="totalActivePower",
                    window_size=24, horizon=1):
    feat = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    irr = df["Irradiance"].values.astype(np.float32)
    temp = df["Temperature"].values.astype(np.float32)
    tv = df["Time"].values

    X, y, ir, tp, ts = [], [], [], [], []
    for i in range(window_size, len(df) - horizon + 1):
        ti = i + horizon - 1
        X.append(feat[i - window_size: i])
        y.append(target[ti])
        ir.append(irr[ti])
        tp.append(temp[ti])
        ts.append(tv[ti])

    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),
            np.array(ir, dtype=np.float32), np.array(tp, dtype=np.float32),
            np.array(ts))


def split_data(X, y, irr, temp, times):
    """Split samples in chronological order."""
    n = len(X)
    i1, i2 = int(n * TRAIN_RATIO), int(n * (TRAIN_RATIO + VAL_RATIO))
    return {
        "train": (X[:i1], y[:i1], irr[:i1], temp[:i1], times[:i1]),
        "val":   (X[i1:i2], y[i1:i2], irr[i1:i2], temp[i1:i2], times[i1:i2]),
        "test":  (X[i2:], y[i2:], irr[i2:], temp[i2:], times[i2:]),
    }


def split_data_stratified(X, y, irr, temp, times):
    """Historical month-wise split: first 70% for training, next 15% for validation, and final 15% for testing within each month."""
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
        "train": (X[train_idx], y[train_idx], irr[train_idx], temp[train_idx], times[train_idx]),
        "val":   (X[val_idx],   y[val_idx],   irr[val_idx],   temp[val_idx],   times[val_idx]),
        "test":  (X[test_idx],  y[test_idx],  irr[test_idx],  temp[test_idx],  times[test_idx]),
    }


def fit_scalers(X_train, y_train):
    xm = X_train.reshape(-1, X_train.shape[-1]).mean(0)
    xs = X_train.reshape(-1, X_train.shape[-1]).std(0)
    xs[xs < 1e-8] = 1.0
    # Scale targets to [0, 1] using the training minimum and range
    y_min = float(np.min(y_train))
    y_max = float(np.max(y_train))
    if y_max - y_min < 1e-8:
        y_max = y_min + 1.0
    return xm, xs, y_min, y_max


# =========================
# Dataset
# =========================
class PVDataset(Dataset):
    def __init__(self, X_s, y_s, y_raw, irr, temp, times):
        self.X = torch.tensor(X_s, dtype=torch.float32)
        self.y_s = torch.tensor(y_s, dtype=torch.float32)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32)
        self.irr = torch.tensor(irr, dtype=torch.float32)
        self.temp = torch.tensor(temp, dtype=torch.float32)
        self.times = pd.to_datetime(times).strftime("%Y-%m-%d %H:%M:%S").tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx], "y_s": self.y_s[idx], "y_raw": self.y_raw[idx],
                "irr": self.irr[idx], "temp": self.temp[idx], "time": self.times[idx]}


# =========================
# Loss functions
# =========================
def postprocess_predictions(preds, irr_raw, rated_power):
    """Apply the historical low-irradiance zeroing rule and clip to physical bounds."""
    preds = np.clip(preds, 0.0, rated_power)
    preds[irr_raw < LOW_IRR_THRESHOLD] = 0.0
    return preds


def baseline_loss(pred_s, y_s, **_kw):
    loss = F.mse_loss(pred_s, y_s)
    return loss, {"total": loss.item(), "data": loss.item()}


def pinn_loss(pred_s, y_s, irr_raw, y_mean, y_std, rated_power, **_kw):
    data_loss = F.mse_loss(pred_s, y_s)

    pred_raw = pred_s * y_std + y_mean
    rp2 = rated_power ** 2

    # Non-negativity penalty
    L_nn = torch.mean(torch.relu(-pred_raw) ** 2) / rp2

    # Upper-bound penalty
    L_ub = torch.mean(torch.relu(pred_raw - rated_power) ** 2) / rp2

    # Low-irradiance penalty
    mask = (irr_raw < LOW_IRR_THRESHOLD).float()
    n_low = torch.sum(mask) + 1e-8
    L_li = torch.sum(pred_raw ** 2 * mask) / (n_low * rp2)

    total = data_loss + LAMBDA_NONNEG * L_nn + LAMBDA_UPPER * L_ub + LAMBDA_LOW_IRR * L_li
    return total, {"total": total.item(), "data": data_loss.item(),
                    "nn": L_nn.item(), "ub": L_ub.item(), "li": L_li.item()}


# =========================
# Training and evaluation
# =========================
def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       y_mean, y_std, rated_power,
                       loss_fn, is_pinn, label, device,
                       optimizer=None, patience=None, epochs=None):

    patience = patience or PATIENCE
    epochs = epochs or EPOCHS
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE,
        min_lr=1e-6)

    ym = torch.tensor(y_mean, dtype=torch.float32, device=device)
    ys = torch.tensor(y_std, dtype=torch.float32, device=device)
    rp = torch.tensor(rated_power, dtype=torch.float32, device=device)

    best_val = float("inf")
    wait = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        t_loss, t_data, t_cnt = 0.0, 0.0, 0
        t_pred, t_true = [], []

        for b in train_loader:
            x = b["x"].to(device)
            y_s = b["y_s"].to(device)
            y_raw = b["y_raw"].to(device)
            irr = b["irr"].to(device)
            tmp = b["temp"].to(device)

            optimizer.zero_grad()
            if is_pinn:
                pred = model(x, G=irr, T_amb=tmp)
            else:
                pred = model(x)

            loss, items = loss_fn(pred, y_s, irr_raw=irr,
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
                irr = b["irr"].to(device)
                tmp = b["temp"].to(device)

                if is_pinn:
                    pred = model(x, G=irr, T_amb=tmp)
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
        scheduler.step(avg_vd)

        history.append({"epoch": epoch, "train_total": t_loss / t_cnt,
                         "train_data": t_data / t_cnt, "train_mae": t_mae,
                         "train_mse": t_mse,
                         "val_data": avg_vd, "val_mae": v_mae,
                         "val_mse": v_mse})

        if epoch == 1 or epoch % 5 == 0:
            print(f"  [{label}] {epoch:03d}/{epochs} | "
                  f"val_loss={avg_vd:.6f} | t_MAE={t_mae:.0f} | v_MAE={v_mae:.0f}")

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
    te_p, te_t, te_ts, te_irr = [], [], [], []
    with torch.no_grad():
        for b in test_loader:
            x = b["x"].to(device)
            irr = b["irr"].to(device)
            tmp = b["temp"].to(device)
            if is_pinn:
                pred = model(x, G=irr, T_amb=tmp)
            else:
                pred = model(x)
            te_p.append((pred * ys + ym).cpu().numpy())
            te_t.append(b["y_raw"].numpy())
            te_irr.append(b["irr"].numpy())
            te_ts.extend(b["time"])

    te_p = np.concatenate(te_p); te_t = np.concatenate(te_t)
    te_irr = np.concatenate(te_irr)
    te_p = postprocess_predictions(te_p, te_irr, rated_power)
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
def select_extreme_samples(Xte_s, yte_s, yte, irr_te, tmp_te, tte,
                           irr_all, temp_all):
    """Select extreme irradiance or temperature samples without injecting missing inputs."""
    irr_q_high = np.percentile(irr_all, EXTREME_IRR_HIGH * 100)
    temp_q_low = np.percentile(temp_all, EXTREME_TEMP_LOW * 100)
    temp_q_high = np.percentile(temp_all, EXTREME_TEMP_HIGH * 100)

    mask = ((irr_te > irr_q_high) |
            (tmp_te < temp_q_low) |
            (tmp_te > temp_q_high))
    idx = np.where(mask)[0]

    if len(idx) < 50:
        irr_dev = np.abs(irr_te - np.median(irr_te))
        tmp_dev = np.abs(tmp_te - np.median(tmp_te))
        combined = irr_dev / (irr_dev.max() + 1e-8) + tmp_dev / (tmp_dev.max() + 1e-8)
        k = max(50, int(len(irr_te) * 0.2))
        idx = np.argsort(combined)[-k:]

    print(f"  极端场景样本数: {len(idx)} / {len(irr_te)} "
          f"(irr>{irr_q_high:.0f}W/m2 or temp<{temp_q_low:.1f}C / >{temp_q_high:.1f}C)")
    return idx


def inject_missing(X, irr, tmp, miss_ratio, seed=42):
    """Historical masking protocol: mask the temporal input but retain target-time weather for the physics branch. This supplies unequal information to the compared branches; do not interpret it as an equal-input robustness benchmark."""
    X_out = X.copy()
    if miss_ratio > 0:
        rng = np.random.RandomState(seed)
        miss_mask = rng.random(X_out.shape) < miss_ratio
        X_out[miss_mask] = 0.0
    return X_out, irr, tmp


def evaluate_on_loader(model, loader, y_mean, y_std, is_pinn, device,
                       rated_power=None):
    """Evaluate a trained model on a loader and return metrics and predictions."""
    ym = torch.tensor(y_mean, dtype=torch.float32, device=device)
    ys = torch.tensor(y_std, dtype=torch.float32, device=device)

    model.eval()
    te_p, te_t, te_ts, te_irr = [], [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device)
            irr = b["irr"].to(device)
            tmp = b["temp"].to(device)
            if is_pinn:
                pred = model(x, G=irr, T_amb=tmp)
            else:
                pred = model(x)
            te_p.append((pred * ys + ym).cpu().numpy())
            te_t.append(b["y_raw"].numpy())
            te_irr.append(b["irr"].numpy())
            te_ts.extend(b["time"])

    te_p = np.concatenate(te_p)
    te_t = np.concatenate(te_t)
    te_irr = np.concatenate(te_irr)
    if rated_power is not None:
        te_p = postprocess_predictions(te_p, te_irr, rated_power)
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
def run_station(station_name, inverter_file):
    """Run the historical training and evaluation pipeline for one inverter."""
    station_dir = OUTPUT_DIR / station_name
    station_dir.mkdir(exist_ok=True)
    set_seed(RANDOM_SEED)
    print(f"\n{'#' * 76}")
    print(f"# 站点: {station_name} ({inverter_file.name})")
    print(f"{'#' * 76}")

    # Data
    print("加载数据...")
    df = merge_all_data(inverter_file)
    df = add_time_features(df)
    df.to_csv(station_dir / "merged_2023_data.csv", index=False, encoding="utf-8-sig")
    print(f"样本: {len(df)},  特征: {FEATURE_COLS}")

    X, y, irr, temp, times = build_sequences(df, FEATURE_COLS, window_size=WINDOW_SIZE)
    # Historical month-wise chronological train/validation/test splits
    print("  -> 使用按月分层划分（避免季节分布偏移）")
    sp = split_data_stratified(X, y, irr, temp, times)
    Xtr, ytr, irr_tr, tmp_tr, ttr = sp["train"]
    Xva, yva, irr_va, tmp_va, tva = sp["val"]
    Xte, yte, irr_te, tmp_te, tte = sp["test"]
    print(f"序列: {len(X)}  (train {len(Xtr)} / val {len(Xva)} / test {len(Xte)})")

    xm, xs, ymin, ymax = fit_scalers(Xtr, ytr)
    Xtr_s = (Xtr - xm) / xs; Xva_s = (Xva - xm) / xs; Xte_s = (Xte - xm) / xs
    ytr_s = (ytr - ymin) / (ymax - ymin)
    yva_s = (yva - ymin) / (ymax - ymin)
    yte_s = (yte - ymin) / (ymax - ymin)
    ym = ymin; ystd = ymax - ymin
    np.savez(station_dir / "scalers.npz", x_mean=xm, x_std=xs, y_mean=ym, y_std=ystd)

    rated_power = float(RATED_POWER or np.max(ytr) * 1.05)
    print(f"额定功率 = {rated_power:.0f} W")

    trn = DataLoader(PVDataset(Xtr_s, ytr_s, ytr, irr_tr, tmp_tr, ttr),
                     batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val = DataLoader(PVDataset(Xva_s, yva_s, yva, irr_va, tmp_va, tva),
                     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    tst = DataLoader(PVDataset(Xte_s, yte_s, yte, irr_te, tmp_te, tte),
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

    # Experiment 6: PINN
    print("\n" + "=" * 60)
    print("实验 6: PINN（物理分支 + LSTM 残差修正）")
    print("=" * 60)
    set_seed(RANDOM_SEED)
    pinn_model = PINNModel(ndim, PINN_HIDDEN_DIM, PINN_NUM_LAYERS, PINN_DROPOUT).to(DEVICE)
    pinn_model.set_output_scale(ym, ystd)
    # Use separate optimizer groups for physical and backbone parameters
    phys_names = {"log_alpha", "beta", "T_ref", "log_noct", "gate_param"}
    phys_params = [p for n, p in pinn_model.named_parameters() if n in phys_names]
    backbone_params = [p for n, p in pinn_model.named_parameters() if n not in phys_names]
    pinn_optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": PINN_LR_BACKBONE},
        {"params": phys_params, "lr": PINN_LR_PHYSICS},
    ], weight_decay=WEIGHT_DECAY)
    pn = train_and_evaluate(pinn_model, trn, val, tst,
                            ym, ystd, rated_power,
                            pinn_loss, True, "PINN", DEVICE,
                            optimizer=pinn_optim, patience=PINN_PATIENCE,
                            epochs=PINN_EPOCHS)

    # Model comparison
    def pct(a, b): return (b - a) / abs(a) * 100 if abs(a) > 1e-8 else 0

    results = {"DNN": dnn, "LSTM": bl, "GRU": gru, "TCN": tcn, "Transformer": tr, "PINN": pn}

    print("\n" + "=" * 88)
    print("       DNN  vs  LSTM  vs  GRU  vs  TCN  vs  Transformer  vs  PINN 对比结果")
    print("=" * 88)
    print(f"{'指标':<10}{'DNN':>12}{'LSTM':>12}{'GRU':>12}{'TCN':>12}{'Trans':>12}{'PINN':>12}")
    print("-" * 82)
    for nm, k in [("MAE(W)", "mae"), ("RMSE(W)", "rmse"),
                  ("MSE", "mse"), ("R2", "r2")]:
        vals = [results[n][k] for n in ["DNN", "LSTM", "GRU", "TCN", "Transformer", "PINN"]]
        print(f"{nm:<10}" + "".join(f"{v:>12.2f}" for v in vals))
    print("-" * 82)
    for ref_name, tgt_name in [("DNN", "LSTM"), ("DNN", "GRU"), ("DNN", "TCN"),
                                ("DNN", "Transformer"), ("LSTM", "PINN"), ("DNN", "PINN")]:
        ref, tgt = results[ref_name], results[tgt_name]
        print(f"{'%s vs %s' % (tgt_name, ref_name):>20} | MAE: {pct(ref['mae'], tgt['mae']):+.2f}%  R2: {pct(ref['r2'], tgt['r2']):+.2f}%")
    print("=" * 76)

    # Evaluate selected checkpoints on training and validation subsets for plotting
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

    print(f"\n{'集合':<8}{'指标':<10}{'DNN':>10}{'LSTM':>10}{'GRU':>10}{'TCN':>10}{'Trans':>10}{'PINN':>10}")
    print("-" * 78)
    for split_label, split_key in [("Train", "train_metrics"), ("Val", "val_metrics"),
                                    ("Test", None)]:
        for nm, k in [("MAE(W)", "mae"), ("RMSE(W)", "rmse"), ("R²", "r2")]:
            if split_key:
                vals = [results[n][split_key][k] for n in ["DNN", "LSTM", "GRU", "TCN", "Transformer", "PINN"]]
            else:
                vals = [results[n][k] for n in ["DNN", "LSTM", "GRU", "TCN", "Transformer", "PINN"]]
            print(f"{split_label:<8}{nm:<10}" + "".join(f"{v:>10.2f}" for v in vals))
    print("-" * 78)

    # Evaluate extreme conditions across missing-input fractions
    print("\n" + "=" * 60)
    print("极端场景评估（极端天气 + 传感器缺失扫描 0%~50%）")
    print("=" * 60)

    ext_idx = select_extreme_samples(
        Xte_s, yte_s, yte, irr_te, tmp_te, tte,
        irr_all=irr_tr, temp_all=tmp_tr)

    # Extract extreme-condition samples before masking
    X_ext_raw = Xte_s[ext_idx]
    y_ext_s = yte_s[ext_idx]
    y_ext = yte[ext_idx]
    irr_ext = irr_te[ext_idx]
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

    # sweep_results[miss_pct] = {"DNN": {...}, "LSTM": {...}, ...}
    sweep_results = {}
    for mr in EXTREME_MISS_RATIOS:
        pct_label = int(round(mr * 100))
        X_miss, irr_miss, tmp_miss = inject_missing(X_ext_raw, irr_ext, tmp_ext, mr, seed=RANDOM_SEED)
        loader = DataLoader(
            PVDataset(X_miss, y_ext_s, y_ext, irr_miss, tmp_miss, t_ext),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        row = {}
        for name, mdl, state, is_p in models_map:
            mdl.load_state_dict(state)
            row[name] = evaluate_on_loader(mdl, loader, ym, ystd, is_p, DEVICE,
                                             rated_power=rated_power)
        sweep_results[pct_label] = row
        line = f"  miss={pct_label:2d}%"
        for name, _, _, _ in models_map:
            line += f"  {name} MAE={row[name]['mae']:.0f}"
        print(line)

    # Report the 30% masking case in the detailed table
    ext_results = sweep_results[30]
    print(f"\n--- 极端场景 (miss=30%) 详细指标 ---")
    print(f"{'指标':<10}{'DNN':>12}{'LSTM':>12}{'GRU':>12}{'TCN':>12}{'Trans':>12}{'PINN':>12}")
    print("-" * 82)
    for nm, k in [("MAE(W)", "mae"), ("RMSE(W)", "rmse"),
                  ("MSE", "mse"), ("R2", "r2")]:
        vals = [ext_results[n][k] for n in ["DNN", "LSTM", "GRU", "TCN", "Transformer", "PINN"]]
        print(f"{nm:<10}" + "".join(f"{v:>12.2f}" for v in vals))
    print("-" * 82)
    ep = ext_results["PINN"]
    for ref in ["DNN", "LSTM", "GRU", "TCN", "Transformer"]:
        er = ext_results[ref]
        print(f"{'PINN vs %s' % ref:>20} | MAE: {pct(er['mae'], ep['mae']):+.2f}%  R2: {pct(er['r2'], ep['r2']):+.2f}%")
    print("=" * 76)

    # Save outputs
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
    # Metrics for extreme conditions with 30% masking
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
    # Metrics across missing-input fractions
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

    print(f"\n{station_name} 结果已保存到 {station_dir}/。")


def main():
    raise RuntimeError('Historical trainer disabled. Use scripts/reproduce.py train for the reviewed protocol.')
    print(f"Device: {DEVICE}")
    for station_name, inverter_file in INVERTER_FILES:
        run_station(station_name, inverter_file)
    print("\n所有站点训练完成！运行 python plot.py 可查看可视化对比。")


if __name__ == "__main__":
    main()
