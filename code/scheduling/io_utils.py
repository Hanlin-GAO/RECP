import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scheduling_paths import FORECAST_CODE


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def find_power_like_col(df: pd.DataFrame,
                        preferred_keywords=None,
                        exclude_keywords=None) -> str:
    preferred_keywords = preferred_keywords or []
    exclude_keywords = exclude_keywords or []
    df.columns = [c.strip() for c in df.columns]
    cols = df.columns.tolist()

    def ok_col(c):
        cl = c.lower()
        for ek in exclude_keywords:
            if ek.lower() in cl:
                return False
        x = _to_numeric_series(df[c])
        return x.notna().sum() > 0

    for kw in preferred_keywords:
        for c in cols:
            if kw.lower() in c.lower() and ok_col(c):
                return c

    for c in cols:
        if ok_col(c):
            return c

    raise ValueError("No suitable power-like column found.")


def read_power_series_from_excel(excel_path: str,
                                 preferred_keywords=("Power",),
                                 exclude_keywords=("time", "日期"),
                                 max_len=None) -> np.ndarray:
    df = pd.read_excel(excel_path)
    col = find_power_like_col(df, preferred_keywords=list(preferred_keywords), exclude_keywords=list(exclude_keywords))
    s = _to_numeric_series(df[col]).dropna().values.astype(float)
    if max_len is not None:
        s = s[:max_len]
    return s


def _ensure_import_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_model_runtime(domain: str):
    if domain == "pv":
        from main import (
            FEATURE_COLS,
            WINDOW_SIZE,
            build_sequences,
            postprocess_predictions,
        )
        from model import BaselineModel, DNNModel, GRUModel, PINNModel, TCNModel, TransformerModel
        classes = {
            "BaselineModel": BaselineModel,
            "DNNModel": DNNModel,
            "GRUModel": GRUModel,
            "PINNModel": PINNModel,
            "TCNModel": TCNModel,
            "TransformerModel": TransformerModel,
        }
        return FEATURE_COLS, WINDOW_SIZE, build_sequences, postprocess_predictions, classes

    from wind_main import (
        FEATURE_COLS,
        WINDOW_SIZE,
        build_sequences,
        postprocess_predictions,
    )
    from wind_model import BaselineModel, DNNModel, GRUModel, TCNModel, TransformerModel, WindPINNModel
    classes = {
        "BaselineModel": BaselineModel,
        "DNNModel": DNNModel,
        "GRUModel": GRUModel,
        "TCNModel": TCNModel,
        "TransformerModel": TransformerModel,
        "WindPINNModel": WindPINNModel,
    }
    return FEATURE_COLS, WINDOW_SIZE, build_sequences, postprocess_predictions, classes


def _count_layers(state_dict: dict, prefix: str, suffix: str) -> int:
    count = 0
    while f"{prefix}{count}{suffix}" in state_dict:
        count += 1
    return max(count, 1)


def _infer_window_size(model_tag: str, state_dict: dict, input_dim: int, fallback_window_size: int) -> int:
    if model_tag != "dnn":
        return int(fallback_window_size)
    flat_dim = int(state_dict["net.0.weight"].shape[1])
    return max(flat_dim // max(int(input_dim), 1), 1)


def _build_model_from_state(domain: str,
                            model_tag: str,
                            state_dict: dict,
                            input_dim: int,
                            window_size: int,
                            rated_power: float,
                            classes: dict):
    dropout = 0.0
    if model_tag == "dnn":
        hidden_dim = int(state_dict["net.3.weight"].shape[0])
        model = classes["DNNModel"](input_dim, hidden_dim, window_size, dropout)
        return model, False

    if model_tag == "lstm":
        hidden_dim = int(state_dict["backbone.lstm.weight_hh_l0"].shape[1])
        num_layers = _count_layers(state_dict, "backbone.lstm.weight_ih_l", "")
        model = classes["BaselineModel"](input_dim, hidden_dim, num_layers, dropout)
        return model, False

    if model_tag == "gru":
        gru_prefix = "gru" if domain == "pv" else "gru"
        hidden_dim = int(state_dict[f"{gru_prefix}.weight_hh_l0"].shape[1])
        num_layers = _count_layers(state_dict, f"{gru_prefix}.weight_ih_l", "")
        model = classes["GRUModel"](input_dim, hidden_dim, num_layers, dropout)
        return model, False

    if model_tag == "tcn":
        hidden_dim = int(state_dict["skip_proj.weight"].shape[0])
        num_layers = len({key.split(".")[1] for key in state_dict if key.startswith("tcn.") and key.endswith("conv1.weight")})
        kernel_size = int(state_dict["tcn.0.conv1.weight"].shape[2])
        model = classes["TCNModel"](input_dim, hidden_dim, num_layers, dropout, kernel_size=kernel_size)
        return model, False

    if model_tag == "trans":
        hidden_dim = int(state_dict["input_proj.weight"].shape[0])
        num_layers = len({key.split(".")[2] for key in state_dict if key.startswith("transformer.layers.")})
        dim_feedforward = int(state_dict["transformer.layers.0.linear1.weight"].shape[0])
        nhead = 4 if hidden_dim % 4 == 0 else 1
        model = classes["TransformerModel"](input_dim, hidden_dim, num_layers, dropout, nhead=nhead, dim_feedforward=dim_feedforward)
        return model, False

    if domain == "pv":
        hidden_dim = int(state_dict["backbone.lstm.weight_hh_l0"].shape[1])
        num_layers = _count_layers(state_dict, "backbone.lstm.weight_ih_l", "")
        model = classes["PINNModel"](input_dim, hidden_dim, num_layers, dropout)
        return model, True

    hidden_dim = int(state_dict["backbone.gru.weight_hh_l0"].shape[1])
    num_layers = _count_layers(state_dict, "backbone.gru.weight_ih_l", "")
    model = classes["WindPINNModel"](input_dim, hidden_dim, num_layers, dropout, rated_power_init=max(float(rated_power), 1.0))
    return model, True


@lru_cache(maxsize=6)
def _infer_full_prediction_frame(vstry_dir: Path, domain: str, source_name: str, model_tag: str) -> pd.DataFrame:
    import torch

    _ensure_import_path(FORECAST_CODE)

    if domain == "pv":
        station_dir = vstry_dir / "models" / domain / source_name
        merged_path = vstry_dir / "data/forecasting/processed" / domain / source_name / "merged_2023_data.csv"
    else:
        station_dir = vstry_dir / "models" / domain / source_name
        merged_path = vstry_dir / "data/forecasting/processed" / domain / source_name / "merged_wind_data.csv"

    ckpt_path = station_dir / f"best_model_{model_tag}.pth"
    metrics_path = vstry_dir / "results_reference/forecasting" / domain / source_name / "test_metrics.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing merged data file: {merged_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    feature_cols, fallback_window_size, build_sequences, postprocess_predictions, classes = _load_model_runtime(domain)

    df = pd.read_csv(merged_path)
    df["Time"] = pd.to_datetime(df["Time"])
    window_size = _infer_window_size(model_tag, state_dict, input_dim=len(feature_cols), fallback_window_size=fallback_window_size)
    X, _, driver_raw, temp_raw, times = build_sequences(df, feature_cols, window_size=window_size)

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])
    x_std = np.where(np.abs(x_std) < 1e-8, 1.0, x_std)
    X_scaled = (X - x_mean) / x_std

    rated_power = None
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        rated_power = metrics.get("config", {}).get("rated_power")
    if rated_power is None:
        target_col = "totalActivePower" if domain == "pv" else "Power"
        rated_power = float(np.max(df[target_col].to_numpy(dtype=float)) * 1.05)

    model, is_pinn = _build_model_from_state(
        domain=domain,
        model_tag=model_tag,
        state_dict=state_dict,
        input_dim=len(feature_cols),
        window_size=window_size,
        rated_power=float(rated_power),
        classes=classes,
    )
    if is_pinn and hasattr(model, "set_output_scale"):
        model.set_output_scale(y_mean, y_std)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(X_scaled), batch_size):
            stop = min(start + batch_size, len(X_scaled))
            x = torch.tensor(X_scaled[start:stop], dtype=torch.float32)
            driver = torch.tensor(driver_raw[start:stop], dtype=torch.float32)
            temp = torch.tensor(temp_raw[start:stop], dtype=torch.float32)
            if is_pinn:
                if domain == "pv":
                    pred_scaled = model(x, G=driver, T_amb=temp)
                else:
                    pred_scaled = model(x, v=driver, T_amb=temp)
            else:
                pred_scaled = model(x)
            preds.append((pred_scaled.numpy() * y_std) + y_mean)

    pred_power = np.concatenate(preds, axis=0)
    pred_power = postprocess_predictions(pred_power, driver_raw, float(rated_power))
    return pd.DataFrame({
        "Time": pd.to_datetime(times),
        "Pred_Power": pred_power.astype(float),
    })


@lru_cache(maxsize=6)
def _load_full_actual_power_frame(vstry_dir: Path, domain: str, source_name: str) -> pd.DataFrame:
    if domain == "pv":
        station_dir = vstry_dir / "data/forecasting/processed" / domain / source_name
        merged_path = station_dir / "merged_2023_data.csv"
        power_col = "totalActivePower"
    else:
        station_dir = vstry_dir / "data/forecasting/processed" / domain / source_name
        merged_path = station_dir / "merged_wind_data.csv"
        power_col = "Power"

    if not merged_path.exists():
        raise FileNotFoundError(f"Missing merged data file: {merged_path}")

    df = pd.read_csv(merged_path)
    df["Time"] = pd.to_datetime(df["Time"])
    if power_col not in df.columns:
        raise KeyError(f"Missing power column {power_col} in {merged_path}")
    return pd.DataFrame({
        "Time": df["Time"],
        "Actual_Power": pd.to_numeric(df[power_col], errors="coerce").fillna(0.0),
    }).dropna(subset=["Time"]).reset_index(drop=True)


def load_vstry_power_frame(vstry_dir: str,
                           domain: str,
                           source_name: str,
                           kind: str,
                           model_tag: str = "pinn") -> pd.DataFrame:
    vstry_path = Path(vstry_dir)
    if kind == "predicted":
        frame = _infer_full_prediction_frame(vstry_path, domain=domain, source_name=source_name, model_tag=model_tag)
        return frame.rename(columns={"Pred_Power": "Power"})
    if kind == "actual":
        frame = _load_full_actual_power_frame(vstry_path, domain=domain, source_name=source_name)
        return frame.rename(columns={"Actual_Power": "Power"})
    raise ValueError(f"Unsupported power frame kind: {kind}")


def resample_power_frame(frame: pd.DataFrame,
                         start_time: str,
                         n_steps: int,
                         dt_hours: float) -> np.ndarray:
    start_ts = pd.Timestamp(start_time)
    clipped = frame[frame["Time"] >= start_ts].copy()
    if clipped.empty:
        raise ValueError(f"No power samples found at or after {start_ts}")

    elapsed_s = (clipped["Time"] - start_ts).dt.total_seconds().to_numpy(dtype=float)
    power = clipped["Power"].to_numpy(dtype=float)
    target_s = np.arange(int(n_steps), dtype=float) * float(dt_hours) * 3600.0
    sampled = np.interp(target_s, elapsed_s, power, left=power[0], right=power[-1])
    return np.maximum(sampled, 0.0)


def load_vstry_scaled_power_profile(vstry_dir: str,
                                    domain: str,
                                    source_name: str,
                                    model_tag: str,
                                    start_time: str,
                                    n_steps: int,
                                    dt_hours: float,
                                    scale_ratio: float) -> np.ndarray:
    frame = load_vstry_power_frame(vstry_dir, domain=domain, source_name=source_name, kind="predicted", model_tag=model_tag)
    sampled = resample_power_frame(frame, start_time=start_time, n_steps=n_steps, dt_hours=dt_hours)
    return sampled * float(scale_ratio)


def _sanitize_col_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9a-zA-Z_\-]+", "", name)
    return name if name else "rover"


def write_rover_positions_to_excel(
        excel_path: str,
        t: np.ndarray,
        rover_names,
        x_mat: np.ndarray,
        y_mat: np.ndarray,
        dt_hours: float = None,
        origin_lower_left_zero: bool = True,
):
    """
    Save all rovers' positions vs time into one Excel file.

    Sheet "positions" (wide):
      step, time_hours (optional), <rover>_x, <rover>_y, ...

    Sheet "meta":
      origin_lower_left_zero, x_offset_added, y_offset_added, note
    """
    t = np.asarray(t).reshape(-1)
    x_mat = np.asarray(x_mat, dtype=float)
    y_mat = np.asarray(y_mat, dtype=float)

    if x_mat.shape != y_mat.shape:
        raise ValueError(f"x_mat shape {x_mat.shape} != y_mat shape {y_mat.shape}")
    if x_mat.shape[0] != t.shape[0]:
        raise ValueError(f"t length {t.shape[0]} != x_mat rows {x_mat.shape[0]}")

    x_off = 0.0
    y_off = 0.0
    x_out = x_mat
    y_out = y_mat

    if origin_lower_left_zero:
        x_min = float(np.nanmin(x_mat))
        y_min = float(np.nanmin(y_mat))
        x_off = -x_min
        y_off = -y_min
        x_out = x_mat + x_off
        y_out = y_mat + y_off

    data = {"step": t}
    if dt_hours is not None:
        data["time_hours"] = t.astype(float) * float(dt_hours)

    rover_names = list(rover_names)
    if len(rover_names) != x_out.shape[1]:
        raise ValueError(f"len(rover_names)={len(rover_names)} != K={x_out.shape[1]}")

    used = set()
    for k, nm in enumerate(rover_names):
        base = _sanitize_col_name(nm)
        col = base
        j = 2
        while col in used:
            col = f"{base}_{j}"
            j += 1
        used.add(col)
        data[f"{col}_x"] = x_out[:, k]
        data[f"{col}_y"] = y_out[:, k]

    df = pd.DataFrame(data)
    meta = pd.DataFrame({
        "key": ["origin_lower_left_zero", "x_offset_added", "y_offset_added", "note"],
        "value": [
            bool(origin_lower_left_zero),
            float(x_off),
            float(y_off),
            "x_export = x_raw + x_offset_added; y_export = y_raw + y_offset_added",
        ]
    })

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="positions")
            meta.to_excel(writer, index=False, sheet_name="meta")
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        base = Path(excel_path)
        df.to_csv(base.with_suffix(".positions.csv"), index=False, encoding="utf-8-sig")
        meta.to_csv(base.with_suffix(".meta.csv"), index=False, encoding="utf-8-sig")
