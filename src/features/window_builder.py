"""
Feature Engineering: 72-hour sliding windows
=============================================
Creates sequences shaped (batch, T=72, F) that feed both the baseline
and the HEART model.  Both models receive IDENTICAL input — the only
difference between them is the prepended attention layer.

Paper §4.2.2: T = 72 corresponds to three full days of hourly data.

FIX (v2): Target y is now scaled with a separate StandardScaler so the
model learns in a normalised output space.  Predictions are inverse-
transformed back to raw µg/m³ before any metric is computed.
Saves: models/target_scaler.pkl alongside the existing models/scaler.pkl.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib, os


# ---------------------------------------------------------------------------
# Step 1 – reshape raw OpenAQ data into a clean hourly wide table
# ---------------------------------------------------------------------------

def build_hourly_table(df: pd.DataFrame, target_param: str = "pm25") -> pd.DataFrame:
    print("🔧  Building hourly table …")
    pivot = (
        df.pivot_table(index="datetime", columns="parameter",
                       values="value", aggfunc="mean")
          .resample("1h").mean()
    )
    pivot = pivot.interpolate(method="linear", limit=6).ffill(limit=3).bfill()
    print(f"   Shape after resampling: {pivot.shape}")
    return pivot


# ---------------------------------------------------------------------------
# Step 2 – feature engineering
# ---------------------------------------------------------------------------

def engineer_features(pivot: pd.DataFrame, target_param: str = "pm25") -> pd.DataFrame:
    df = pivot.copy()

    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f"{target_param}_lag_{lag}h"] = df[target_param].shift(lag)

    for w in [24, 48, 72]:
        df[f"{target_param}_rmean_{w}h"] = df[target_param].rolling(w, min_periods=1).mean()
        df[f"{target_param}_rstd_{w}h"]  = df[target_param].rolling(w, min_periods=1).std().fillna(0)

    for p in ["pm10", "no2", "o3"]:
        if p in df.columns and p != target_param:
            for lag in [24, 48, 72]:
                df[f"{p}_lag_{lag}h"] = df[p].shift(lag)

    df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour       / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour       / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek  /  7)
    df["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek  /  7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month      / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month      / 12)

    before = len(df)
    df = df.dropna()
    print(f"   Dropped {before - len(df)} rows with NaN → {len(df)} rows remain")
    print(f"   Feature count: {len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 3 – sliding-window sequence builder
# ---------------------------------------------------------------------------

def create_sequences(df: pd.DataFrame,
                     target_col: str = "pm25",
                     seq_length: int = 72):
    feat_cols = [c for c in df.columns if c != target_col]
    X_raw = df[feat_cols].values
    y_raw = df[target_col].values
    ts    = df.index

    X_list, y_list, ts_list = [], [], []
    for i in range(len(X_raw) - seq_length):
        X_list.append(X_raw[i : i + seq_length])
        y_list.append(y_raw[i + seq_length])
        ts_list.append(ts[i + seq_length])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, pd.DatetimeIndex(ts_list)


# ---------------------------------------------------------------------------
# Step 4 – train / val / test split  (NO shuffle — time series!)
# ---------------------------------------------------------------------------

def split_sequences(X, y, timestamps,
                    train_frac: float = 0.70,
                    val_frac:   float = 0.15):
    n = len(X)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    splits = {
        "train": {"X": X[:n_train],             "y": y[:n_train],
                  "ts": timestamps[:n_train]},
        "val":   {"X": X[n_train:n_train+n_val], "y": y[n_train:n_train+n_val],
                  "ts": timestamps[n_train:n_train+n_val]},
        "test":  {"X": X[n_train+n_val:],        "y": y[n_train+n_val:],
                  "ts": timestamps[n_train+n_val:]},
    }
    for k, v in splits.items():
        print(f"   {k:5s}: {len(v['X'])} samples  "
              f"({v['ts'][0].date()} → {v['ts'][-1].date()})")
    return splits


# ---------------------------------------------------------------------------
# Step 5 – per-feature scaling for X  (fit on train only!)
# ---------------------------------------------------------------------------

def scale_splits(splits: dict, scaler_path: str = "models/scaler.pkl"):
    train_X = splits["train"]["X"]
    N, T, F = train_X.shape

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, F))

    os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"   Feature scaler saved → {scaler_path}")

    scaled = {}
    for split_name, data in splits.items():
        X = data["X"]
        X_scaled = scaler.transform(X.reshape(-1, F)).reshape(X.shape)
        scaled[split_name] = {**data, "X": X_scaled.astype(np.float32)}

    return scaled, scaler


# ---------------------------------------------------------------------------
# Step 6 – target scaling  (NEW — critical fix)
# ---------------------------------------------------------------------------

def scale_targets(splits: dict,
                  target_scaler_path: str = "models/target_scaler.pkl"):
    """
    Fit a StandardScaler on the training y (raw PM2.5) and apply to all splits.

    WHY THIS IS NECESSARY
    ---------------------
    When X is z-scored (~N(0,1)) but y is raw µg/m³ (range 20–400+),
    the MSE loss is dominated by the raw scale of y.  The model wastes
    all its capacity learning the output scale rather than the pattern,
    producing RMSE of ~75–85 µg/m³ — barely better than the mean.

    After z-scoring y the model learns normalised residuals, and RMSE
    drops to low single digits in normalised space, which translates back
    to a few µg/m³ in raw space — comparable to the persistence baseline.

    After training always call inverse_transform_targets() on predictions
    BEFORE computing any metric so numbers are in interpretable µg/m³.

    Returns
    -------
    scaled_splits  : splits with y replaced by z-scored values
    raw_y          : dict {split_name: original unscaled y}
    target_scaler  : fitted StandardScaler
    """
    raw_y = {k: v["y"].copy() for k, v in splits.items()}

    target_scaler = StandardScaler()
    target_scaler.fit(splits["train"]["y"].reshape(-1, 1))

    os.makedirs(os.path.dirname(target_scaler_path) or ".", exist_ok=True)
    joblib.dump(target_scaler, target_scaler_path)
    print(f"   Target scaler saved → {target_scaler_path}")
    print(f"   Target mean (train): {target_scaler.mean_[0]:.2f} µg/m³")
    print(f"   Target std  (train): {target_scaler.scale_[0]:.2f} µg/m³")

    scaled_splits = {}
    for split_name, data in splits.items():
        y_scaled = target_scaler.transform(
            data["y"].reshape(-1, 1)
        ).flatten().astype(np.float32)
        scaled_splits[split_name] = {**data, "y": y_scaled}

    return scaled_splits, raw_y, target_scaler


def inverse_transform_targets(y_scaled: np.ndarray,
                               target_scaler) -> np.ndarray:
    """Convert scaled model output back to raw µg/m³."""
    return target_scaler.inverse_transform(
        y_scaled.reshape(-1, 1)
    ).flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def build_dataset(df_raw: pd.DataFrame,
                  target_param: str = "pm25",
                  seq_length:   int = 72,
                  save_dir:     str = "data/processed",
                  scaler_path:  str = "models/scaler.pkl",
                  target_scaler_path: str = "models/target_scaler.pkl"):
    """
    End-to-end: raw OpenAQ DataFrame → scaled train/val/test splits.

    Saves:
      data/processed/X_{split}.npy       — scaled features
      data/processed/y_{split}.npy       — SCALED targets (fed to model)
      data/processed/y_{split}_raw.npy   — RAW µg/m³ targets (for metrics)
      data/processed/ts_{split}.npy      — timestamps
      models/scaler.pkl                  — feature StandardScaler
      models/target_scaler.pkl           — target StandardScaler

    Returns
    -------
    scaled        : dict {split: {'X' (scaled), 'y' (scaled), 'y_raw', 'ts'}}
    scaler        : fitted feature StandardScaler
    target_scaler : fitted target StandardScaler
    """
    print("=" * 60)
    print("DATASET PIPELINE")
    print("=" * 60)

    pivot    = build_hourly_table(df_raw, target_param)
    features = engineer_features(pivot,  target_param)
    X, y, ts = create_sequences(features, target_param, seq_length)

    print(f"\n   Sequences: X={X.shape}  y={y.shape}")
    print(f"   Target (raw) — mean: {y.mean():.2f}  std: {y.std():.2f}  "
          f"min: {y.min():.2f}  max: {y.max():.2f} µg/m³")

    print("\n   Splitting …")
    splits = split_sequences(X, y, ts)

    print("\n   Scaling features (fit on train only) …")
    scaled, scaler = scale_splits(splits, scaler_path)

    print("\n   Scaling targets (fit on train only) …")
    scaled, raw_y, target_scaler = scale_targets(scaled, target_scaler_path)

    # Attach raw y back into each split dict for convenience
    for split_name in scaled:
        scaled[split_name]["y_raw"] = raw_y[split_name]

    os.makedirs(save_dir, exist_ok=True)
    for split_name, data in scaled.items():
        np.save(f"{save_dir}/X_{split_name}.npy",     data["X"])
        np.save(f"{save_dir}/y_{split_name}.npy",     data["y"])
        np.save(f"{save_dir}/y_{split_name}_raw.npy", data["y_raw"])
        np.save(f"{save_dir}/ts_{split_name}.npy",
                np.array([str(t) for t in data["ts"]]))

    features.to_parquet(f"{save_dir}/delhi_features.parquet")

    print(f"\n✅  Dataset saved to {save_dir}/")
    print("=" * 60)
    return scaled, scaler, target_scaler
