"""
train_pipeline.py
=================
End-to-end training run:
  1. Loads raw parquet
  2. Builds feature dataset — X and y BOTH z-scored
  3. Trains Baseline + HEART on scaled y
  4. Inverse-transforms predictions to raw µg/m³ before ALL metrics
  5. Logs to MLflow + local SQLite
  6. Promotes champion if HEART beats previous best

KEY FIX vs v1
-------------
The original pipeline trained on unscaled y (raw µg/m³, range 20–400+).
With z-scored X but raw y the MSE gradient is dominated by the output
scale → model can't learn the pattern → RMSE ~75–85 µg/m³, R² negative.

Fix: build_dataset() now returns a target_scaler.  y fed to .fit() is
z-scored.  Predictions are inverse-transformed before any metric so all
reported numbers are interpretable µg/m³.
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf

from src.features.window_builder import build_dataset, inverse_transform_targets
from src.models.baseline_model import create_baseline_model, get_callbacks as baseline_cbs
from src.models.heart_model import create_heart_model, get_callbacks as heart_cbs
from src.utils.metrics import calculate_metrics, compare_models
from metrics.metrics_logger import log_run_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR  = Path("data/processed")
MODELS_DIR     = Path("models")
CHAMPION_FILE  = MODELS_DIR / "champion_metrics.json"
PRODUCTION_DIR = MODELS_DIR / "production"
MLFLOW_URI     = "mlruns"


# ---------------------------------------------------------------------------
# Champion helpers
# ---------------------------------------------------------------------------

def load_champion_rmse() -> float:
    if CHAMPION_FILE.exists():
        data = json.loads(CHAMPION_FILE.read_text())
        rmse = data.get("heart_rmse", float("inf"))
        log.info("Current champion HEART RMSE: %.4f µg/m³", rmse)
        return rmse
    log.info("No champion on file – treating as first run.")
    return float("inf")


def promote_champion(heart_rmse: float, run_id: str) -> bool:
    champion_rmse = load_champion_rmse()
    if heart_rmse >= champion_rmse:
        log.info("No promotion: RMSE %.4f >= champion %.4f", heart_rmse, champion_rmse)
        return False

    log.info("New champion! RMSE %.4f < %.4f — promoting.", heart_rmse, champion_rmse)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in [
        ("heart_best.keras",    "heart_production.keras"),
        ("baseline_best.keras", "baseline_production.keras"),
        ("scaler.pkl",          "scaler.pkl"),
        ("target_scaler.pkl",   "target_scaler.pkl"),   # NEW: must copy target scaler too
    ]:
        src = MODELS_DIR / src_name
        if src.exists():
            shutil.copy2(src, PRODUCTION_DIR / dst_name)

    meta = {
        "heart_rmse":    heart_rmse,
        "mlflow_run_id": run_id,
        "promoted_at":   datetime.now(timezone.utc).isoformat(),
    }
    CHAMPION_FILE.write_text(json.dumps(meta, indent=2))
    log.info("Champion metadata written to %s", CHAMPION_FILE)
    return True


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_model(model, name: str, X_train, y_train, X_val, y_val,
                epochs: int, batch_size: int, ckpt_path: str):
    cb_fn = baseline_cbs if name == "baseline" else heart_cbs
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_fn(ckpt_path),
        verbose=1,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path:    str   = "data/raw/delhi_merged.parquet",
    epochs:       int   = 100,
    batch_size:   int   = 32,
    seq_length:   int   = 72,
    num_heads:    int   = 4,
    num_layers:   int   = 2,
    dropout:      float = 0.1,
    rolling_days: int   = 90,
) -> dict:

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("air_quality_delhi")

    batch_id = datetime.now(timezone.utc).strftime("batch_%Y%m%d_%H%M%S")
    log.info("Starting pipeline run: %s", batch_id)

    # 1. Raw data ─────────────────────────────────────────────────────────
    log.info("Loading %s", data_path)
    df_raw = pd.read_parquet(data_path)
    log.info("Shape: %s | %s → %s",
             df_raw.shape, df_raw["datetime"].min(), df_raw["datetime"].max())

    # 2. Feature engineering — returns (scaled, feature_scaler, target_scaler)
    log.info("Building dataset …")
    scaled, scaler, target_scaler = build_dataset(
        df_raw,
        target_param="pm25",
        seq_length=seq_length,
        save_dir=str(PROCESSED_DIR),
        scaler_path=str(MODELS_DIR / "scaler.pkl"),
        target_scaler_path=str(MODELS_DIR / "target_scaler.pkl"),
    )

    # Scaled arrays fed to model
    X_train = scaled["train"]["X"]
    y_train = scaled["train"]["y"]       # z-scored
    X_val   = scaled["val"]["X"]
    y_val   = scaled["val"]["y"]         # z-scored
    X_test  = scaled["test"]["X"]

    # Raw µg/m³ used for every metric calculation
    y_test_raw  = scaled["test"]["y_raw"]
    y_train_raw = scaled["train"]["y_raw"]

    _, SEQ_LEN, FEAT_DIM = X_train.shape

    log.info("X_train=%s  y_train=%s (z-scored) | raw mean=%.2f std=%.2f µg/m³",
             X_train.shape, y_train.shape,
             y_train_raw.mean(), y_train_raw.std())

    # Persistence baseline (consecutive-hour difference)
    persist_rmse = float(np.sqrt(np.mean((y_test_raw[1:] - y_test_raw[:-1]) ** 2)))
    log.info("Persistence RMSE: %.4f µg/m³", persist_rmse)

    # 3. MLflow run ───────────────────────────────────────────────────────
    with mlflow.start_run(run_name=batch_id) as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "batch_id":          batch_id,
            "seq_length":        SEQ_LEN,
            "feat_dim":          FEAT_DIM,
            "epochs":            epochs,
            "batch_size":        batch_size,
            "num_heads":         num_heads,
            "num_layers":        num_layers,
            "dropout":           dropout,
            "rolling_days":      rolling_days,
            "train_samples":     len(X_train),
            "val_samples":       len(X_val),
            "test_samples":      len(X_test),
            "raw_data_path":     data_path,
            "target_scaled":     True,
            "target_mean_train": float(target_scaler.mean_[0]),
            "target_std_train":  float(target_scaler.scale_[0]),
        })
        mlflow.log_metric("persistence_rmse", persist_rmse)

        # 4. Train Baseline ───────────────────────────────────────────────
        log.info("Training Baseline …")
        baseline = create_baseline_model(seq_len=SEQ_LEN, feat_dim=FEAT_DIM)
        train_model(baseline, "baseline", X_train, y_train, X_val, y_val,
                    epochs, batch_size, str(MODELS_DIR / "baseline_best.keras"))
        baseline.save(str(MODELS_DIR / "baseline_model.keras"))

        # 5. Train HEART ──────────────────────────────────────────────────
        log.info("Training HEART …")
        heart = create_heart_model(seq_len=SEQ_LEN, feat_dim=FEAT_DIM,
                                   num_heads=num_heads, num_layers=num_layers,
                                   dropout_rate=dropout)
        train_model(heart, "heart", X_train, y_train, X_val, y_val,
                    epochs, batch_size, str(MODELS_DIR / "heart_best.keras"))
        heart.save(str(MODELS_DIR / "heart_model.keras"))

        # 6. Predict → inverse-transform → raw µg/m³ ─────────────────────
        log.info("Predicting on test set …")
        base_pred  = inverse_transform_targets(
            baseline.predict(X_test, verbose=0).flatten(), target_scaler)
        heart_pred = inverse_transform_targets(
            heart.predict(X_test, verbose=0).flatten(), target_scaler)
        y_true = y_test_raw  # raw µg/m³

        # 7. Metrics ──────────────────────────────────────────────────────
        comparison   = compare_models(base_pred, heart_pred, y_true)
        m_base       = comparison["baseline"]
        m_heart      = comparison["heart"]
        improvements = comparison["improvements"]

        log.info("=" * 60)
        log.info("RESULTS (all in µg/m³, inverse-transformed)")
        log.info("  Persistence  RMSE=%.2f", persist_rmse)
        log.info("  Baseline     RMSE=%.2f  MAE=%.2f  R²=%.4f",
                 m_base["RMSE"], m_base["MAE"], m_base["R²"])
        log.info("  HEART        RMSE=%.2f  MAE=%.2f  R²=%.4f",
                 m_heart["RMSE"], m_heart["MAE"], m_heart["R²"])
        log.info("  MSE improvement: %.2f%%  (paper target ≥7.5%%)",
                 improvements["MSE (%)"])
        log.info("  Meets paper claim: %s", comparison["meets_paper_claim"])
        log.info("=" * 60)

        if m_heart["RMSE"] > persist_rmse * 3:
            log.warning(
                "HEART RMSE (%.2f) still >3× persistence (%.2f). "
                "Increase --rolling-days or check data quality.",
                m_heart["RMSE"], persist_rmse)

        # Log to MLflow
        for prefix, m in [("baseline", m_base), ("heart", m_heart)]:
            mlflow.log_metrics({
                f"{prefix}_rmse":           m["RMSE"],
                f"{prefix}_mae":            m["MAE"],
                f"{prefix}_mse":            m["MSE"],
                f"{prefix}_r2":             m["R²"],
                f"{prefix}_mape":           m["MAPE (%)"],
                f"{prefix}_bias":           m["Bias"],
                f"{prefix}_within_25pct":   m["Within 25% (%)"],
                f"{prefix}_index_of_agree": m["Index of Agreement"],
            })
        mlflow.log_metrics({
            "mse_improvement_pct":  improvements["MSE (%)"],
            "mae_improvement_pct":  improvements["MAE (%)"],
            "rmse_improvement_pct": improvements["RMSE (%)"],
            "meets_paper_claim":    float(comparison["meets_paper_claim"]),
            "persistence_rmse":     persist_rmse,
        })

        for fname in ["scaler.pkl", "target_scaler.pkl",
                      "baseline_best.keras", "heart_best.keras"]:
            p = MODELS_DIR / fname
            if p.exists():
                mlflow.log_artifact(str(p))

        # 8. Persist to SQLite/CSV ─────────────────────────────────────────
        log_run_metrics(
            batch_id=batch_id, run_id=run_id,
            baseline_metrics=m_base, heart_metrics=m_heart,
            improvements=improvements, data_path=data_path,
            rolling_days=rolling_days,
        )

        # 9. Champion promotion ────────────────────────────────────────────
        promoted = promote_champion(m_heart["RMSE"], run_id)
        mlflow.log_param("promoted_to_champion", promoted)
        log.info("Pipeline complete. Promoted: %s", promoted)

    return {
        "batch_id":            batch_id,
        "run_id":              run_id,
        "persistence_rmse":    persist_rmse,
        "baseline_rmse":       m_base["RMSE"],
        "baseline_r2":         m_base["R²"],
        "heart_rmse":          m_heart["RMSE"],
        "heart_r2":            m_heart["R²"],
        "mse_improvement_pct": improvements["MSE (%)"],
        "meets_paper_claim":   comparison["meets_paper_claim"],
        "promoted":            promoted,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Just update the default values in the CLI section (bottom of file):

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air quality MLOps training pipeline")
    parser.add_argument("--data",         default="data/raw/delhi_merged.parquet")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--seq-length",   type=int,   default=72)
    parser.add_argument("--num-heads",    type=int,   default=2)      # CHANGED: 4 → 2
    parser.add_argument("--num-layers",   type=int,   default=1)      # CHANGED: 2 → 1
    parser.add_argument("--dropout",      type=float, default=0.3)    # CHANGED: 0.1 → 0.3
    parser.add_argument("--rolling-days", type=int,   default=90)
    args = parser.parse_args()

    result = run_pipeline(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rolling_days=args.rolling_days,
    )
    log.info("Final result: %s", result)