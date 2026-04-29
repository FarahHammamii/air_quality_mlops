"""
metrics/metrics_logger.py
==========================
Appends every training run to a local SQLite database that Grafana reads.
Also writes a CSV mirror for easy inspection without SQL.
"""

import csv
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH  = Path("metrics/runs.db")
CSV_PATH = Path("metrics/runs.csv")

# All numeric columns that end up in the DB
METRIC_COLS = [
    "baseline_rmse", "baseline_mae", "baseline_mse", "baseline_r2",
    "heart_rmse",    "heart_mae",    "heart_mse",    "heart_r2",
    "mse_improvement_pct", "mae_improvement_pct", "rmse_improvement_pct",
]

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id              TEXT    NOT NULL,
    run_id                TEXT    NOT NULL,
    run_ts                TEXT    NOT NULL,   -- ISO-8601 UTC
    data_path             TEXT,
    rolling_days          INTEGER,
    baseline_rmse         REAL,
    baseline_mae          REAL,
    baseline_mse          REAL,
    baseline_r2           REAL,
    heart_rmse            REAL,
    heart_mae             REAL,
    heart_mse             REAL,
    heart_r2              REAL,
    mse_improvement_pct   REAL,
    mae_improvement_pct   REAL,
    rmse_improvement_pct  REAL
);
"""

INSERT_SQL = """
INSERT INTO runs (
    batch_id, run_id, run_ts, data_path, rolling_days,
    baseline_rmse, baseline_mae, baseline_mse, baseline_r2,
    heart_rmse,    heart_mae,    heart_mse,    heart_r2,
    mse_improvement_pct, mae_improvement_pct, rmse_improvement_pct
) VALUES (
    :batch_id, :run_id, :run_ts, :data_path, :rolling_days,
    :baseline_rmse, :baseline_mae, :baseline_mse, :baseline_r2,
    :heart_rmse,    :heart_mae,    :heart_mse,    :heart_r2,
    :mse_improvement_pct, :mae_improvement_pct, :rmse_improvement_pct
);
"""


def _ensure_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(CREATE_SQL)


def _append_csv(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_run_metrics(
    batch_id: str,
    run_id: str,
    baseline_metrics: dict,
    heart_metrics: dict,
    improvements: dict,
    data_path: str = "",
    rolling_days: int = 90,
) -> None:
    """
    Persist one training run to SQLite + CSV.

    Parameters match what train_pipeline.py collects from compare_models().
    """
    _ensure_db()

    row = {
        "batch_id":    batch_id,
        "run_id":      run_id,
        "run_ts":      datetime.now(timezone.utc).isoformat(),
        "data_path":   data_path,
        "rolling_days": rolling_days,
        # Baseline
        "baseline_rmse": baseline_metrics.get("RMSE"),
        "baseline_mae":  baseline_metrics.get("MAE"),
        "baseline_mse":  baseline_metrics.get("MSE"),
        "baseline_r2":   baseline_metrics.get("R²"),
        # HEART
        "heart_rmse": heart_metrics.get("RMSE"),
        "heart_mae":  heart_metrics.get("MAE"),
        "heart_mse":  heart_metrics.get("MSE"),
        "heart_r2":   heart_metrics.get("R²"),
        # Improvements
        "mse_improvement_pct":  improvements.get("MSE (%)"),
        "mae_improvement_pct":  improvements.get("MAE (%)"),
        "rmse_improvement_pct": improvements.get("RMSE (%)"),
    }

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(INSERT_SQL, row)
    log.info("Metrics logged to SQLite: %s", DB_PATH)

    _append_csv(row)
    log.info("Metrics appended to CSV: %s", CSV_PATH)
