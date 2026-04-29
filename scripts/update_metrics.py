"""
Update Metrics Database with latest run results
"""

import pandas as pd
import sqlite3
import json
from datetime import datetime
from pathlib import Path


def update_metrics_database():
    """Update SQLite database with latest metrics"""
    
    # Create metrics directory
    Path("metrics").mkdir(exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect("metrics/runs.db")
    
    # Load champion metrics
    champion_file = Path("models/champion_metrics.json")
    
    if not champion_file.exists():
        print("No champion metrics found")
        return
    
    with open(champion_file, "r") as f:
        champion = json.load(f)
    
    # Load latest run metrics from training output
    # In practice, you'd parse the training log or MLflow
    
    # Create runs table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            run_ts TIMESTAMP,
            heart_rmse REAL,
            baseline_rmse REAL,
            mse_improvement_pct REAL,
            meets_paper_claim BOOLEAN,
            promoted BOOLEAN
        )
    """)
    
    # Insert or update champion run
    conn.execute("""
        INSERT OR REPLACE INTO runs 
        (run_id, run_ts, heart_rmse, baseline_rmse, mse_improvement_pct, meets_paper_claim, promoted)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        champion.get("mlflow_run_id", "unknown"),
        champion.get("promoted_at", datetime.now().isoformat()),
        champion.get("heart_rmse", 0),
        0,  # baseline_rmse would need to be stored
        0,  # improvement would need to be stored
        champion.get("heart_rmse", 100) < 40,  # meets claim if RMSE < 40
        True
    ))
    
    conn.commit()
    
    # Create summary table for daily metrics
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_metrics (
            date DATE PRIMARY KEY,
            avg_prediction REAL,
            min_prediction REAL,
            max_prediction REAL,
            predictions_count INTEGER,
            model_version TEXT
        )
    """)
    
    # Load latest predictions if available
    pred_file = Path("predictions/latest_predictions.parquet")
    
    if pred_file.exists():
        df = pd.read_parquet(pred_file)
        
        today = datetime.now().date()
        
        conn.execute("""
            INSERT OR REPLACE INTO daily_metrics
            (date, avg_prediction, min_prediction, max_prediction, predictions_count, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            today.isoformat(),
            df["predicted_pm25"].mean(),
            df["predicted_pm25"].min(),
            df["predicted_pm25"].max(),
            len(df),
            "production"
        ))
        
        conn.commit()
    
    # Export to CSV for Grafana
    runs_df = pd.read_sql_query("SELECT * FROM runs ORDER BY run_ts DESC", conn)
    runs_df.to_csv("metrics/runs.csv", index=False)
    
    daily_df = pd.read_sql_query("SELECT * FROM daily_metrics ORDER BY date DESC", conn)
    daily_df.to_csv("metrics/daily_metrics.csv", index=False)
    
    conn.close()
    
    print(f"Metrics updated: {len(runs_df)} total runs")
    print(f"Daily metrics: {len(daily_df)} days")


def generate_metrics_summary():
    """Generate a summary JSON for email notifications"""
    
    conn = sqlite3.connect("metrics/runs.db")
    
    # Get latest run
    latest = pd.read_sql_query(
        "SELECT * FROM runs ORDER BY run_ts DESC LIMIT 1", 
        conn
    )
    
    # Get best run
    best = pd.read_sql_query(
        "SELECT * FROM runs ORDER BY heart_rmse ASC LIMIT 1",
        conn
    )
    
    conn.close()
    
    summary = {
        "latest_rmse": latest.iloc[0]["heart_rmse"] if not latest.empty else None,
        "best_rmse": best.iloc[0]["heart_rmse"] if not best.empty else None,
        "total_runs": len(latest),
        "last_updated": datetime.now().isoformat()
    }
    
    with open("metrics/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    update_metrics_database()
    generate_metrics_summary()
    print("✅ Metrics updated successfully")