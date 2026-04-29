# Air Quality MLOps — What Changed & How to Run

## New files (drop these into your project root)

```
batch_ingest.py          # fetches new OpenAQ data, manages watermark
train_pipeline.py        # full train → evaluate → promote script
scheduler.py             # weekly APScheduler loop
docker-compose.yml       # MLflow + Grafana containers
metrics/
  __init__.py
  metrics_logger.py      # writes run results to SQLite + CSV
grafana/
  provisioning/
    datasources/sqlite.yaml
    dashboards/provider.yaml
    dashboards/air_quality_mlops.json
```

## Modified files (replace the originals)

### `src/extract/openaq_client.py`
**One change only:** `fetch_city_data()` gains a `save_path=` parameter.
When you pass it a path, the file is saved there instead of the
hardcoded `../data/raw/{label}.parquet`. The original fallback
still works exactly as before — the notebooks are unaffected.

### `src/features/window_builder.py`
**One change only:** `build_dataset()` defaults changed from
`../data/processed` / `../models/scaler.pkl` (notebook-relative)
to `data/processed` / `models/scaler.pkl` (project-root-relative).
The notebooks call it with explicit paths so they are unaffected.
The new pipeline scripts call it without paths and it just works.

### `requirements.txt`
Added: `mlflow>=2.13.0`, `APScheduler>=3.10.0`, `pyarrow>=14.0.0`

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. First-ever run (manual)
```bash
# Fetch data + train in one go
python batch_ingest.py
python train_pipeline.py

# Or combined:
python scheduler.py --run-now
```

### 3. View MLflow UI (no Docker needed)
```bash
mlflow ui --backend-store-uri mlruns
# open http://localhost:5000
```

### 4. Start Grafana + MLflow via Docker
```bash
docker compose up -d
# MLflow:  http://localhost:5000
# Grafana: http://localhost:3000  (admin / admin)
```
Grafana reads `metrics/runs.db` automatically via the SQLite plugin.
The dashboard is provisioned automatically on first start.

### 5. Run on a weekly schedule
```bash
# Keeps running, fires every Monday at 06:00 UTC
python scheduler.py

# Dev mode: fire every 5 minutes
python scheduler.py --interval 300

# Or use cron instead:
# 0 6 * * 1 cd /your/project && python batch_ingest.py && python train_pipeline.py
```

---

## Data flow

```
OpenAQ API
    │
    ▼
batch_ingest.py  ──► data/raw/delhi_YYYYMMDD_YYYYMMDD.parquet  (one shard per run)
                 ──► data/raw/delhi_merged.parquet              (rolling 90-day window)
                 ──► data/watermark.json                        (last-fetch timestamp)
    │
    ▼
train_pipeline.py
    ├── src/features/window_builder.py  →  data/processed/  +  models/scaler.pkl
    ├── src/models/baseline_model.py    →  models/baseline_best.keras
    ├── src/models/heart_model.py       →  models/heart_best.keras
    ├── MLflow                          →  mlruns/
    ├── metrics/metrics_logger.py       →  metrics/runs.db  +  metrics/runs.csv
    └── model promotion                 →  models/production/  +  models/champion_metrics.json
    │
    ▼
Grafana (reads metrics/runs.db)
MLflow UI (reads mlruns/)
```

---

## Directory structure after first run

```
your-project/
├── batch_ingest.py
├── train_pipeline.py
├── scheduler.py
├── docker-compose.yml
├── requirements.txt
├── .env                         
├── data/
│   ├── watermark.json
│   └── raw/
│       ├── delhi_20250101_20250108.parquet
│       └── delhi_merged.parquet
│   └── processed/
│       ├── X_train.npy  y_train.npy  ts_train.npy
│       ├── X_val.npy    y_val.npy    ts_val.npy
│       └── X_test.npy   y_test.npy   ts_test.npy
├── models/
│   ├── scaler.pkl
│   ├── baseline_best.keras
│   ├── heart_best.keras
│   ├── champion_metrics.json
│   └── production/
│       ├── heart_production.keras
│       ├── baseline_production.keras
│       └── scaler.pkl
├── mlruns/                      # MLflow data
├── metrics/
│   ├── runs.db                  # SQLite — Grafana source
│   └── runs.csv                 # human-readable mirror
└── grafana/
    └── provisioning/
        ├── datasources/sqlite.yaml
        └── dashboards/
            ├── provider.yaml
            └── air_quality_mlops.json
```

---

## Grafana dashboard panels

| Panel | Query |
|-------|-------|
| RMSE over time | `heart_rmse` + `baseline_rmse` vs `run_ts` |
| MSE improvement % | bar gauge, green ≥ 7.5% (paper target) |
| Latest HEART RMSE | stat, thresholds at 20 / 35 µg/m³ |
| Latest R² | stat, thresholds at 0.70 / 0.85 |
| Last batch run | timestamp of most recent run |
| MAE over time | `heart_mae` + `baseline_mae` vs `run_ts` |
| Run history table | last 20 runs, colour-coded MSE improvement |
