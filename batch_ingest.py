"""
batch_ingest.py
===============
Fetches new Delhi air quality data from OpenAQ since the last run,
saves a dated parquet shard, and updates the watermark.

Run manually:   python batch_ingest.py
Run via cron:   0 6 * * 1 cd /your/project && python batch_ingest.py
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from src.extract.openaq_client import OpenAQClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR        = Path("data/raw")
WATERMARK_FILE = Path("data/watermark.json")
DELHI_BBOX     = [76.8, 28.4, 77.4, 28.9]   # lon_min, lat_min, lon_max, lat_max
PARAMETERS     = ["pm25", "pm10", "no2", "o3"]
MAX_LOCATIONS  = 8
DEFAULT_DAYS_BACK = 90   # Request 90 days, API may return less

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Watermark helpers
# ---------------------------------------------------------------------------

def load_watermark() -> datetime:
    """Return last-fetch UTC datetime, or DEFAULT_DAYS_BACK ago on first run."""
    if WATERMARK_FILE.exists():
        data = json.loads(WATERMARK_FILE.read_text())
        ts = datetime.fromisoformat(data["last_fetch_utc"])
        log.info("Watermark loaded: %s", ts.isoformat())
        return ts
    default = datetime.now(timezone.utc) - timedelta(days=DEFAULT_DAYS_BACK)
    log.info("No watermark found – defaulting to %s", default.isoformat())
    return default


def save_watermark(ts: datetime) -> None:
    WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATERMARK_FILE.write_text(json.dumps({"last_fetch_utc": ts.isoformat()}))
    log.info("Watermark updated: %s", ts.isoformat())


# ---------------------------------------------------------------------------
# Deduplication against existing shards
# ---------------------------------------------------------------------------

def already_fetched_range(start: datetime, end: datetime) -> bool:
    """Return True if a shard covering [start, end) already exists."""
    label = _shard_label(start, end)
    return (RAW_DIR / f"{label}.parquet").exists()


def _shard_label(start: datetime, end: datetime) -> str:
    return f"delhi_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"


# ---------------------------------------------------------------------------
# Data validation and standardization
# ---------------------------------------------------------------------------

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame to match notebook's exact format."""
    df = df.copy()
    
    # Ensure datetime is timezone-aware UTC (matches notebook)
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
    
    # Ensure unit column exists (notebook has it)
    if 'unit' not in df.columns:
        unit_map = {
            'pm25': 'µg/m³',
            'pm10': 'µg/m³', 
            'no2': 'µg/m³',
            'o3': 'µg/m³'
        }
        df['unit'] = df['parameter'].map(unit_map)
    
    # Ensure location is string
    df['location'] = df['location'].astype(str)
    
    # Ensure coordinates are float
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return df


# ---------------------------------------------------------------------------
# Merge all raw shards into one combined file for training
# ---------------------------------------------------------------------------

def merge_raw_shards(rolling_days: int = 90) -> pd.DataFrame:
    """
    Merge all shards within the rolling window into one DataFrame.
    This is what train_pipeline.py reads.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=rolling_days)
    shards = sorted(RAW_DIR.glob("delhi_*.parquet"))
    
    # Exclude merged file itself
    shards = [s for s in shards if "merged" not in s.name]

    if not shards:
        raise RuntimeError(f"No raw data shards found in {RAW_DIR}")

    frames = []
    shard_info = []
    
    for shard in shards:
        try:
            df = pd.read_parquet(shard)
            original_len = len(df)
            
            # Standardize format
            df = standardize_dataframe(df)
            
            # Log the actual date range in this shard
            actual_start = df['datetime'].min()
            actual_end = df['datetime'].max()
            
            # Filter by cutoff (keep only recent data)
            df = df[df['datetime'] >= cutoff]
            
            if not df.empty:
                frames.append(df)
                shard_info.append(
                    f"{shard.name}: {original_len} rows, "
                    f"dates {actual_start.date()} to {actual_end.date()}"
                )
            else:
                log.info("Shard %s has no data within rolling window", shard.name)
        except Exception as e:
            log.error("Error reading shard %s: %s", shard, e)
            continue

    if not frames:
        raise RuntimeError(f"No raw data found within the last {rolling_days} days.")

    merged = pd.concat(frames, ignore_index=True)
    
    # Remove duplicates (same datetime, parameter, location)
    dup_before = len(merged)
    merged = merged.drop_duplicates(subset=['datetime', 'parameter', 'location'])
    dup_after = len(merged)
    
    if dup_before != dup_after:
        log.info("Removed %d duplicate records", dup_before - dup_after)
    
    merged = merged.sort_values('datetime').reset_index(drop=True)
    
    # Log shard contributions
    for info in shard_info:
        log.info("  %s", info)
    
    out_path = RAW_DIR / "delhi_merged.parquet"
    merged.to_parquet(out_path, index=False)
    log.info("Merged shard written: %s  (%d rows, %d columns)", 
             out_path, len(merged), len(merged.columns))
    
    return merged


# ---------------------------------------------------------------------------
# Compare with original data structure
# ---------------------------------------------------------------------------

def compare_with_original(merged_df: pd.DataFrame) -> dict:
    """Compare merged data structure with original notebook data."""
    original_path = RAW_DIR / "Delhi_90days.parquet"
    
    if not original_path.exists():
        log.warning("Original data not found at %s - skipping comparison", original_path)
        return {}
    
    original = pd.read_parquet(original_path)
    original = standardize_dataframe(original)
    
    comparison = {
        "columns_match": set(merged_df.columns) == set(original.columns),
        "original_columns": list(original.columns),
        "merged_columns": list(merged_df.columns),
        "original_date_range": (original['datetime'].min(), original['datetime'].max()),
        "merged_date_range": (merged_df['datetime'].min(), merged_df['datetime'].max()),
        "original_parameters": list(original['parameter'].unique()),
        "merged_parameters": list(merged_df['parameter'].unique()),
        "original_locations": list(original['location'].unique()),
        "merged_locations": list(merged_df['location'].unique()),
    }
    
    # Check value ranges
    for param in original['parameter'].unique():
        orig_vals = original[original['parameter'] == param]['value']
        merged_vals = merged_df[merged_df['parameter'] == param]['value']
        
        if len(orig_vals) > 0 and len(merged_vals) > 0:
            comparison[f"{param}_original_range"] = (orig_vals.min(), orig_vals.max())
            comparison[f"{param}_merged_range"] = (merged_vals.min(), merged_vals.max())
    
    log.info("=" * 60)
    log.info("DATA STRUCTURE COMPARISON:")
    log.info("Columns match: %s", comparison["columns_match"])
    if not comparison["columns_match"]:
        log.info("  Original: %s", comparison["original_columns"])
        log.info("  Merged:   %s", comparison["merged_columns"])
    log.info("Original date range: %s to %s", *comparison["original_date_range"])
    log.info("Merged date range:   %s to %s", *comparison["merged_date_range"])
    
    # Check if merged data actually contains newer data
    if comparison["merged_date_range"][1] > comparison["original_date_range"][1]:
        log.info("✅ Merged data contains NEWER data (beyond original)")
    else:
        log.warning("⚠️ No newer data beyond original dataset")
    
    log.info("=" * 60)
    
    return comparison


# ---------------------------------------------------------------------------
# Get actual data availability from API
# ---------------------------------------------------------------------------

def check_data_availability() -> dict:
    """Check what date range is actually available from the API."""
    client = OpenAQClient()
    
    # Make a small request to see what's available
    try:
        # Try to get just 1 day to check availability
        df = client.fetch_city_data(
            bbox=DELHI_BBOX,
            parameters=["pm25"],
            days_back=1,
            max_locations=5,
            save_local=False,
            label="check_availability",
        )
        
        if not df.empty:
            return {
                "available": True,
                "earliest": df['datetime'].min(),
                "latest": df['datetime'].max(),
                "records": len(df),
            }
        else:
            return {"available": False, "reason": "No data returned"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ingestion(rolling_days: int = 90, force_refresh: bool = False) -> dict:
    """
    Run the ingestion pipeline.
    
    Args:
        rolling_days: Number of days to keep in merged file
        force_refresh: If True, rebuild merged file from all shards even without new fetch
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # First, check what data is actually available
    log.info("Checking data availability from OpenAQ API...")
    availability = check_data_availability()
    
    if not availability.get("available", False):
        log.warning("API data check: %s", availability.get("reason", "Unknown"))
    else:
        log.info("API has data from %s to %s (%d records)",
                 availability['earliest'], availability['latest'], availability['records'])

    fetch_start = load_watermark()
    fetch_end   = datetime.now(timezone.utc)

    days_to_fetch = (fetch_end - fetch_start).days
    if days_to_fetch < 1 and not force_refresh:
        log.info("Last fetch was less than 1 day ago – skipping.")
        return {"status": "skipped", "reason": "too recent"}

    label = _shard_label(fetch_start, fetch_end)

    if already_fetched_range(fetch_start, fetch_end) and not force_refresh:
        log.info("Shard %s already exists – skipping API call.", label)
    else:
        log.info("Fetching up to %d days of data (label=%s)…", days_to_fetch, label)
        client = OpenAQClient()
        df = client.fetch_city_data(
            bbox=DELHI_BBOX,
            parameters=PARAMETERS,
            days_back=days_to_fetch,
            max_locations=MAX_LOCATIONS,
            save_local=False,
            label=label,
        )

        if df.empty:
            log.warning("API returned no data for this window.")
            return {"status": "empty", "label": label}
        
        # Log actual data range
        log.info("API returned data from %s to %s (%d rows)",
                 df['datetime'].min(), df['datetime'].max(), len(df))
        
        # Standardize format BEFORE saving
        df = standardize_dataframe(df)

        out_path = RAW_DIR / f"{label}.parquet"
        df.to_parquet(out_path, index=False)
        log.info("Shard saved: %s  (%d rows, %d columns)", out_path, len(df), len(df.columns))

    save_watermark(fetch_end)

    # Always rebuild the merged file after a successful fetch
    merged = merge_raw_shards(rolling_days=rolling_days)
    
    # Compare with original data structure
    comparison = compare_with_original(merged)
    
    # Verify merged data has expected columns
    expected_cols = {'datetime', 'value', 'parameter', 'unit', 'location', 'latitude', 'longitude'}
    if not expected_cols.issubset(set(merged.columns)):
        missing = expected_cols - set(merged.columns)
        log.error("Merged data missing required columns: %s", missing)
        return {"status": "failed", "reason": f"missing_columns: {missing}"}
    
    # Check date range makes sense
    days_span = (merged['datetime'].max() - merged['datetime'].min()).days
    log.info("Merged data spans %d days (%s to %s)", 
             days_span, merged['datetime'].min(), merged['datetime'].max())
    
    if days_span < 30:
        log.warning("Merged data spans only %d days. The API may have limited historical data.", days_span)
        log.warning("This is expected if you're working with a recent dataset. Training will work with available data.")
    
    # Provide guidance based on data availability
    if availability.get("available", False):
        api_earliest = availability['earliest']
        merged_earliest = merged['datetime'].min()
        
        if merged_earliest > api_earliest:
            log.info("Note: Your merged data starts at %s, but API has data back to %s",
                     merged_earliest, api_earliest)

    return {
        "status":       "ok",
        "label":        label,
        "fetch_start":  fetch_start.isoformat(),
        "fetch_end":    fetch_end.isoformat(),
        "merged_rows":  len(merged),
        "merged_days":  days_span,
        "columns_match": comparison.get("columns_match", False),
        "api_availability": availability if availability.get("available") else None,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest new air quality data")
    parser.add_argument("--force", action="store_true", help="Force rebuild merged file")
    parser.add_argument("--rolling-days", type=int, default=90, help="Rolling window in days")
    args = parser.parse_args()
    
    result = run_ingestion(rolling_days=args.rolling_days, force_refresh=args.force)
    log.info("Ingestion result: %s", result)