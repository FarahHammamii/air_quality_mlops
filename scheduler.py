"""
scheduler.py
============
Runs the full ingest → train pipeline on a weekly schedule.
Keeps running until killed (Ctrl-C or SIGTERM).

Usage:
    python scheduler.py                  # weekly on Monday 06:00 UTC
    python scheduler.py --interval 3600  # every hour (dev/testing)
    python scheduler.py --run-now        # fire immediately then schedule
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from batch_ingest import run_ingestion
from train_pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def full_pipeline_job(rolling_days: int = 90) -> None:
    """Single job that runs ingest then training."""
    log.info("=" * 60)
    log.info("Scheduled job started at %s", datetime.now(timezone.utc).isoformat())
    log.info("=" * 60)

    try:
        ingest_result = run_ingestion(rolling_days=rolling_days)
        log.info("Ingest result: %s", ingest_result)

        if ingest_result.get("status") == "skipped":
            log.info("Ingestion skipped – no training will run.")
            return

        train_result = run_pipeline(
            data_path="data/raw/delhi_merged.parquet",
            rolling_days=rolling_days,
        )
        log.info("Training result: %s", train_result)

    except Exception:
        log.exception("Pipeline job failed!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Air quality MLOps scheduler")
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Run every N seconds instead of weekly cron (useful for testing)",
    )
    parser.add_argument(
        "--rolling-days", type=int, default=90,
        help="Rolling window in days used for training data (default: 90)",
    )
    parser.add_argument(
        "--run-now", action="store_true",
        help="Execute the pipeline immediately before entering the schedule loop",
    )
    args = parser.parse_args()

    if args.run_now:
        log.info("--run-now flag set: executing pipeline immediately.")
        full_pipeline_job(rolling_days=args.rolling_days)

    scheduler = BlockingScheduler(timezone="UTC")

    if args.interval:
        trigger = IntervalTrigger(seconds=args.interval)
        log.info("Scheduling pipeline every %d seconds.", args.interval)
    else:
        # Every Monday at 06:00 UTC
        trigger = CronTrigger(day_of_week="mon", hour=6, minute=0)
        log.info("Scheduling pipeline weekly on Monday at 06:00 UTC.")

    scheduler.add_job(
        full_pipeline_job,
        trigger=trigger,
        kwargs={"rolling_days": args.rolling_days},
        id="air_quality_pipeline",
        name="Air Quality MLOps Pipeline",
        max_instances=1,       # never overlap
        misfire_grace_time=3600,
    )

    log.info("Scheduler started. Press Ctrl-C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
