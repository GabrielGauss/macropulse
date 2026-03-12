#!/usr/bin/env python3
"""
Run the MacroPulse daily pipeline.

Usage:
    python scripts/run_daily_pipeline.py
    python scripts/run_daily_pipeline.py --date 2026-03-10 --model-version v2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pipelines.daily_pipeline import run_daily_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="MacroPulse daily pipeline runner")
    parser.add_argument(
        "--date",
        type=lambda s: dt.date.fromisoformat(s),
        default=None,
        help="Target date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Model artifact version to use (e.g. v1).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    result = run_daily_pipeline(
        target_date=args.date,
        model_version=args.model_version,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
