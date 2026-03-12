#!/usr/bin/env python3
"""
Initialise the MacroPulse database schema.

Useful when running outside Docker (Docker auto-runs schema.sql
via the init.d volume mount).

Usage:
    python scripts/init_db.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.connection import init_schema  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    init_schema()
    print("Schema initialised successfully.")


if __name__ == "__main__":
    main()
