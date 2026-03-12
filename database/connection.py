"""
Database connection helpers for MacroPulse.

Provides both synchronous (psycopg2) and async (asyncpg) connections
for use in the pipeline and API layers respectively.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Generator

import psycopg2
import psycopg2.extras

from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_sync_connection() -> psycopg2.extensions.connection:
    """Return a new synchronous psycopg2 connection."""
    settings = get_settings()
    return psycopg2.connect(settings.database_url)


@contextlib.contextmanager
def get_sync_cursor(
    autocommit: bool = False,
) -> Generator[psycopg2.extras.RealDictCursor, None, None]:
    """Context-managed cursor with automatic commit / rollback."""
    conn = get_sync_connection()
    conn.autocommit = autocommit
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
            if not autocommit:
                conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Execute the DDL schema against the connected database."""
    from pathlib import Path

    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text()
    with get_sync_cursor(autocommit=True) as cur:
        cur.execute(sql)
    logger.info("Database schema initialised.")
