"""
Parameterised database queries for MacroPulse.

All write operations use upsert semantics (INSERT … ON CONFLICT)
so the pipeline is safely idempotent.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from database.connection import get_sync_cursor

logger = logging.getLogger(__name__)


# ── Writes ───────────────────────────────────────────────────────────


def upsert_macro_features(row: dict[str, Any]) -> None:
    """Insert or update a single row in macro_features."""
    sql = """
        INSERT INTO macro_features (
            time, net_liquidity, d_liquidity, d_sp500, d_vix,
            d_dxy, d_hy_spread, d_yield_curve, d_10y, d_2y
        ) VALUES (
            %(time)s, %(net_liquidity)s, %(d_liquidity)s, %(d_sp500)s, %(d_vix)s,
            %(d_dxy)s, %(d_hy_spread)s, %(d_yield_curve)s, %(d_10y)s, %(d_2y)s
        )
        ON CONFLICT (time) DO UPDATE SET
            net_liquidity  = EXCLUDED.net_liquidity,
            d_liquidity    = EXCLUDED.d_liquidity,
            d_sp500        = EXCLUDED.d_sp500,
            d_vix          = EXCLUDED.d_vix,
            d_dxy          = EXCLUDED.d_dxy,
            d_hy_spread    = EXCLUDED.d_hy_spread,
            d_yield_curve  = EXCLUDED.d_yield_curve,
            d_10y          = EXCLUDED.d_10y,
            d_2y           = EXCLUDED.d_2y;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, row)


def upsert_macro_factors(row: dict[str, Any]) -> None:
    """Insert or update a single row in macro_factors."""
    sql = """
        INSERT INTO macro_factors (
            time, factor_1, factor_2, factor_3, factor_4, model_version
        ) VALUES (
            %(time)s, %(factor_1)s, %(factor_2)s, %(factor_3)s,
            %(factor_4)s, %(model_version)s
        )
        ON CONFLICT (time) DO UPDATE SET
            factor_1      = EXCLUDED.factor_1,
            factor_2      = EXCLUDED.factor_2,
            factor_3      = EXCLUDED.factor_3,
            factor_4      = EXCLUDED.factor_4,
            model_version = EXCLUDED.model_version;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, row)


def upsert_macro_regime(row: dict[str, Any]) -> None:
    """Insert or update a single row in macro_regimes."""
    sql = """
        INSERT INTO macro_regimes (
            time, regime, prob_expansion, prob_tightening,
            prob_risk_off, prob_recovery, risk_score,
            volatility_state, model_version
        ) VALUES (
            %(time)s, %(regime)s, %(prob_expansion)s, %(prob_tightening)s,
            %(prob_risk_off)s, %(prob_recovery)s, %(risk_score)s,
            %(volatility_state)s, %(model_version)s
        )
        ON CONFLICT (time) DO UPDATE SET
            regime           = EXCLUDED.regime,
            prob_expansion   = EXCLUDED.prob_expansion,
            prob_tightening  = EXCLUDED.prob_tightening,
            prob_risk_off    = EXCLUDED.prob_risk_off,
            prob_recovery    = EXCLUDED.prob_recovery,
            risk_score       = EXCLUDED.risk_score,
            volatility_state = EXCLUDED.volatility_state,
            model_version    = EXCLUDED.model_version;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, row)


def upsert_drift_metrics(row: dict[str, Any]) -> None:
    """Insert or update a single row in model_drift_metrics."""
    sql = """
        INSERT INTO model_drift_metrics (
            time, pca_explained_variance, regime_persistence,
            feature_mean_shift, feature_std_shift, model_version
        ) VALUES (
            %(time)s, %(pca_explained_variance)s, %(regime_persistence)s,
            %(feature_mean_shift)s, %(feature_std_shift)s, %(model_version)s
        )
        ON CONFLICT (time) DO UPDATE SET
            pca_explained_variance = EXCLUDED.pca_explained_variance,
            regime_persistence     = EXCLUDED.regime_persistence,
            feature_mean_shift     = EXCLUDED.feature_mean_shift,
            feature_std_shift      = EXCLUDED.feature_std_shift,
            model_version          = EXCLUDED.model_version;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, row)


def insert_pipeline_run(row: dict[str, Any]) -> None:
    """Log a pipeline execution."""
    sql = """
        INSERT INTO pipeline_runs (
            run_ts, status, data_lag, duration_sec,
            error_message, model_version
        ) VALUES (
            %(run_ts)s, %(status)s, %(data_lag)s, %(duration_sec)s,
            %(error_message)s, %(model_version)s
        );
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, row)


# ── Reads ────────────────────────────────────────────────────────────


def fetch_current_regime() -> dict[str, Any] | None:
    """Return the most recent regime row."""
    sql = """
        SELECT * FROM macro_regimes
        ORDER BY time DESC
        LIMIT 1;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()  # type: ignore[return-value]


def fetch_regime_history(
    start: dt.date | None = None,
    end: dt.date | None = None,
    limit: int = 90,
) -> list[dict[str, Any]]:
    """Return regime history within an optional date window."""
    conditions: list[str] = []
    params: dict[str, Any] = {"limit": limit}
    if start:
        conditions.append("time >= %(start)s")
        params["start"] = start
    if end:
        conditions.append("time <= %(end)s")
        params["end"] = end
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"""
        SELECT * FROM macro_regimes
        {where}
        ORDER BY time DESC
        LIMIT %(limit)s;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()  # type: ignore[return-value]


def fetch_latest_liquidity(limit: int = 30) -> list[dict[str, Any]]:
    """Return recent net liquidity values."""
    sql = """
        SELECT time, net_liquidity, d_liquidity
        FROM macro_features
        ORDER BY time DESC
        LIMIT %(limit)s;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"limit": limit})
        return cur.fetchall()  # type: ignore[return-value]


def fetch_latest_factors(limit: int = 30) -> list[dict[str, Any]]:
    """Return recent PCA factors."""
    sql = """
        SELECT * FROM macro_factors
        ORDER BY time DESC
        LIMIT %(limit)s;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"limit": limit})
        return cur.fetchall()  # type: ignore[return-value]


def fetch_latest_drift(limit: int = 30) -> list[dict[str, Any]]:
    """Return recent drift metrics."""
    sql = """
        SELECT * FROM model_drift_metrics
        ORDER BY time DESC
        LIMIT %(limit)s;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"limit": limit})
        return cur.fetchall()  # type: ignore[return-value]


# ── User management ──────────────────────────────────────────────────


def create_user(email: str, name: str | None = None) -> dict[str, Any]:
    """Insert a new user row. Raises if email already exists."""
    sql = """
        INSERT INTO users (email, name)
        VALUES (%(email)s, %(name)s)
        RETURNING id, email, name, created_at;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"email": email, "name": name})
        return cur.fetchone()  # type: ignore[return-value]


def get_user_by_email(email: str) -> dict[str, Any] | None:
    sql = "SELECT id, email, name, created_at, paddle_customer_id, paddle_subscription_id FROM users WHERE email = %(email)s;"
    with get_sync_cursor() as cur:
        cur.execute(sql, {"email": email})
        return cur.fetchone()  # type: ignore[return-value]


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    sql = "SELECT id, email, name, created_at, paddle_customer_id, paddle_subscription_id FROM users WHERE id = %(id)s;"
    with get_sync_cursor() as cur:
        cur.execute(sql, {"id": user_id})
        return cur.fetchone()  # type: ignore[return-value]


def update_paddle_customer(
    user_id: int,
    paddle_customer_id: str,
    paddle_subscription_id: str | None = None,
) -> None:
    sql = """
        UPDATE users
        SET paddle_customer_id     = %(cid)s,
            paddle_subscription_id = COALESCE(%(sid)s, paddle_subscription_id)
        WHERE id = %(uid)s;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"cid": paddle_customer_id, "sid": paddle_subscription_id, "uid": user_id})


def get_user_by_paddle_customer(paddle_customer_id: str) -> dict[str, Any] | None:
    sql = """
        SELECT id, email, name, paddle_customer_id, paddle_subscription_id
        FROM users WHERE paddle_customer_id = %(cid)s;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"cid": paddle_customer_id})
        return cur.fetchone()  # type: ignore[return-value]


def upgrade_user_tier(user_id: int, tier: str) -> None:
    """Set the tier on all active API keys for a user."""
    sql = "UPDATE api_keys SET tier = %(tier)s WHERE user_id = %(uid)s AND is_active = TRUE;"
    with get_sync_cursor() as cur:
        cur.execute(sql, {"tier": tier, "uid": user_id})


def create_api_key(
    user_id: int,
    key_hash: str,
    key_prefix: str,
    tier: str = "free",
) -> dict[str, Any]:
    """Store a hashed API key. Returns the created row (no plaintext)."""
    sql = """
        INSERT INTO api_keys (user_id, key_hash, key_prefix, tier)
        VALUES (%(user_id)s, %(key_hash)s, %(key_prefix)s, %(tier)s)
        RETURNING id, user_id, key_prefix, tier, is_active, created_at;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {
            "user_id": user_id,
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "tier": tier,
        })
        return cur.fetchone()  # type: ignore[return-value]


def get_api_key_by_hash(key_hash: str) -> dict[str, Any] | None:
    """
    Look up a key by its SHA-256 hash.

    Returns the key row joined with the user email, or None if not found
    or the key has been revoked.
    """
    sql = """
        SELECT k.id, k.user_id, k.key_prefix, k.tier, k.is_active,
               k.created_at, k.last_used_at, u.email, u.name
        FROM api_keys k
        JOIN users u ON u.id = k.user_id
        WHERE k.key_hash = %(hash)s
          AND k.is_active = TRUE;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"hash": key_hash})
        return cur.fetchone()  # type: ignore[return-value]


def revoke_api_keys_for_user(user_id: int) -> None:
    """Deactivate all active keys for a user (used during rotation)."""
    sql = """
        UPDATE api_keys
        SET is_active = FALSE, revoked_at = now()
        WHERE user_id = %(user_id)s AND is_active = TRUE;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"user_id": user_id})


def touch_api_key(key_hash: str) -> None:
    """Update last_used_at for a key (fire-and-forget, best effort)."""
    sql = "UPDATE api_keys SET last_used_at = now() WHERE key_hash = %(hash)s;"
    with get_sync_cursor() as cur:
        cur.execute(sql, {"hash": key_hash})


def fetch_latest_features(limit: int = 252) -> list[dict[str, Any]]:
    """Return recent macro feature rows (used by performance attribution)."""
    sql = """
        SELECT time, d_sp500, d_vix, d_dxy, d_hy_spread,
               d_yield_curve, d_10y, d_2y, net_liquidity, d_liquidity
        FROM macro_features
        ORDER BY time DESC
        LIMIT %(limit)s;
    """
    with get_sync_cursor() as cur:
        cur.execute(sql, {"limit": limit})
        return cur.fetchall()  # type: ignore[return-value]
