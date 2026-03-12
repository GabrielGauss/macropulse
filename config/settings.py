"""
MacroPulse configuration module.

Loads environment variables with sensible defaults for local development.
Uses pydantic-settings for typed, validated configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the MacroPulse system."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── General ──────────────────────────────────────────────────────
    app_name: str = "MacroPulse"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # ── Database ─────────────────────────────────────────────────────
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "macropulse"
    db_password: str = "macropulse"
    db_name: str = "macropulse"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def async_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── FRED API ─────────────────────────────────────────────────────
    fred_api_key: str = ""

    # ── Anthropic API (for AI commentary endpoint) ───────────────────
    anthropic_api_key: str = ""

    # ── Model artifacts ──────────────────────────────────────────────
    model_artifacts_dir: str = str(_PROJECT_ROOT / "models" / "artifacts")
    default_model_version: str = "v1"

    # ── PCA / HMM hyper-parameters ───────────────────────────────────
    pca_n_components: int = 4
    pca_variance_threshold: float = 0.80
    hmm_n_regimes: int = 4
    hmm_n_iter: int = 200
    hmm_covariance_type: str = "full"

    # ── Auth ────────────────────────────────────────────────────────
    api_keys: list[str] = []  # Empty = dev-mode (no auth enforced)

    # ── Paddle Billing ───────────────────────────────────────────────
    paddle_api_key: str = ""                        # Bearer token for Paddle API
    paddle_webhook_secret: str = ""                 # From Paddle dashboard → Notifications
    paddle_environment: str = "sandbox"             # "sandbox" | "production"
    paddle_starter_price_id: str = ""               # pri_... for $49/mo
    paddle_pro_price_id: str = ""                   # pri_... for $199/mo
    # Product IDs (used for webhook event filtering)
    paddle_starter_product_id: str = "pro_01kkhzzr1c1f1fta693c6p6nzv"
    paddle_pro_product_id: str = "pro_01kkj01cx467jt6v4c5g2hakrd"
    # URL Paddle redirects to after successful checkout
    paddle_success_url: str = "https://macropulse.live/welcome"
    paddle_client_token: str = ""  # From Paddle dashboard → Developer Tools → Authentication

    # ── CORS ────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── Rate limiting ────────────────────────────────────────────────
    # Requests per API key per calendar day.  0 = unlimited (internal / Pro tier).
    # Free: 50  |  Starter: 500  |  Pro: 0
    rate_limit_per_day: int = 0  # default off for local dev

    # ── Alerting ────────────────────────────────────────────────────
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_recipients: list[str] = []
    webhook_url: str = ""  # Slack / Discord / generic webhook

    # ── Scheduler ───────────────────────────────────────────────────
    pipeline_cron_hour: int = 18  # 18:00 UTC (after US market close)
    pipeline_cron_minute: int = 30

    # ── Pipeline ─────────────────────────────────────────────────────
    data_lookback_days: int = 756  # ~3 years of trading days
    fred_series: list[str] = [
        "WALCL",       # Fed Total Assets
        "RRPONTSYD",   # Reverse Repo
        "WTREGEN",     # Treasury General Account
        "DGS10",       # 10-Year Treasury Yield
        "DGS2",        # 2-Year Treasury Yield
        "BAMLH0A0HYM2",  # HY OAS Spread
    ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
