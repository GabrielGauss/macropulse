"""
Data validation for MacroPulse.

Validates raw ingested data and engineered features before they
enter the model pipeline.  Returns structured validation reports
so the pipeline can decide whether to proceed or halt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Result of a data validation pass."""

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("Validation warning: %s", msg)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)
        logger.error("Validation error: %s", msg)


def validate_raw_fred(df: pd.DataFrame) -> ValidationReport:
    """Validate raw FRED data before feature engineering."""
    report = ValidationReport()
    required = {"WALCL", "RRPONTSYD", "WTREGEN", "DGS10", "DGS2", "BAMLH0A0HYM2"}
    missing = required - set(df.columns)
    if missing:
        report.fail(f"Missing FRED columns: {missing}")

    if len(df) < 60:
        report.fail(f"Insufficient FRED data: {len(df)} rows (need ≥60)")

    # Check for excessive NaN ratio
    nan_ratio = df.isna().mean()
    for col in df.columns:
        if nan_ratio[col] > 0.3:
            report.warn(f"Column {col} has {nan_ratio[col]:.0%} NaN values")

    # Check for stale data (all values identical in tail)
    for col in df.columns:
        tail = df[col].tail(10)
        if tail.nunique() <= 1 and not tail.isna().all():
            report.warn(f"Column {col} appears stale (constant in last 10 rows)")

    return report


def validate_market_data(df: pd.DataFrame) -> ValidationReport:
    """Validate market data from Yahoo Finance."""
    report = ValidationReport()
    expected = {"sp500", "vix", "dxy"}
    missing = expected - set(df.columns)
    if missing:
        report.fail(f"Missing market columns: {missing}")

    if len(df) < 60:
        report.fail(f"Insufficient market data: {len(df)} rows (need ≥60)")

    # Sanity-check price ranges
    if "sp500" in df.columns:
        latest = df["sp500"].dropna().iloc[-1] if not df["sp500"].dropna().empty else 0
        if latest < 100 or latest > 100_000:
            report.fail(f"S&P 500 value out of range: {latest}")

    if "vix" in df.columns:
        latest = df["vix"].dropna().iloc[-1] if not df["vix"].dropna().empty else 0
        if latest < 0 or latest > 150:
            report.warn(f"VIX value unusual: {latest}")

    return report


def validate_features(df: pd.DataFrame) -> ValidationReport:
    """Validate the engineered feature matrix before model inference."""
    report = ValidationReport()

    if df.empty:
        report.fail("Feature matrix is empty.")
        return report

    if len(df) < 60:
        report.warn(f"Short feature history: {len(df)} rows")

    # Check for inf / NaN in model columns
    from data.processing.feature_engineering import MODEL_FEATURE_COLS

    for col in MODEL_FEATURE_COLS:
        if col not in df.columns:
            report.fail(f"Missing feature column: {col}")
            continue
        series = df[col]
        if series.isna().any():
            report.warn(f"NaN values in {col}: {series.isna().sum()}")
        if np.isinf(series).any():
            report.fail(f"Inf values in {col}")

    # Check for extreme outliers (>6 std from mean)
    for col in MODEL_FEATURE_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue
        mean, std = series.mean(), series.std()
        if std == 0:
            report.warn(f"Zero variance in {col}")
            continue
        z_max = ((series - mean) / std).abs().max()
        if z_max > 6:
            report.warn(f"Extreme outlier in {col}: max |z| = {z_max:.1f}")

    return report
