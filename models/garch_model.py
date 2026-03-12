"""
GARCH(1,1) volatility model for MacroPulse.

Fits ARCH-family GARCH(1,1) models on return series (d_sp500, d_vix)
to produce conditional volatility estimates and classify the current
vol regime.  The GARCH classification replaces the simple VIX
threshold used in the legacy RegimeClassifier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from arch import arch_model

from config.settings import get_settings

logger = logging.getLogger(__name__)

VolState = Literal["low", "normal", "elevated", "crisis"]

# Standard-deviation thresholds for volatility bucket classification.
_VOL_LOW = 0.5
_VOL_NORMAL = 1.5
_VOL_ELEVATED = 2.5


class GARCHModel:
    """
    GARCH(1,1) wrapper with fit / forecast / classify / persist semantics.

    One instance is created per return series (e.g. d_sp500 or d_vix).
    After fitting, the model stores the long-run volatility standard
    deviation so that conditional vols can be expressed in normalised
    (z-score) units for classification.
    """

    def __init__(self, series_name: str = "returns") -> None:
        """
        Parameters
        ----------
        series_name:
            Human-readable label for the return series (used in log messages
            and artifact filenames).
        """
        self.series_name = series_name
        self._arch_result = None  # fitted ARCHModelResult
        self._long_run_vol: float | None = None  # unconditional vol (std dev)

    # ── Training ─────────────────────────────────────────────────────

    def fit(self, returns_series: pd.Series) -> "GARCHModel":
        """
        Fit GARCH(1,1) on the supplied return series.

        The unconditional (long-run) volatility is computed from the
        fitted omega / alpha / beta parameters so that classify_vol_state
        can operate without external calibration.

        Parameters
        ----------
        returns_series:
            Daily log-return series (e.g. d_sp500 or d_vix).  NaN rows
            are dropped before fitting.
        """
        clean = returns_series.dropna().astype(float)
        if len(clean) < 60:
            raise ValueError(
                f"GARCH fit requires >=60 observations; got {len(clean)} "
                f"for series '{self.series_name}'."
            )

        model = arch_model(
            clean * 100,  # scale to percentage returns for numerical stability
            vol="Garch",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        self._arch_result = model.fit(disp="off", show_warning=False)

        # Unconditional variance: omega / (1 - alpha - beta)
        params = self._arch_result.params
        omega = float(params.get("omega", params.iloc[1]))
        alpha = float(params.get("alpha[1]", params.iloc[2]))
        beta = float(params.get("beta[1]", params.iloc[3]))
        denom = max(1 - alpha - beta, 1e-6)
        self._long_run_vol = float(np.sqrt(omega / denom))

        logger.info(
            "GARCH fitted for '%s': omega=%.4f alpha=%.4f beta=%.4f "
            "long_run_vol=%.4f (in pct-return units)",
            self.series_name,
            omega,
            alpha,
            beta,
            self._long_run_vol,
        )
        return self

    # ── Inference ────────────────────────────────────────────────────

    def forecast_vol(self, returns_series: pd.Series) -> float:
        """
        Compute the 1-step-ahead conditional volatility forecast.

        The series is re-fitted on the supplied data so that the forecast
        reflects the most recent observations.  Returns the conditional
        standard deviation in the same (percentage-return) units used
        internally.

        Parameters
        ----------
        returns_series:
            Daily log-return series.  At least 30 observations are required.

        Returns
        -------
        float
            1-day-ahead conditional volatility (percentage-return std dev).
        """
        if self._arch_result is None:
            raise RuntimeError(
                "GARCHModel must be fitted before calling forecast_vol()."
            )
        clean = returns_series.dropna().astype(float)
        if len(clean) < 30:
            logger.warning(
                "forecast_vol: only %d observations for '%s'; "
                "using long-run vol as fallback.",
                len(clean),
                self.series_name,
            )
            return self._long_run_vol or 1.0

        model = arch_model(
            clean * 100,
            vol="Garch",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        result = model.fit(disp="off", show_warning=False)
        forecast = result.forecast(horizon=1, reindex=False)
        cond_var = float(forecast.variance.iloc[-1, 0])
        cond_vol = float(np.sqrt(max(cond_var, 0.0)))
        logger.info(
            "GARCH 1-step forecast for '%s': cond_vol=%.4f",
            self.series_name,
            cond_vol,
        )
        return cond_vol

    def classify_vol_state(self, conditional_vol: float) -> VolState:
        """
        Map a conditional volatility value to a named volatility bucket.

        The classification is relative to the long-run (unconditional)
        volatility estimated during fitting, expressed as a z-score
        normalised by the long-run standard deviation.

        Buckets
        -------
        - "low"      : normalised vol < 0.5 std
        - "normal"   : 0.5 – 1.5 std
        - "elevated" : 1.5 – 2.5 std
        - "crisis"   : > 2.5 std

        Parameters
        ----------
        conditional_vol:
            Conditional standard deviation in percentage-return units.

        Returns
        -------
        VolState
        """
        if self._long_run_vol is None or self._long_run_vol == 0:
            logger.warning(
                "Long-run vol not set for '%s'; defaulting to 'normal'.",
                self.series_name,
            )
            return "normal"

        normalised = conditional_vol / self._long_run_vol

        if normalised < _VOL_LOW:
            state: VolState = "low"
        elif normalised < _VOL_NORMAL:
            state = "normal"
        elif normalised < _VOL_ELEVATED:
            state = "elevated"
        else:
            state = "crisis"

        logger.info(
            "Vol state for '%s': cond_vol=%.4f long_run_vol=%.4f "
            "normalised=%.2f -> '%s'",
            self.series_name,
            conditional_vol,
            self._long_run_vol,
            normalised,
            state,
        )
        return state

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, version: str | None = None) -> Path:
        """
        Serialize the fitted GARCH model to the artifacts directory.

        Saves both the ARCHModelResult and the long-run volatility scalar
        as a single joblib bundle so that classify_vol_state works after
        loading without refitting.

        Parameters
        ----------
        version:
            Artifact version label (default from settings).

        Returns
        -------
        Path
            The artifacts directory.
        """
        if self._arch_result is None:
            raise RuntimeError("Cannot save an unfitted GARCHModel.")
        settings = get_settings()
        version = version or settings.default_model_version
        artifacts = Path(settings.model_artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)

        bundle = {
            "series_name": self.series_name,
            "arch_result": self._arch_result,
            "long_run_vol": self._long_run_vol,
        }
        path = artifacts / f"garch_{self.series_name}_{version}.pkl"
        joblib.dump(bundle, path)
        logger.info("Saved GARCH artifact to %s", path)
        return artifacts

    @classmethod
    def load(cls, series_name: str, version: str | None = None) -> "GARCHModel":
        """
        Load a previously saved GARCH model from disk.

        Parameters
        ----------
        series_name:
            The return series label used when the model was saved.
        version:
            Artifact version label.

        Returns
        -------
        GARCHModel
        """
        settings = get_settings()
        version = version or settings.default_model_version
        artifacts = Path(settings.model_artifacts_dir)
        path = artifacts / f"garch_{series_name}_{version}.pkl"

        bundle: dict = joblib.load(path)
        instance = cls.__new__(cls)
        instance.series_name = bundle["series_name"]
        instance._arch_result = bundle["arch_result"]
        instance._long_run_vol = bundle["long_run_vol"]
        logger.info(
            "Loaded GARCH model for '%s' (version=%s)", series_name, version
        )
        return instance
