"""
forecasting.py — Forecast Generation & Inference Preparation
=============================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Load the production-selected model artefact from models/production/.
- Load the fitted FeatureEngineer artefact (same object used during training).
- Construct the inference feature matrix for the forecast origin date.
- Generate an 8-week (or configured horizon) demand forecast for all SKUs.
- Format forecast output for consumption by:
    1. risk_engine.py — for stockout/overstock scoring
    2. app/streamlit_app.py — for dashboard visualisation
    3. api/inference.py — for REST API responses
- Write forecast outputs to artifacts/predictions/.

Inference-Time Leakage Prevention
----------------------------------
- The FeatureEngineer MUST be loaded from the same artefact created during
  training — never re-fit on data that includes the forecast window.
- Lag features at inference time use only data available up to the origin date.
- Never access inventory snapshots with Snapshot_Date >= origin_date.

Output Schema
-------------
forecast_output : pd.DataFrame
  Columns:
    forecast_run_date  : date the forecast was generated
    forecast_date      : the week being forecast (Mon start)
    SKU                : product identifier
    horizon_step       : 1 … forecast_horizon_weeks
    units_forecast     : point forecast (units)
    lower_bound        : lower prediction interval (optional)
    upper_bound        : upper prediction interval (optional)
    model_name         : name of the model used
    feature_version    : version tag of the feature engineer artefact

Public API (to be implemented in Phase 6)
-----------------------------------------
    class ForecastEngine:
        __init__(model_path, feature_engineer_path)
        generate(origin_date, skus=None)  -> pd.DataFrame
        save_forecast(df, path)

Implementation Notes (TODO: fill in during Phase 6)
----------------------------------------------------
- Load model and feature_engineer lazily (on first call) for API performance.
- Add prediction interval generation (quantile regression or bootstrap).
- Include a forecast version hash for reproducibility.
- Write JSON metadata sidecar for each forecast run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import CFG, PATHS

logger = logging.getLogger(__name__)


class ForecastEngine:
    """
    Production forecast engine for weekly SKU-level demand.

    Loads the selected model and feature engineer from disk and generates
    an 8-week forward demand forecast from any given origin date.
    """

    def __init__(
        self,
        model_path:            Optional[Path] = None,
        feature_engineer_path: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        model_path             : Path to the serialised production model.
        feature_engineer_path  : Path to the serialised FeatureEngineer.
        """
        self.model_path            = model_path or PATHS.production_dir / "model.pkl"
        self.feature_engineer_path = (
            feature_engineer_path or PATHS.production_dir / "feature_engineer.pkl"
        )
        self._model            = None
        self._feature_engineer = None

    def _load_artefacts(self) -> None:
        """Lazy-load model and feature engineer from disk."""
        # TODO (Phase 6): implement using joblib.load
        raise NotImplementedError("_load_artefacts() — implement in Phase 6.")

    def generate(
        self,
        origin_date: pd.Timestamp,
        skus:        Optional[list[str]] = None,
        horizon:     Optional[int]       = None,
    ) -> pd.DataFrame:
        """
        Generate demand forecasts from the given origin date.

        Parameters
        ----------
        origin_date : pd.Timestamp
            The last week for which actuals are available.
        skus        : list[str] | None
            SKUs to forecast (default: all active SKUs).
        horizon     : int | None
            Forecast horizon in weeks (default: from CFG).

        Returns
        -------
        pd.DataFrame with forecast output schema (see module docstring).
        """
        if self._model is None:
            self._load_artefacts()
        # TODO (Phase 6): implement
        raise NotImplementedError("ForecastEngine.generate() — implement in Phase 6.")

    def save_forecast(self, df: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """
        Persist a forecast DataFrame to artifacts/predictions/.

        Returns
        -------
        Path — location of the saved file.
        """
        raise NotImplementedError("save_forecast() — implement in Phase 6.")
