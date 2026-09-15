"""
backtesting.py — Rolling-Origin Time-Series Cross-Validation
=============================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Implement rolling-origin (walk-forward) backtesting for all forecasting
  models including the seasonal-naive baseline.
- Ensure strictly no look-ahead bias: the training window for fold k
  contains ONLY data with date < origin_k.
- Produce fold-level and aggregate WAPE metrics for each model.
- Enable side-by-side comparison of all candidate models.
- Write backtesting results to artifacts/metrics/ for reproducibility.

Rolling-Origin Protocol
-----------------------
For each fold k (k = 1 … n_backtest_folds):

  Training window   : [start_date, origin_k)             — all history before origin
  Evaluation window : [origin_k, origin_k + H weeks)     — H = forecast_horizon_weeks

  origin_k advances by STEP weeks each fold (default STEP = 1 week).

  Minimum training requirement: min_train_weeks (from config, default 52).
  First valid origin: start_date + min_train_weeks.

Schematic
---------
  Week: 1 ... 52 | 53 ... 60    <- fold 1: train=1..52, eval=53..60 (H=8)
  Week: 1 ... 53 | 54 ... 61    <- fold 2: train=1..53, eval=54..61
  ...
  Week: 1 ... N  | N+1 ... N+8  <- fold K

WAPE aggregation:
  Per-fold WAPE  = WAPE over all (SKU, week) cells in that fold's eval window.
  Overall WAPE   = WAPE over all (SKU, week, fold) cells across all folds.

Public API (to be implemented in Phase 5)
-----------------------------------------
    class RollingOriginCV:
        __init__(n_folds, min_train_weeks, step_weeks, horizon)
        run(df, model, feature_engineer)  -> BacktestResults
        compare_models(df, models, fe)    -> pd.DataFrame

    class BacktestResults:
        fold_metrics   : pd.DataFrame
        overall_wape   : float
        model_name     : str
        plot()
        save(path)

Implementation Notes (TODO: fill in during Phase 5)
----------------------------------------------------
- Re-fit FeatureEngineer on each fold's training window separately.
- Re-fit model on each fold's training window (no weight carry-over).
- Store (fold_id, origin_date, SKU, y_true, y_pred) for every fold.
- Include the seasonal-naive baseline in every comparison run.
- A model is only considered "better" if its overall WAPE is statistically
  lower than the baseline (use bootstrap confidence intervals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from src.config import CFG
from src.models import BaseForecastModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Metrics and predictions for a single backtesting fold."""
    fold_id:      int
    origin_date:  pd.Timestamp
    model_name:   str
    wape:         float
    mae:          float
    predictions:  pd.DataFrame = field(repr=False)  # (SKU, forecast_date, y_true, y_pred)


@dataclass
class BacktestResults:
    """Aggregated results across all backtesting folds for one model."""
    model_name:    str
    n_folds:       int
    overall_wape:  float
    fold_results:  list[FoldResult] = field(repr=False)

    @property
    def fold_metrics(self) -> pd.DataFrame:
        """Return a DataFrame of per-fold WAPE and MAE."""
        return pd.DataFrame([
            {"fold_id": fr.fold_id, "origin_date": fr.origin_date,
             "wape": fr.wape, "mae": fr.mae}
            for fr in self.fold_results
        ])

    def summary(self) -> str:
        return (
            f"BacktestResults(model={self.model_name}, "
            f"folds={self.n_folds}, overall_WAPE={self.overall_wape:.2f}%)"
        )

    def save(self, path: Any) -> None:
        """Save fold-level metrics to CSV and all predictions to parquet."""
        raise NotImplementedError("BacktestResults.save() — implement in Phase 5.")

    def plot(self) -> None:
        """Plot WAPE per fold and prediction vs actuals for a sample SKU."""
        raise NotImplementedError("BacktestResults.plot() — implement in Phase 5.")


# ---------------------------------------------------------------------------
# Rolling-origin CV engine
# ---------------------------------------------------------------------------
class RollingOriginCV:
    """
    Rolling-origin (walk-forward) cross-validation engine.

    Guarantees:
    - No look-ahead bias (strict temporal ordering).
    - FeatureEngineer re-fitted on each fold's training window.
    - Model re-trained from scratch on each fold.
    """

    def __init__(
        self,
        n_folds:         int = CFG.n_backtest_folds,
        min_train_weeks: int = CFG.min_train_weeks,
        step_weeks:      int = 1,
        horizon:         int = CFG.forecast_horizon_weeks,
    ):
        self.n_folds         = n_folds
        self.min_train_weeks = min_train_weeks
        self.step_weeks      = step_weeks
        self.horizon         = horizon

    def run(
        self,
        df:               pd.DataFrame,
        model:            BaseForecastModel,
        feature_engineer: Any,
    ) -> BacktestResults:
        """
        Execute rolling-origin backtesting for a single model.

        Parameters
        ----------
        df               : Weekly demand panel (all available history).
        model            : Any class implementing BaseForecastModel.
        feature_engineer : FeatureEngineer instance (will be re-fit each fold).

        Returns
        -------
        BacktestResults
        """
        raise NotImplementedError("RollingOriginCV.run() — implement in Phase 5.")

    def compare_models(
        self,
        df:               pd.DataFrame,
        models:           list[BaseForecastModel],
        feature_engineer: Any,
    ) -> pd.DataFrame:
        """
        Run backtesting for multiple models and return a comparison table.

        Returns
        -------
        pd.DataFrame with columns: model_name, overall_wape, n_folds, ...
        Sorted by overall_wape ascending (best model first).
        """
        raise NotImplementedError("RollingOriginCV.compare_models() — implement in Phase 5.")
