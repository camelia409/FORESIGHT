"""
models.py — Model Definitions, Training, and Serialization
===========================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Define the candidate ML models for weekly SKU-level demand forecasting:
  1. LightGBM Regressor (primary candidate)
  2. XGBoost Regressor (comparison candidate)
  3. Random Forest Regressor (ensemble baseline)
  4. Linear Regression (interpretable baseline)
- Provide uniform fit / predict / evaluate interface for all models.
- Support direct multi-horizon forecasting (h=1..8).
- Strictly enforce feature/target separation and non-negative demand constraints.
- Manage serialization to models/candidates/.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.utils import mae as mae_fn, rmse as rmse_fn, wape as wape_fn

logger = logging.getLogger(__name__)

# Fixed categorical features in the 50 SAFE features
CATEGORICAL_COLS = ["Category", "Subcategory", "Price_Tier"]

CATEGORY_MAPS = {
    "Category": ["Furniture", "Home Decor", "Kitchen", "Lighting", "Storage"],
    "Subcategory": [
        "Appliance", "Bedding", "Chair", "Cookware", "Dinnerware",
        "Lamp", "Organizer", "Pillow", "Racks", "Table"
    ],
    "Price_Tier": ["Budget", "Mid-Range", "Premium"]
}


# ---------------------------------------------------------------------------
# Abstract base model — all candidates must implement this interface
# ---------------------------------------------------------------------------
class BaseForecastModel(ABC):
    """Abstract base class for all FORESIGHT demand forecasting models."""

    model_name: str = "base"
    library: str = "custom"

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> "BaseForecastModel":
        """Train the model on X_train / y_train."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts for the given feature matrix."""

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series | np.ndarray) -> dict[str, float]:
        """Return a dict of evaluation metrics including WAPE, MAE, RMSE."""

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist the trained model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> "BaseForecastModel":
        """Load a previously saved model from disk."""

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {}

    @staticmethod
    def _wape(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
        """Compute WAPE — shared across all model implementations."""
        return wape_fn(y_true, y_pred)

    @staticmethod
    def _validate_inputs(X: pd.DataFrame) -> None:
        """Enforce strict feature separation: no target or metadata columns."""
        target_prefixes = ["target_", "y_t", "Units_Sold"]
        bad_targets = [c for c in X.columns if any(c.startswith(tp) for tp in target_prefixes)]
        if bad_targets:
            raise ValueError(f"CRITICAL LEAKAGE: TARGET columns found in predictors X: {bad_targets}")

        meta_cols = ["SKU", "forecast_origin_date"]
        bad_meta = [c for c in meta_cols if c in X.columns]
        if bad_meta:
            raise ValueError(f"CRITICAL: Metadata columns found in predictors X: {bad_meta}")


# ---------------------------------------------------------------------------
# LightGBM Regressor
# ---------------------------------------------------------------------------
class LightGBMForecaster(BaseForecastModel):
    """
    LightGBM-based weekly demand forecaster.
    Primary candidate model for FORESIGHT.
    """

    model_name = "lightgbm"
    library = f"lightgbm {lgb.__version__}"

    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: list[str] = []
        self.training_duration_: float = 0.0
        self.training_rows_: int = 0

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(X)
        X_out = X.copy()
        for col in CATEGORICAL_COLS:
            if col in X_out.columns:
                categories = CATEGORY_MAPS.get(col, None)
                X_out[col] = pd.Categorical(X_out[col], categories=categories)
        if "negative_margin_flag" in X_out.columns:
            X_out["negative_margin_flag"] = X_out["negative_margin_flag"].astype(int)
        return X_out

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> "LightGBMForecaster":
        t0 = time.perf_counter()
        X_prep = self._prepare_features(X_train)
        self.feature_names_ = list(X_prep.columns)
        self.training_rows_ = len(X_prep)

        y_arr = np.asarray(y_train, dtype=float)
        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(X_prep, y_arr)
        self.training_duration_ = time.perf_counter() - t0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_prep = self._prepare_features(X)
        preds = self._model.predict(X_prep)
        # Non-negative demand constraint
        return np.maximum(0.0, preds)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series | np.ndarray) -> dict[str, float]:
        preds = self.predict(X)
        y_arr = np.asarray(y_true, dtype=float)
        return {
            "WAPE": wape_fn(y_arr, preds),
            "MAE": mae_fn(y_arr, preds),
            "RMSE": rmse_fn(y_arr, preds),
        }

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self._model,
            "params": self.params,
            "feature_names": self.feature_names_,
            "training_duration": self.training_duration_,
            "training_rows": self.training_rows_,
            "model_name": self.model_name,
            "library": self.library,
        }
        joblib.dump(data, p)
        # JSON manifest sidecar
        meta_path = p.with_suffix(".json")
        meta = {
            "model_name": self.model_name,
            "library": self.library,
            "training_rows": self.training_rows_,
            "feature_count": len(self.feature_names_),
            "training_duration_seconds": self.training_duration_,
            "hyperparameters": {k: str(v) for k, v in self.params.items()},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "LightGBMForecaster":
        p = Path(path)
        data = joblib.load(p)
        instance = cls(params=data["params"])
        instance._model = data["model"]
        instance.feature_names_ = data.get("feature_names", [])
        instance.training_duration_ = data.get("training_duration", 0.0)
        instance.training_rows_ = data.get("training_rows", 0)
        return instance


# ---------------------------------------------------------------------------
# XGBoost Regressor
# ---------------------------------------------------------------------------
class XGBoostForecaster(BaseForecastModel):
    """XGBoost-based weekly demand forecaster — comparison candidate."""

    model_name = "xgboost"
    library = f"xgboost {xgb.__version__}"

    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "enable_categorical": True,
        "verbosity": 0,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[xgb.XGBRegressor] = None
        self.feature_names_: list[str] = []
        self.training_duration_: float = 0.0
        self.training_rows_: int = 0

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(X)
        X_out = X.copy()
        for col in CATEGORICAL_COLS:
            if col in X_out.columns:
                categories = CATEGORY_MAPS.get(col, None)
                X_out[col] = pd.Categorical(X_out[col], categories=categories)
        if "negative_margin_flag" in X_out.columns:
            X_out["negative_margin_flag"] = X_out["negative_margin_flag"].astype(int)
        return X_out

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> "XGBoostForecaster":
        t0 = time.perf_counter()
        X_prep = self._prepare_features(X_train)
        self.feature_names_ = list(X_prep.columns)
        self.training_rows_ = len(X_prep)

        y_arr = np.asarray(y_train, dtype=float)
        self._model = xgb.XGBRegressor(**self.params)
        self._model.fit(X_prep, y_arr)
        self.training_duration_ = time.perf_counter() - t0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_prep = self._prepare_features(X)
        preds = self._model.predict(X_prep)
        # Non-negative demand constraint
        return np.maximum(0.0, preds)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series | np.ndarray) -> dict[str, float]:
        preds = self.predict(X)
        y_arr = np.asarray(y_true, dtype=float)
        return {
            "WAPE": wape_fn(y_arr, preds),
            "MAE": mae_fn(y_arr, preds),
            "RMSE": rmse_fn(y_arr, preds),
        }

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self._model,
            "params": self.params,
            "feature_names": self.feature_names_,
            "training_duration": self.training_duration_,
            "training_rows": self.training_rows_,
            "model_name": self.model_name,
            "library": self.library,
        }
        joblib.dump(data, p)
        meta_path = p.with_suffix(".json")
        meta = {
            "model_name": self.model_name,
            "library": self.library,
            "training_rows": self.training_rows_,
            "feature_count": len(self.feature_names_),
            "training_duration_seconds": self.training_duration_,
            "hyperparameters": {k: str(v) for k, v in self.params.items()},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "XGBoostForecaster":
        p = Path(path)
        data = joblib.load(p)
        instance = cls(params=data["params"])
        instance._model = data["model"]
        instance.feature_names_ = data.get("feature_names", [])
        instance.training_duration_ = data.get("training_duration", 0.0)
        instance.training_rows_ = data.get("training_rows", 0)
        return instance


# ---------------------------------------------------------------------------
# Random Forest Regressor
# ---------------------------------------------------------------------------
class RandomForestForecaster(BaseForecastModel):
    """Random Forest weekly demand forecaster — non-boosting tree benchmark."""

    model_name = "random_forest"
    library = "scikit-learn RandomForestRegressor"

    DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[RandomForestRegressor] = None
        self.feature_names_: list[str] = []
        self.training_duration_: float = 0.0
        self.training_rows_: int = 0

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(X)
        X_out = X.copy()
        for col in CATEGORICAL_COLS:
            if col in X_out.columns:
                categories = CATEGORY_MAPS.get(col, None)
                cat_series = pd.Categorical(X_out[col], categories=categories)
                X_out[col] = cat_series.codes
        if "negative_margin_flag" in X_out.columns:
            X_out["negative_margin_flag"] = X_out["negative_margin_flag"].astype(int)
        return X_out

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> "RandomForestForecaster":
        t0 = time.perf_counter()
        X_prep = self._prepare_features(X_train)
        self.feature_names_ = list(X_prep.columns)
        self.training_rows_ = len(X_prep)

        y_arr = np.asarray(y_train, dtype=float)
        self._model = RandomForestRegressor(**self.params)
        self._model.fit(X_prep, y_arr)
        self.training_duration_ = time.perf_counter() - t0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_prep = self._prepare_features(X)
        preds = self._model.predict(X_prep)
        # Non-negative demand constraint
        return np.maximum(0.0, preds)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series | np.ndarray) -> dict[str, float]:
        preds = self.predict(X)
        y_arr = np.asarray(y_true, dtype=float)
        return {
            "WAPE": wape_fn(y_arr, preds),
            "MAE": mae_fn(y_arr, preds),
            "RMSE": rmse_fn(y_arr, preds),
        }

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self._model,
            "params": self.params,
            "feature_names": self.feature_names_,
            "training_duration": self.training_duration_,
            "training_rows": self.training_rows_,
            "model_name": self.model_name,
            "library": self.library,
        }
        joblib.dump(data, p)
        meta_path = p.with_suffix(".json")
        meta = {
            "model_name": self.model_name,
            "library": self.library,
            "training_rows": self.training_rows_,
            "feature_count": len(self.feature_names_),
            "training_duration_seconds": self.training_duration_,
            "hyperparameters": {k: str(v) for k, v in self.params.items()},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "RandomForestForecaster":
        p = Path(path)
        data = joblib.load(p)
        instance = cls(params=data["params"])
        instance._model = data["model"]
        instance.feature_names_ = data.get("feature_names", [])
        instance.training_duration_ = data.get("training_duration", 0.0)
        instance.training_rows_ = data.get("training_rows", 0)
        return instance


# ---------------------------------------------------------------------------
# Linear Forecaster (Interpretable baseline ML model)
# ---------------------------------------------------------------------------
class LinearForecaster(BaseForecastModel):
    """Linear regression demand forecaster — interpretable baseline ML model."""

    model_name = "linear"
    library = "scikit-learn LinearRegression"

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        self._model = LinearRegression(**self.params)
        self.feature_names_: list[str] = []
        self.training_duration_: float = 0.0
        self.training_rows_: int = 0

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_inputs(X)
        X_out = X.copy()
        for col in CATEGORICAL_COLS:
            if col in X_out.columns:
                categories = CATEGORY_MAPS.get(col, None)
                cat_series = pd.Categorical(X_out[col], categories=categories)
                X_out[col] = cat_series.codes
        if "negative_margin_flag" in X_out.columns:
            X_out["negative_margin_flag"] = X_out["negative_margin_flag"].astype(int)
        return X_out

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> "LinearForecaster":
        t0 = time.perf_counter()
        X_prep = self._prepare_features(X_train)
        self.feature_names_ = list(X_prep.columns)
        self.training_rows_ = len(X_prep)

        y_arr = np.asarray(y_train, dtype=float)
        self._model.fit(X_prep, y_arr)
        self.training_duration_ = time.perf_counter() - t0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_prep = self._prepare_features(X)
        preds = self._model.predict(X_prep)
        return np.maximum(0.0, preds)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series | np.ndarray) -> dict[str, float]:
        preds = self.predict(X)
        y_arr = np.asarray(y_true, dtype=float)
        return {
            "WAPE": wape_fn(y_arr, preds),
            "MAE": mae_fn(y_arr, preds),
            "RMSE": rmse_fn(y_arr, preds),
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "params": self.params}, p)

    @classmethod
    def load(cls, path: Path | str) -> "LinearForecaster":
        p = Path(path)
        data = joblib.load(p)
        inst = cls(params=data.get("params", {}))
        inst._model = data["model"]
        return inst


# ---------------------------------------------------------------------------
# Direct Multi-Horizon Forecaster
# ---------------------------------------------------------------------------
class DirectMultiHorizonForecaster:
    """
    Direct multi-horizon forecaster managing 8 independent horizon models.
    Prevents recursive forecasting error propagation.
    """

    def __init__(self, model_family: str, params: Optional[dict] = None, horizons: int = 8):
        self.model_family = model_family
        self.params = params or {}
        self.horizons = horizons
        self.models: dict[int, BaseForecastModel] = {
            h: get_model(model_family, self.params) for h in range(1, horizons + 1)
        }

    def fit_horizon(
        self,
        h: int,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray
    ) -> BaseForecastModel:
        """Train a single direct model for horizon h."""
        if h not in self.models:
            raise ValueError(f"Horizon {h} not in configured horizons 1..{self.horizons}")
        self.models[h].fit(X_train, y_train)
        return self.models[h]

    def predict_horizon(self, h: int, X: pd.DataFrame) -> np.ndarray:
        """Generate forecasts for horizon h."""
        if h not in self.models:
            raise ValueError(f"Horizon {h} not in configured horizons 1..{self.horizons}")
        return self.models[h].predict(X)

    def save(self, dir_path: Path | str) -> None:
        """Save all 8 horizon models to destination directory."""
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        for h, model in self.models.items():
            model.save(d / f"model_h{h}.joblib")

    @classmethod
    def load(cls, dir_path: Path | str, model_family: str, horizons: int = 8) -> "DirectMultiHorizonForecaster":
        """Load all 8 horizon models from destination directory."""
        d = Path(dir_path)
        instance = cls(model_family=model_family, horizons=horizons)
        model_cls = MODEL_REGISTRY[model_family]
        for h in range(1, horizons + 1):
            instance.models[h] = model_cls.load(d / f"model_h{h}.joblib")
        return instance


# ---------------------------------------------------------------------------
# Model registry — used by pipeline.py, backtesting.py, and test_forecasting.py
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[BaseForecastModel]] = {
    "lightgbm": LightGBMForecaster,
    "xgboost": XGBoostForecaster,
    "random_forest": RandomForestForecaster,
    "linear": LinearForecaster,
}


def get_model(name: str, params: Optional[dict] = None) -> BaseForecastModel:
    """
    Instantiate a model by name from the registry.

    Parameters
    ----------
    name   : str — key in MODEL_REGISTRY
    params : dict | None — hyperparameters passed to the constructor

    Raises
    ------
    ValueError if name is not in MODEL_REGISTRY.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[name]
    return cls(params) if params else cls()
