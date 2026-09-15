"""
test_forecasting.py — Tests for src/baseline.py, src/models.py, src/backtesting.py
====================================================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Covers
------
- SeasonalNaiveBaseline: fit / predict interface, WAPE computation, output schema
- BaseForecastModel: interface contract enforcement
- Model registry: get_model() by name
- RollingOriginCV: fold construction (no look-ahead), WAPE aggregation
- BacktestResults: fold metric DataFrame, overall WAPE

Status
------
All tests are PLACEHOLDERS — implement progressively in Phases 3, 4, and 5.
"""

import pytest
import pandas as pd
import numpy as np
from src.utils import wape as wape_fn


# ---------------------------------------------------------------------------
# WAPE utility tests — these CAN be implemented now (utils.py is complete)
# ---------------------------------------------------------------------------

class TestWAPEMetric:
    """Tests for the shared WAPE utility function in utils.py."""

    def test_wape_perfect_forecast(self):
        """WAPE must be 0 when y_pred == y_true."""
        y_true = pd.Series([100.0, 200.0, 50.0])
        y_pred = pd.Series([100.0, 200.0, 50.0])
        assert wape_fn(y_true, y_pred) == pytest.approx(0.0)

    def test_wape_known_value(self):
        """WAPE for a known example must match hand-calculated result."""
        y_true = pd.Series([100.0, 100.0])
        y_pred = pd.Series([80.0, 120.0])
        # |100-80| + |100-120| = 20 + 20 = 40; sum(y_true) = 200; WAPE = 20%
        assert wape_fn(y_true, y_pred) == pytest.approx(20.0)

    def test_wape_zero_denominator_returns_nan(self):
        """WAPE must return NaN when all actual values are zero."""
        y_true = pd.Series([0.0, 0.0])
        y_pred = pd.Series([10.0, 5.0])
        assert np.isnan(wape_fn(y_true, y_pred))

    def test_wape_all_ones(self):
        """WAPE for a 50% over-forecast must return 50%."""
        y_true = pd.Series([2.0, 2.0])
        y_pred = pd.Series([3.0, 3.0])
        assert wape_fn(y_true, y_pred) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# SeasonalNaiveBaseline tests
# ---------------------------------------------------------------------------

class TestSeasonalNaiveBaseline:

    def test_fit_returns_self(self):
        """fit() must return self for method chaining."""
        pytest.skip("Implement in Phase 3.")

    def test_predict_output_schema(self):
        """predict() must return DataFrame with expected columns."""
        pytest.skip("Implement in Phase 3.")

    def test_seasonal_naive_formula(self):
        """
        Forecast for week t+h must equal demand from week t+h-52.
        Verify this holds for a simple synthetic demand series.
        """
        pytest.skip("Implement in Phase 3 — core formula validation.")

    def test_cold_start_sku_fallback(self):
        """SKUs with < 52 weeks of history must use the configured fallback."""
        pytest.skip("Implement in Phase 3.")

    def test_wape_method(self):
        """SeasonalNaiveBaseline.wape() must match utils.wape()."""
        pytest.skip("Implement in Phase 3.")


# ---------------------------------------------------------------------------
# Model registry tests
# ---------------------------------------------------------------------------

class TestModelRegistry:

    def test_get_lightgbm_model(self):
        """get_model('lightgbm') must return a LightGBMForecaster instance."""
        from src.models import get_model, LightGBMForecaster
        model = get_model("lightgbm")
        assert isinstance(model, LightGBMForecaster)

    def test_get_unknown_model_raises(self):
        """get_model('unknown') must raise ValueError."""
        from src.models import get_model
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("not_a_real_model")

    def test_all_models_implement_interface(self):
        """All registered models must implement BaseForecastModel interface."""
        from src.models import MODEL_REGISTRY, BaseForecastModel
        for name, cls in MODEL_REGISTRY.items():
            assert issubclass(cls, BaseForecastModel), f"{name} does not extend BaseForecastModel"


# ---------------------------------------------------------------------------
# Rolling-origin backtesting tests
# ---------------------------------------------------------------------------

class TestRollingOriginCV:

    def test_fold_count(self):
        """RollingOriginCV must produce exactly n_folds folds."""
        pytest.skip("Implement in Phase 5.")

    def test_no_look_ahead(self):
        """
        Each training window must contain NO data at or after the origin date.
        This is the most critical test in the backtesting suite.
        """
        pytest.skip("Implement in Phase 5 — critical leakage test.")

    def test_minimum_train_requirement(self):
        """Folds must not be created if training history < min_train_weeks."""
        pytest.skip("Implement in Phase 5.")

    def test_compare_models_returns_ranked_table(self):
        """compare_models() must return DataFrame sorted by WAPE ascending."""
        pytest.skip("Implement in Phase 5.")

    def test_baseline_always_included(self):
        """The seasonal-naive baseline must appear in every comparison run."""
        pytest.skip("Implement in Phase 5.")
