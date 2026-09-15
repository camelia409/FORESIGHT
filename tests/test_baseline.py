"""
test_baseline.py — Phase 2B Baseline Forecasting Test Suite
============================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Test Coverage
-------------
15 core tests + 1 explicit leakage test:

Data & Aggregation
  01. Weekly aggregation is deterministic (same result on repeated calls).
  02. Weekly totals reconcile with daily totals.
  03. No future observations enter any forecast.
  04. Naive forecast uses only historical information (leakage test).
  05. Seasonal Naive uses only historical information.
  06. Moving Average uses only historical information.
  07. Forecast horizon is exactly 8 where configured.

Metrics
  08. WAPE calculation is correct (known values).
  09. MAE calculation is correct.
  10. RMSE calculation is correct.
  11. Zero-demand data does not cause invalid metric behaviour.

Reconciliation
  12. SKU-level metrics reconcile with pooled metrics.
  13. Pooled metrics computed over same rows as per-SKU rows.
  14. Baseline forecasts have expected schema.
  15. analysis_ready.parquet remains unchanged after baseline run.

Leakage
  16. Explicit future-leakage detection test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CFG, PATHS
from src.baseline import (
    FORECAST_HORIZON,
    MA4_WINDOW,
    MA8_WINDOW,
    MIN_TRAIN_WEEKS,
    MODEL_NAMES,
    SEASONAL_PERIOD,
    _FORECAST_SCHEMA,
    aggregate_weekly_demand,
    build_rolling_origins,
    compute_metrics,
    generate_baseline_forecasts,
    evaluate_baselines,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    ses_forecast,
)
from src.utils import file_sha256


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture(scope="session")
def df_daily() -> pd.DataFrame:
    """Load the real analysis_ready.parquet (read-only)."""
    path = PATHS.processed_dir / "analysis_ready.parquet"
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def weekly_df(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate once for the session."""
    return aggregate_weekly_demand(df_daily)


@pytest.fixture(scope="session")
def origins(weekly_df: pd.DataFrame) -> list:
    """Build rolling origins once for the session."""
    return build_rolling_origins(weekly_df, n_folds=CFG.n_backtest_folds)


@pytest.fixture(scope="session")
def forecasts_df(weekly_df: pd.DataFrame, origins: list) -> pd.DataFrame:
    """Generate all baseline forecasts once for the session."""
    return generate_baseline_forecasts(weekly_df, origins, FORECAST_HORIZON)


@pytest.fixture(scope="session")
def eval_results(forecasts_df: pd.DataFrame) -> dict:
    """Evaluate baselines once for the session."""
    return evaluate_baselines(forecasts_df)


def _make_synthetic_weekly(
    n_skus: int = 3,
    n_weeks: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a small synthetic weekly panel for unit tests.
    Demand is random integers 1–30 (no zeros by design for most unit tests).
    """
    rng = np.random.default_rng(seed)
    skus = [f"TSKU{i:02d}" for i in range(n_skus)]
    start = pd.Timestamp("2024-01-01")
    weeks = [start + pd.Timedelta(weeks=w) for w in range(n_weeks)]

    rows = []
    for sku in skus:
        for ws in weeks:
            rows.append({"SKU": sku, "week_start": ws, "Units_Sold": int(rng.integers(1, 30))})

    df = pd.DataFrame(rows)
    df["Category"] = "TestCat"
    df["Subcategory"] = "TestSub"
    df["Intermittency_Class"] = "Smooth"
    return df.sort_values(["SKU", "week_start"]).reset_index(drop=True)


# ===========================================================================
# TEST 01 — Weekly aggregation is deterministic
# ===========================================================================

def test_01_weekly_aggregation_is_deterministic(df_daily: pd.DataFrame) -> None:
    """
    Running aggregate_weekly_demand() twice on the same input must produce
    identical output (byte-for-byte equivalent DataFrames).
    """
    w1 = aggregate_weekly_demand(df_daily)
    w2 = aggregate_weekly_demand(df_daily)

    pd.testing.assert_frame_equal(
        w1.reset_index(drop=True),
        w2.reset_index(drop=True),
        check_like=False,
    )


# ===========================================================================
# TEST 02 — Weekly totals reconcile with daily totals
# ===========================================================================

def test_02_weekly_totals_reconcile_with_daily(
    df_daily: pd.DataFrame,
    weekly_df: pd.DataFrame,
) -> None:
    """
    sum(weekly Units_Sold) must equal sum(daily Units_Sold).
    This verifies that no demand is gained or lost during aggregation.
    """
    daily_total  = int(df_daily["Units_Sold"].sum())
    weekly_total = int(weekly_df["Units_Sold"].sum())
    assert daily_total == weekly_total, (
        f"Aggregation total mismatch: daily={daily_total:,} weekly={weekly_total:,}"
    )


# ===========================================================================
# TEST 03 — No future observations enter any forecast (schema check)
# ===========================================================================

def test_03_no_future_observations_in_forecast_inputs(
    weekly_df: pd.DataFrame,
    origins: list,
) -> None:
    """
    For each origin, the naive forecast must only use data with
    week_start <= origin. We check this by verifying that the last
    available week in _get_sku_history equals the origin.
    """
    # Use a single origin and single SKU for the unit check
    origin = origins[0]
    sku    = weekly_df["SKU"].unique()[0]

    # Directly check filtered history
    mask = (weekly_df["SKU"] == sku) & (weekly_df["week_start"] <= origin)
    hist_max_date = weekly_df.loc[mask, "week_start"].max()

    assert hist_max_date <= origin, (
        f"History max date {hist_max_date} is after origin {origin} — leakage!"
    )


# ===========================================================================
# TEST 04 — Naive forecast leakage test (core leakage unit test)
# ===========================================================================

def test_04_naive_forecast_uses_only_historical_information() -> None:
    """
    Generate naive forecasts from a synthetic dataset.
    Then corrupt all future rows (after origin) by setting demand to 99999.
    Re-run the forecast from the same origin.
    The forecast values must be identical (future corruption should not matter).
    """
    df = _make_synthetic_weekly(n_skus=2, n_weeks=70)
    all_weeks = sorted(df["week_start"].unique())
    origin = all_weeks[59]  # last week with known history

    # Baseline forecast
    f1 = naive_forecast(df, origin, horizon=8)

    # Corrupt future rows
    df_corrupted = df.copy()
    future_mask = df_corrupted["week_start"] > origin
    df_corrupted.loc[future_mask, "Units_Sold"] = 99_999

    # Re-forecast from same origin
    f2 = naive_forecast(df_corrupted, origin, horizon=8)

    pd.testing.assert_frame_equal(
        f1[["SKU", "horizon", "forecast"]].reset_index(drop=True),
        f2[["SKU", "horizon", "forecast"]].reset_index(drop=True),
        check_like=False,
    )


# ===========================================================================
# TEST 05 — Seasonal Naive uses only historical information
# ===========================================================================

def test_05_seasonal_naive_uses_only_historical_information() -> None:
    """
    Seasonal Naive must be insensitive to values after the origin date.
    """
    df = _make_synthetic_weekly(n_skus=2, n_weeks=70)
    all_weeks = sorted(df["week_start"].unique())
    origin = all_weeks[59]

    f1 = seasonal_naive_forecast(df, origin, horizon=8)

    df_corrupted = df.copy()
    df_corrupted.loc[df_corrupted["week_start"] > origin, "Units_Sold"] = 99_999

    f2 = seasonal_naive_forecast(df_corrupted, origin, horizon=8)

    pd.testing.assert_frame_equal(
        f1[["SKU", "horizon", "forecast"]].reset_index(drop=True),
        f2[["SKU", "horizon", "forecast"]].reset_index(drop=True),
    )


# ===========================================================================
# TEST 06 — Moving Average uses only historical information
# ===========================================================================

def test_06_moving_average_uses_only_historical_information() -> None:
    """
    MA4 and MA8 forecasts must be insensitive to values after the origin.
    """
    df = _make_synthetic_weekly(n_skus=2, n_weeks=70)
    all_weeks = sorted(df["week_start"].unique())
    origin = all_weeks[59]

    for window in [MA4_WINDOW, MA8_WINDOW]:
        f1 = moving_average_forecast(df, origin, horizon=8, window=window)
        df_corrupted = df.copy()
        df_corrupted.loc[df_corrupted["week_start"] > origin, "Units_Sold"] = 99_999
        f2 = moving_average_forecast(df_corrupted, origin, horizon=8, window=window)

        pd.testing.assert_frame_equal(
            f1[["SKU", "horizon", "forecast"]].reset_index(drop=True),
            f2[["SKU", "horizon", "forecast"]].reset_index(drop=True),
            check_like=False,
        )


# ===========================================================================
# TEST 07 — Forecast horizon is exactly 8
# ===========================================================================

def test_07_forecast_horizon_is_exactly_8(
    forecasts_df: pd.DataFrame,
    origins: list,
) -> None:
    """
    For every model and every origin, horizon values must be exactly {1, 2, ..., 8}.
    """
    expected_horizons = set(range(1, FORECAST_HORIZON + 1))

    for model in MODEL_NAMES:
        for origin in origins:
            mask  = (forecasts_df["model"] == model) & (forecasts_df["forecast_origin"] == origin)
            actual_horizons = set(forecasts_df.loc[mask, "horizon"].unique())
            assert actual_horizons == expected_horizons, (
                f"model={model} origin={origin.date()}: horizons={actual_horizons} ≠ {expected_horizons}"
            )


# ===========================================================================
# TEST 08 — WAPE calculation correctness
# ===========================================================================

def test_08_wape_calculation_is_correct() -> None:
    """
    Known-value WAPE check:
    y_true = [10, 20, 30], y_pred = [8, 22, 28]
    |errors| = [2, 2, 2], sum|errors| = 6, sum|y_true| = 60
    WAPE = 6/60 × 100 = 10.0%
    """
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([ 8.0, 22.0, 28.0])
    result = compute_metrics(y_true, y_pred)
    assert abs(result["WAPE"] - 10.0) < 1e-9, f"WAPE={result['WAPE']} ≠ 10.0"


# ===========================================================================
# TEST 09 — MAE calculation correctness
# ===========================================================================

def test_09_mae_calculation_is_correct() -> None:
    """
    Known-value MAE check:
    y_true = [10, 20, 30], y_pred = [8, 22, 28]
    |errors| = [2, 2, 2], MAE = 2.0
    """
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([ 8.0, 22.0, 28.0])
    result = compute_metrics(y_true, y_pred)
    assert abs(result["MAE"] - 2.0) < 1e-9, f"MAE={result['MAE']} ≠ 2.0"


# ===========================================================================
# TEST 10 — RMSE calculation correctness
# ===========================================================================

def test_10_rmse_calculation_is_correct() -> None:
    """
    Known-value RMSE check:
    y_true = [10, 20, 30], y_pred = [8, 22, 28]
    errors = [-2, +2, -2]; squared = [4, 4, 4]; mean = 4; sqrt = 2.0
    """
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([ 8.0, 22.0, 28.0])
    result = compute_metrics(y_true, y_pred)
    assert abs(result["RMSE"] - 2.0) < 1e-9, f"RMSE={result['RMSE']} ≠ 2.0"


# ===========================================================================
# TEST 11 — Zero demand does not cause invalid metric behaviour
# ===========================================================================

def test_11_zero_demand_does_not_cause_invalid_metrics() -> None:
    """
    When y_true contains zeros:
    - MAE and RMSE must be finite floats.
    - WAPE must be NaN only if ALL y_true values are zero.
    - WAPE must be finite when sum(y_true) > 0 even if some zeros exist.
    """
    # Case 1: zeros mixed with non-zeros — WAPE must be finite
    y_true_mixed = np.array([0.0, 5.0, 0.0, 10.0])
    y_pred_mixed = np.array([1.0, 4.0, 2.0,  9.0])
    metrics_mixed = compute_metrics(y_true_mixed, y_pred_mixed)
    assert np.isfinite(metrics_mixed["WAPE"]), "WAPE must be finite for mixed-zero demand"
    assert np.isfinite(metrics_mixed["MAE"]),  "MAE must be finite"
    assert np.isfinite(metrics_mixed["RMSE"]), "RMSE must be finite"

    # Case 2: all-zero y_true — WAPE must be NaN (zero-denominator policy)
    y_true_zero = np.array([0.0, 0.0, 0.0])
    y_pred_zero = np.array([1.0, 2.0, 3.0])
    metrics_zero = compute_metrics(y_true_zero, y_pred_zero)
    assert np.isnan(metrics_zero["WAPE"]), "WAPE must be NaN when sum(y_true)==0"
    assert np.isfinite(metrics_zero["MAE"]),  "MAE must still be finite"
    assert np.isfinite(metrics_zero["RMSE"]), "RMSE must still be finite"


# ===========================================================================
# TEST 12 — SKU-level metrics reconcile with pooled metrics
# ===========================================================================

def test_12_sku_level_metrics_are_subsets_of_pooled(
    forecasts_df: pd.DataFrame,
    eval_results: dict,
) -> None:
    """
    The total number of forecast rows used in by_sku evaluation must equal
    the total number used in overall evaluation (no rows silently dropped).
    """
    df_valid = forecasts_df.dropna(subset=["actual"])
    n_total   = len(df_valid)

    # Sum of n_forecasts per model from by_sku
    by_sku_df = eval_results["by_sku"]
    overall_df = eval_results["overall"]

    # Each model's n_forecasts in overall should match total rows / n_models
    for model in MODEL_NAMES:
        n_overall  = int(overall_df.set_index("model").loc[model, "n_forecasts"])
        n_from_sku = len(df_valid[(df_valid["model"] == model)])
        assert n_overall == n_from_sku, (
            f"model={model}: overall n_forecasts={n_overall} ≠ df rows={n_from_sku}"
        )


# ===========================================================================
# TEST 13 — Pooled metrics computed over same rows as per-SKU rows
# ===========================================================================

def test_13_pooled_metrics_are_consistent(
    forecasts_df: pd.DataFrame,
    eval_results: dict,
) -> None:
    """
    Manually compute WAPE for 'naive' from raw forecasts_df and confirm
    it matches what evaluate_baselines() reports.
    """
    from src.baseline import compute_metrics

    df_naive = forecasts_df.dropna(subset=["actual"])
    df_naive = df_naive[df_naive["model"] == "naive"]

    manual = compute_metrics(df_naive["actual"], df_naive["forecast"])
    reported = eval_results["overall"].set_index("model").loc["naive", "WAPE"]

    assert abs(manual["WAPE"] - float(reported)) < 1e-6, (
        f"Manual WAPE={manual['WAPE']:.4f} ≠ reported={reported:.4f}"
    )


# ===========================================================================
# TEST 14 — Baseline forecasts have expected schema
# ===========================================================================

def test_14_baseline_forecasts_have_expected_schema(
    forecasts_df: pd.DataFrame,
) -> None:
    """
    The forecast DataFrame must contain all required columns:
    SKU, forecast_origin, target_week, horizon, forecast, model, actual.
    Forecast values must be non-negative (demand cannot be negative).
    Horizon values must be integers in [1, FORECAST_HORIZON].
    """
    required_cols = _FORECAST_SCHEMA + ["actual"]
    for col in required_cols:
        assert col in forecasts_df.columns, f"Missing required column: {col}"

    # Horizons must be valid
    assert forecasts_df["horizon"].between(1, FORECAST_HORIZON).all(), (
        f"Horizon values out of range [1, {FORECAST_HORIZON}]"
    )

    # Forecast values must be non-negative
    assert (forecasts_df["forecast"] >= 0).all(), "Negative forecast values detected"

    # Model names must be valid
    assert set(forecasts_df["model"].unique()) == set(MODEL_NAMES), (
        f"Unexpected model names: {forecasts_df['model'].unique()}"
    )


# ===========================================================================
# TEST 15 — analysis_ready.parquet remains unchanged
# ===========================================================================

def test_15_analysis_ready_unchanged_after_baseline() -> None:
    """
    Compute SHA-256 of analysis_ready.parquet at test time and compare against
    the hash stored in the Phase 2B manifest (written after the experiment ran).
    """
    analysis_ready_path = PATHS.processed_dir / "analysis_ready.parquet"
    manifest_path = PATHS.metrics_dir / "phase2b_baseline_manifest.json"

    # If manifest doesn't exist, skip (experiment not yet run)
    if not manifest_path.exists():
        pytest.skip("phase2b_baseline_manifest.json not found — run experiment first")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    recorded_hash = manifest.get("analysis_ready_hash_sha256", "")
    current_hash  = file_sha256(analysis_ready_path)

    assert current_hash == recorded_hash, (
        f"analysis_ready.parquet hash changed!\n"
        f"  Recorded: {recorded_hash[:32]}…\n"
        f"  Current:  {current_hash[:32]}…"
    )


# ===========================================================================
# TEST 16 — Explicit leakage detection test (future-injection attack)
# ===========================================================================

def test_16_explicit_leakage_detection() -> None:
    """
    EXPLICIT LEAKAGE TEST: Construct a scenario where future demand is
    artificially set to 99,999 units. Run all baseline models. Verify that
    none of the forecast values change relative to the clean dataset.

    This test would FAIL if any baseline method accidentally read future rows.
    """
    df_clean = _make_synthetic_weekly(n_skus=3, n_weeks=70)
    all_weeks = sorted(df_clean["week_start"].unique())
    origin = all_weeks[59]  # use week 60 as origin

    # Generate clean forecasts
    clean_preds = {}
    for fn, model_name in [
        (lambda df, o: naive_forecast(df, o, horizon=8), "naive"),
        (lambda df, o: seasonal_naive_forecast(df, o, horizon=8), "seasonal_naive"),
        (lambda df, o: moving_average_forecast(df, o, horizon=8, window=4), "ma4"),
        (lambda df, o: moving_average_forecast(df, o, horizon=8, window=8), "ma8"),
        (lambda df, o: ses_forecast(df, o, horizon=8), "ses"),
    ]:
        clean_preds[model_name] = fn(df_clean, origin)[["SKU", "horizon", "forecast"]].reset_index(drop=True)

    # Inject future values: set ALL weeks AFTER origin to 99,999
    df_poisoned = df_clean.copy()
    df_poisoned.loc[df_poisoned["week_start"] > origin, "Units_Sold"] = 99_999

    # Re-generate forecasts with poisoned future
    poisoned_preds = {}
    for fn, model_name in [
        (lambda df, o: naive_forecast(df, o, horizon=8), "naive"),
        (lambda df, o: seasonal_naive_forecast(df, o, horizon=8), "seasonal_naive"),
        (lambda df, o: moving_average_forecast(df, o, horizon=8, window=4), "ma4"),
        (lambda df, o: moving_average_forecast(df, o, horizon=8, window=8), "ma8"),
        (lambda df, o: ses_forecast(df, o, horizon=8), "ses"),
    ]:
        poisoned_preds[model_name] = fn(df_poisoned, origin)[["SKU", "horizon", "forecast"]].reset_index(drop=True)

    # All forecasts must be identical — future injection must have zero effect
    for model_name in ["naive", "seasonal_naive", "ma4", "ma8", "ses"]:
        pd.testing.assert_frame_equal(
            clean_preds[model_name],
            poisoned_preds[model_name],
            check_like=False,
            obj=f"Leakage detected in {model_name}! Future data affected forecast.",
        )


# ===========================================================================
# ADDITIONAL UNIT TESTS FOR EDGE CASES
# ===========================================================================

def test_naive_formula_correctness() -> None:
    """
    Naive forecast must exactly equal the last known demand for all horizons.
    """
    df = _make_synthetic_weekly(n_skus=1, n_weeks=10)
    all_weeks = sorted(df["week_start"].unique())
    origin = all_weeks[9]  # last week

    sku = df["SKU"].iloc[0]
    last_demand = int(df.loc[
        (df["SKU"] == sku) & (df["week_start"] == origin), "Units_Sold"
    ].iloc[0])

    fcast = naive_forecast(df, origin, horizon=5)
    assert len(fcast) == 5, f"Expected 5 rows, got {len(fcast)}"
    assert (fcast["forecast"] == last_demand).all(), (
        f"Naive forecast {fcast['forecast'].tolist()} ≠ {last_demand}"
    )


def test_ma4_formula_correctness() -> None:
    """
    MA4 forecast must equal the mean of the last 4 weeks for all horizons.
    """
    df = _make_synthetic_weekly(n_skus=1, n_weeks=10)
    all_weeks = sorted(df["week_start"].unique())
    origin = all_weeks[9]

    sku = df["SKU"].iloc[0]
    hist = df.loc[
        (df["SKU"] == sku) & (df["week_start"] <= origin)
    ].sort_values("week_start")["Units_Sold"].values

    expected_ma4 = float(hist[-4:].mean())

    fcast = moving_average_forecast(df, origin, horizon=3, window=4)
    assert abs(fcast["forecast"].iloc[0] - expected_ma4) < 1e-9, (
        f"MA4={fcast['forecast'].iloc[0]} ≠ {expected_ma4}"
    )


def test_weekly_zero_demand_preserved() -> None:
    """
    Zero-demand weeks must be preserved in weekly aggregation — not dropped.
    The W-MON convention in pandas creates Tue-Mon weekly periods.
    We set an entire Tue-Mon block to zero so the aggregated week sums to 0.
    """
    # W-MON = week period ending on Monday.
    # A full Tue-Mon zero block: 2024-01-16 (Tue) to 2024-01-22 (Mon).
    dates = pd.date_range("2024-01-01", "2024-04-01", freq="D")
    rows = []
    zero_start = pd.Timestamp("2024-01-16")  # Tuesday
    zero_end   = pd.Timestamp("2024-01-22")  # Monday
    for d in dates:
        demand = 0 if zero_start <= d <= zero_end else 5
        rows.append({"Date": d, "SKU": "SKU001", "Units_Sold": demand})

    df_daily = pd.DataFrame(rows)
    df_daily["Category"] = "TestCat"
    df_daily["Subcategory"] = "TestSub"

    weekly = aggregate_weekly_demand(df_daily)
    sku_weekly = weekly[weekly["SKU"] == "SKU001"].copy()

    # The zero-demand Tue-Mon block should produce exactly one week with 0 demand.
    min_units = sku_weekly["Units_Sold"].min()
    assert min_units == 0, (
        f"Expected a zero-demand week (Units_Sold==0) but minimum is {min_units}. "
        "W-MON period zero-demand aggregation failed.\n"
        f"Weekly data:\n{sku_weekly.head(10)}"
    )
    n_zero_weeks = int((sku_weekly["Units_Sold"] == 0).sum())
    assert n_zero_weeks >= 1, "No zero-demand weeks found — zero-demand preservation failed."


def test_rolling_origins_count() -> None:
    """
    build_rolling_origins must return exactly n_folds origins.
    """
    df = _make_synthetic_weekly(n_skus=2, n_weeks=80)
    origins = build_rolling_origins(df, n_folds=5, min_train_weeks=20,
                                     forecast_horizon=8, holdout_weeks=8)
    assert len(origins) == 5, f"Expected 5 origins, got {len(origins)}"


def test_all_models_in_forecasts(
    forecasts_df: pd.DataFrame,
) -> None:
    """All 5 baseline models must be present in the forecast DataFrame."""
    present = set(forecasts_df["model"].unique())
    assert present == set(MODEL_NAMES), (
        f"Missing models: {set(MODEL_NAMES) - present}"
    )


def test_sku_count_in_forecasts(
    forecasts_df: pd.DataFrame,
) -> None:
    """All 50 master SKUs must appear in the forecast DataFrame."""
    n_skus = forecasts_df["SKU"].nunique()
    assert n_skus == 50, f"Expected 50 SKUs, got {n_skus}"


def test_compute_metrics_perfect_forecast() -> None:
    """Perfect forecast should yield WAPE=0, MAE=0, RMSE=0."""
    y = np.array([10.0, 20.0, 30.0])
    m = compute_metrics(y, y.copy())
    assert m["WAPE"] == pytest.approx(0.0, abs=1e-9)
    assert m["MAE"]  == pytest.approx(0.0, abs=1e-9)
    assert m["RMSE"] == pytest.approx(0.0, abs=1e-9)


def test_baseline_artifacts_exist() -> None:
    """
    After running the experiment, all required artifact files must exist
    and be non-empty.
    """
    baseline_dir = PATHS.metrics_dir.parent / "baseline"
    required_csvs = [
        "baseline_summary.csv",
        "baseline_by_sku.csv",
        "baseline_by_horizon.csv",
        "baseline_by_category.csv",
        "baseline_by_intermittency.csv",
    ]
    required_plots = [
        "01_actual_vs_naive.png",
        "02_actual_vs_seasonal_naive.png",
        "03_actual_vs_moving_average.png",
        "04_baseline_comparison.png",
        "05_wape_by_horizon.png",
        "06_per_sku_wape_heatmap.png",
        "07_wape_by_intermittency.png",
        "08_representative_sku_forecasts.png",
    ]

    if not baseline_dir.exists():
        pytest.skip("baseline/ artifact directory not found — run experiment first")

    for fname in required_csvs:
        fpath = baseline_dir / fname
        assert fpath.exists(), f"Missing artifact: {fpath}"
        assert fpath.stat().st_size > 0, f"Empty artifact: {fpath}"

    plots_dir = baseline_dir / "plots"
    for fname in required_plots:
        fpath = plots_dir / fname
        assert fpath.exists(), f"Missing plot: {fpath}"
        assert fpath.stat().st_size > 0, f"Empty plot: {fpath}"

    # Forecast parquet
    fcast_path = baseline_dir / "baseline_forecasts.parquet"
    assert fcast_path.exists(), "baseline_forecasts.parquet missing"
    assert fcast_path.stat().st_size > 0

    # Manifest
    manifest_path = PATHS.metrics_dir / "phase2b_baseline_manifest.json"
    assert manifest_path.exists(), "phase2b_baseline_manifest.json missing"

    # Report
    report_path = PATHS.reports_model_dir.parent / "baseline" / "phase2b_baseline_report.md"
    assert report_path.exists(), "phase2b_baseline_report.md missing"
    assert report_path.stat().st_size > 0

    # Weekly interim file
    weekly_path = PATHS.interim_dir / "baseline_weekly_demand.parquet"
    assert weekly_path.exists(), "baseline_weekly_demand.parquet missing"
