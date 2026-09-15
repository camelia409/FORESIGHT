"""
validation.py — Schema & Data-Quality Validation
=================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Enforce the data contract established during Phase 0 (Data Audit).
- Validate schema, primary keys, referential integrity, domain constraints,
  and business rules discovered during audit.
- Produce structured ValidationResult objects — never modify DataFrames.
- Distinguish FAIL (structural data corruption) from WARNING (known business
  policy issues) from PASS (clean checks).
- Generate a complete phase-level validation report and CSV artefact.

Design Principles
-----------------
- Validation is ALWAYS read-only. This module NEVER modifies a DataFrame.
- Known business-policy issues (orphan SKUs, negative margins, unresolved IV)
  produce WARNING results — they do NOT cause pipeline failure.
- Structural failures (missing columns, duplicate PKs, negative prices) produce
  FAIL results and halt the pipeline.
- Every check returns a ValidationResult with enough evidence to reproduce it.

Public API
----------
    validate_sales(df)              -> list[ValidationResult]
    validate_sku_master(df)         -> list[ValidationResult]
    validate_calendar(df)           -> list[ValidationResult]
    validate_inventory(df, sku_df)  -> list[ValidationResult]
    validate_referential_integrity(sales, sku, cal, inv) -> list[ValidationResult]
    validate_panel_completeness(sales) -> list[ValidationResult]
    run_all_validations(data)       -> ValidationSuite
    save_validation_report(suite, report_dir, metrics_dir)
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import CFG, PATHS

logger = logging.getLogger(__name__)

# Sentinel for "no evidence collected"
_NO_EVIDENCE: str = ""


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class ValidationResult:
    """
    Result of one atomic validation check.

    Attributes
    ----------
    dataset      : Logical dataset name ('sales', 'sku_master', etc.)
    check_name   : Short, machine-readable name for the check.
    status       : 'PASS' | 'WARNING' | 'FAIL'
    severity     : 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
    affected_rows: Number of rows that failed this check (0 for PASS).
    message      : Human-readable description of the result.
    detail       : Optional extra evidence (sample values, counts, etc.)
    """
    dataset:       str
    check_name:    str
    status:        str         # PASS | WARNING | FAIL
    severity:      str         # INFO | WARNING | ERROR | CRITICAL
    affected_rows: int
    message:       str
    detail:        str = field(default=_NO_EVIDENCE)

    def __post_init__(self) -> None:
        if self.status not in {"PASS", "WARNING", "FAIL"}:
            raise ValueError(f"Invalid status: {self.status!r}")
        if self.severity not in {"INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid severity: {self.severity!r}")


@dataclass
class ValidationSuite:
    """
    Aggregated collection of ValidationResults across all datasets.

    Properties
    ----------
    n_pass, n_warning, n_fail : counts by status
    has_fails                 : True if any FAIL results exist
    has_critical              : True if any CRITICAL+FAIL results exist
    """
    run_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    def add_all(self, results: list[ValidationResult]) -> None:
        self.results.extend(results)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def n_warning(self) -> int:
        return sum(1 for r in self.results if r.status == "WARNING")

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def has_fails(self) -> bool:
        return self.n_fail > 0

    @property
    def has_critical(self) -> bool:
        return any(r.status == "FAIL" and r.severity == "CRITICAL" for r in self.results)

    def summary_line(self) -> str:
        return (
            f"ValidationSuite: {len(self.results)} checks — "
            f"PASS={self.n_pass}  WARNING={self.n_warning}  FAIL={self.n_fail}"
        )


class ValidationError(RuntimeError):
    """Raised when a CRITICAL/FAIL condition blocks the pipeline."""


# ===========================================================================
# Internal helpers
# ===========================================================================

def _pass(dataset: str, check: str, msg: str = "") -> ValidationResult:
    return ValidationResult(
        dataset=dataset, check_name=check,
        status="PASS", severity="INFO",
        affected_rows=0, message=msg or f"{check}: OK",
    )


def _warning(dataset: str, check: str, n_rows: int, msg: str, detail: str = "") -> ValidationResult:
    return ValidationResult(
        dataset=dataset, check_name=check,
        status="WARNING", severity="WARNING",
        affected_rows=n_rows, message=msg, detail=detail,
    )


def _fail(
    dataset: str, check: str, n_rows: int, msg: str,
    detail: str = "", severity: str = "ERROR",
) -> ValidationResult:
    return ValidationResult(
        dataset=dataset, check_name=check,
        status="FAIL", severity=severity,
        affected_rows=n_rows, message=msg, detail=detail,
    )


def _required_columns(df: pd.DataFrame, dataset: str, required: list[str]) -> list[ValidationResult]:
    """Check that all required columns are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [_fail(
            dataset, "required_columns",
            n_rows=0,
            msg=f"Missing required columns: {missing}",
            severity="CRITICAL",
            detail=f"Present: {sorted(df.columns.tolist())}",
        )]
    return [_pass(dataset, "required_columns",
                  f"All {len(required)} required columns present.")]


def _unique_pk(df: pd.DataFrame, dataset: str, pk_cols: list[str]) -> ValidationResult:
    """Check primary-key uniqueness."""
    dupes = df.duplicated(subset=pk_cols, keep=False)
    n = dupes.sum()
    if n > 0:
        sample = df.loc[dupes, pk_cols].head(5).to_dict(orient="records")
        return _fail(
            dataset, "pk_uniqueness", n,
            msg=f"Duplicate primary key ({pk_cols}): {n} affected rows.",
            detail=f"Sample duplicates: {sample}",
            severity="CRITICAL",
        )
    return _pass(dataset, "pk_uniqueness",
                 f"Primary key ({pk_cols}) is unique across all {len(df)} rows.")


def _no_nat(df: pd.DataFrame, dataset: str, col: str) -> ValidationResult:
    """Check a date column has no NaT values."""
    n = df[col].isna().sum()
    if n > 0:
        return _fail(
            dataset, f"no_nat_{col}", n,
            msg=f"Column '{col}' has {n} NaT/null values after date parsing.",
            severity="CRITICAL",
        )
    return _pass(dataset, f"no_nat_{col}",
                 f"Column '{col}': 0 NaT values.")


def _non_negative(df: pd.DataFrame, dataset: str, col: str) -> ValidationResult:
    """Check that a numeric column has no negative values."""
    n = (df[col] < 0).sum()
    if n > 0:
        sample = df.loc[df[col] < 0, col].head(5).tolist()
        return _fail(
            dataset, f"non_negative_{col}", n,
            msg=f"Column '{col}' has {n} negative value(s).",
            detail=f"Sample: {sample}",
        )
    return _pass(dataset, f"non_negative_{col}",
                 f"Column '{col}': all values >= 0.")


def _positive(df: pd.DataFrame, dataset: str, col: str) -> ValidationResult:
    """Check that a numeric column has only strictly positive values."""
    n = (df[col] <= 0).sum()
    if n > 0:
        sample = df.loc[df[col] <= 0, col].head(5).tolist()
        return _fail(
            dataset, f"positive_{col}", n,
            msg=f"Column '{col}' has {n} zero-or-negative value(s).",
            detail=f"Sample: {sample}",
        )
    return _pass(dataset, f"positive_{col}",
                 f"Column '{col}': all values > 0.")


def _no_nulls(df: pd.DataFrame, dataset: str, col: str) -> ValidationResult:
    """Check that a column has no null values."""
    n = df[col].isna().sum()
    if n > 0:
        return _fail(dataset, f"no_nulls_{col}", n,
                     msg=f"Column '{col}' has {n} null values.",
                     severity="ERROR")
    return _pass(dataset, f"no_nulls_{col}",
                 f"Column '{col}': 0 nulls.")


def _allowed_values(df: pd.DataFrame, dataset: str, col: str, allowed: set) -> ValidationResult:
    """Check that a column contains only allowed values."""
    invalid = ~df[col].isin(allowed)
    n = invalid.sum()
    if n > 0:
        bad = df.loc[invalid, col].unique().tolist()[:10]
        return _fail(
            dataset, f"allowed_values_{col}", n,
            msg=f"Column '{col}' has {n} row(s) with values outside {allowed}.",
            detail=f"Found: {bad}",
        )
    return _pass(dataset, f"allowed_values_{col}",
                 f"Column '{col}': all values in {allowed}.")


def _sku_pattern(df: pd.DataFrame, dataset: str, col: str = "SKU") -> ValidationResult:
    """Check that SKU values match ^SKU[0-9]+$."""
    pattern = re.compile(r"^SKU\d+$")
    bad = ~df[col].astype(str).str.match(pattern)
    n = bad.sum()
    if n > 0:
        sample = df.loc[bad, col].unique().tolist()[:10]
        return _fail(
            dataset, "sku_format", n,
            msg=f"Column '{col}' has {n} value(s) not matching ^SKU[0-9]+$.",
            detail=f"Offending values: {sample}",
        )
    return _pass(dataset, "sku_format",
                 f"All {df[col].nunique()} unique SKUs match ^SKU[0-9]+$.")


# ===========================================================================
# Per-dataset validators
# ===========================================================================

_SALES_REQUIRED = ["Date", "SKU", "Units_Sold", "Revenue", "Price", "Promotion"]
_SKU_REQUIRED   = ["SKU", "Product_Name", "Category", "Subcategory", "Launch_Date",
                   "Cost_Price", "Selling_Price", "Gross_Margin_Per_Unit"]
_CAL_REQUIRED   = ["date", "year", "month", "quarter", "week", "day_of_week",
                   "is_weekend", "season", "holiday", "is_holiday", "promotion_event"]
_INV_REQUIRED   = ["Snapshot_Date", "SKU", "Current_Stock", "On_Order",
                   "Lead_Time_Days", "Safety_Stock", "Reorder_Point", "Inventory_Value"]


def validate_sales(df: pd.DataFrame) -> list[ValidationResult]:
    """
    Validate sales_daily against the Phase 0 data contract.

    Checks
    ------
    A. Required columns present
    B. Date: no NaT
    C. SKU: matches ^SKU[0-9]+$, no nulls
    D. Primary key (Date, SKU) unique
    E. Units_Sold >= 0
    F. Price > 0
    G. Revenue >= 0
    H. Promotion in {0, 1}
    I. Revenue ≈ Units_Sold × Price (within revenue_price_tol)
    """
    dataset = "sales_daily"
    results: list[ValidationResult] = []

    # A — required columns
    col_check = _required_columns(df, dataset, _SALES_REQUIRED)
    results.extend(col_check)
    if any(r.status == "FAIL" for r in col_check):
        return results  # cannot continue without required columns

    # B — Date validity
    results.append(_no_nat(df, dataset, "Date"))

    # C — SKU format + no nulls
    results.append(_no_nulls(df, dataset, "SKU"))
    results.append(_sku_pattern(df, dataset, "SKU"))

    # D — Primary key uniqueness
    results.append(_unique_pk(df, dataset, ["Date", "SKU"]))

    # E — Units_Sold >= 0
    results.append(_non_negative(df, dataset, "Units_Sold"))

    # F — Price > 0
    results.append(_positive(df, dataset, "Price"))

    # G — Revenue >= 0
    results.append(_non_negative(df, dataset, "Revenue"))

    # H — Promotion in {0, 1}
    results.append(_allowed_values(df, dataset, "Promotion", {0, 1}))

    # I — Revenue ≈ Units_Sold × Price
    tol = CFG.revenue_price_tol
    implied    = df["Units_Sold"] * df["Price"]
    diff       = (df["Revenue"] - implied).abs()
    bad_rev    = diff > tol
    n_bad      = bad_rev.sum()
    if n_bad > 0:
        worst = diff[bad_rev].nlargest(3).to_dict()
        results.append(_fail(
            dataset, "revenue_consistency", n_bad,
            msg=f"Revenue != Units_Sold × Price for {n_bad} row(s) (tol={tol}).",
            detail=f"Largest discrepancies (index: diff): {worst}",
        ))
    else:
        results.append(_pass(dataset, "revenue_consistency",
                             f"Revenue = Units_Sold × Price for all {len(df)} rows (tol={tol})."))

    _log_results(results)
    return results


def validate_sku_master(df: pd.DataFrame) -> list[ValidationResult]:
    """
    Validate sku_master against the Phase 0 data contract.

    Checks
    ------
    A. Required columns present
    B. SKU: unique, no nulls, matches pattern
    C. Cost_Price > 0
    D. Selling_Price > 0
    E. Gross_Margin_Per_Unit ≈ Selling_Price − Cost_Price
    F. negative_margin_flag (WARNING, not FAIL — engineering policy)
    G. Launch_Date: no NaT
    """
    dataset = "sku_master"
    results: list[ValidationResult] = []

    col_check = _required_columns(df, dataset, _SKU_REQUIRED)
    results.extend(col_check)
    if any(r.status == "FAIL" for r in col_check):
        return results

    # B
    results.append(_no_nulls(df, dataset, "SKU"))
    results.append(_unique_pk(df, dataset, ["SKU"]))
    results.append(_sku_pattern(df, dataset, "SKU"))

    # C, D
    results.append(_positive(df, dataset, "Cost_Price"))
    results.append(_positive(df, dataset, "Selling_Price"))

    # E — Gross_Margin_Per_Unit ≈ Selling_Price − Cost_Price
    tol = 0.02
    implied_margin = df["Selling_Price"] - df["Cost_Price"]
    diff = (df["Gross_Margin_Per_Unit"] - implied_margin).abs()
    n_bad = (diff > tol).sum()
    if n_bad > 0:
        sample = df.loc[diff > tol, ["SKU","Gross_Margin_Per_Unit"]].head(3).to_dict("records")
        results.append(_fail(
            dataset, "gross_margin_consistency", n_bad,
            msg=f"Gross_Margin_Per_Unit != Selling_Price - Cost_Price for {n_bad} row(s).",
            detail=str(sample),
        ))
    else:
        results.append(_pass(dataset, "gross_margin_consistency",
                             "Gross_Margin_Per_Unit is consistent for all SKUs."))

    # F — NEGATIVE MARGIN: WARNING per engineering policy (preserve_and_flag)
    neg = df["Cost_Price"] > df["Selling_Price"]
    n_neg = neg.sum()
    if n_neg > 0:
        neg_skus = df.loc[neg, "SKU"].tolist()
        results.append(_warning(
            dataset, "negative_margin_flag", n_neg,
            msg=(
                f"{n_neg} SKU(s) have Cost_Price > Selling_Price. "
                f"Policy: preserve_and_flag. These SKUs remain eligible for forecasting. "
                f"Business review required."
            ),
            detail=f"Affected SKUs: {neg_skus}",
        ))
    else:
        results.append(_pass(dataset, "negative_margin_flag",
                             "No SKUs with negative gross margin."))

    # G — Launch_Date
    results.append(_no_nat(df, dataset, "Launch_Date"))

    _log_results(results)
    return results


def validate_calendar(df: pd.DataFrame) -> list[ValidationResult]:
    """
    Validate calendar against the Phase 0 data contract.

    Checks
    ------
    A. Required columns present
    B. date: unique, no NaT
    C. year in [2024, 2025]
    D. month in [1..12]
    E. week in [1..53]
    F. is_weekend in {0, 1}
    G. is_holiday in {0, 1}
    H. holiday NaN: structural absence (INFO) when is_holiday == 0
       holiday NaN when is_holiday == 1: ERROR
    I. promotion_event NaN: structural absence (INFO) — not an error
    """
    dataset = "calendar"
    results: list[ValidationResult] = []

    col_check = _required_columns(df, dataset, _CAL_REQUIRED)
    results.extend(col_check)
    if any(r.status == "FAIL" for r in col_check):
        return results

    # B
    results.append(_no_nat(df, dataset, "date"))
    results.append(_unique_pk(df, dataset, ["date"]))

    # C — year range
    valid_years = {2024, 2025}
    bad_year = ~df["year"].isin(valid_years)
    n = bad_year.sum()
    if n > 0:
        results.append(_fail(dataset, "year_range", n,
                             f"'year' has {n} row(s) outside {valid_years}.",
                             detail=str(df.loc[bad_year, "year"].unique().tolist())))
    else:
        results.append(_pass(dataset, "year_range",
                             f"'year': all values in {valid_years}."))

    # D — month range
    bad_month = ~df["month"].between(1, 12)
    n = bad_month.sum()
    if n > 0:
        results.append(_fail(dataset, "month_range", n,
                             f"'month' has {n} value(s) outside [1, 12]."))
    else:
        results.append(_pass(dataset, "month_range",
                             "'month': all values in [1, 12]."))

    # E — week range
    bad_week = ~df["week"].between(1, 53)
    n = bad_week.sum()
    if n > 0:
        results.append(_fail(dataset, "week_range", n,
                             f"'week' has {n} value(s) outside [1, 53]."))
    else:
        results.append(_pass(dataset, "week_range",
                             "'week': all values in [1, 53]."))

    # F, G
    results.append(_allowed_values(df, dataset, "is_weekend", {0, 1}))
    results.append(_allowed_values(df, dataset, "is_holiday", {0, 1}))

    # H — holiday NaN semantics
    holiday_nan   = df["holiday"].isna()
    is_hol_flag   = df["is_holiday"] == 1
    # NaN where is_holiday == 0: structural absence (INFO, not error)
    structural_nan = (holiday_nan & ~is_hol_flag).sum()
    results.append(ValidationResult(
        dataset=dataset, check_name="holiday_nan_structural",
        status="PASS", severity="INFO",
        affected_rows=int(structural_nan),
        message=(
            f"{structural_nan} row(s) with holiday=NaN and is_holiday=0. "
            "Per audit finding: NaN = structural absence of event. Not an error."
        ),
    ))
    # NaN where is_holiday == 1: ERROR (holiday name should be present)
    error_nan = (holiday_nan & is_hol_flag).sum()
    if error_nan > 0:
        results.append(_fail(
            dataset, "holiday_nan_with_flag", int(error_nan),
            msg=f"{error_nan} row(s) have is_holiday=1 but holiday=NaN (name missing).",
            severity="ERROR",
        ))
    else:
        results.append(_pass(dataset, "holiday_nan_with_flag",
                             "No rows with is_holiday=1 and holiday=NaN."))

    # I — promotion_event NaN: structural absence (INFO only)
    promo_nan = df["promotion_event"].isna().sum()
    results.append(ValidationResult(
        dataset=dataset, check_name="promotion_event_nan",
        status="PASS", severity="INFO",
        affected_rows=int(promo_nan),
        message=(
            f"{promo_nan} row(s) with promotion_event=NaN. "
            "Per audit finding: NaN = no promotion on that date. Not an error."
        ),
    ))

    _log_results(results)
    return results


def validate_inventory(
    df:     pd.DataFrame,
    sku_df: Optional[pd.DataFrame] = None,
) -> list[ValidationResult]:
    """
    Validate inventory_snapshots against the Phase 0 data contract.

    Parameters
    ----------
    df     : inventory_snapshots DataFrame.
    sku_df : sku_master DataFrame (used for orphan-SKU check).
             If None, the orphan check is skipped with a WARNING.

    Checks
    ------
    A. Required columns present
    B. Snapshot_Date: no NaT
    C. SKU: no nulls, matches pattern
    D. Primary key (Snapshot_Date, SKU) unique
    E. Current_Stock >= 0
    F. On_Order >= 0
    G. Safety_Stock >= 0
    H. Reorder_Point >= 0
    I. Lead_Time_Days > 0
    J. Inventory_Value >= 0
    K. Orphan SKU check — WARNING (not FAIL) per engineering policy
    L. Inventory_Value basis — WARNING (unresolved) per engineering policy
    """
    dataset = "inventory_snapshots"
    results: list[ValidationResult] = []

    col_check = _required_columns(df, dataset, _INV_REQUIRED)
    results.extend(col_check)
    if any(r.status == "FAIL" for r in col_check):
        return results

    # B
    results.append(_no_nat(df, dataset, "Snapshot_Date"))

    # C
    results.append(_no_nulls(df, dataset, "SKU"))
    results.append(_sku_pattern(df, dataset, "SKU"))

    # D
    results.append(_unique_pk(df, dataset, ["Snapshot_Date", "SKU"]))

    # E–H — non-negative inventory quantities
    for col in ["Current_Stock", "On_Order", "Safety_Stock", "Reorder_Point"]:
        results.append(_non_negative(df, dataset, col))

    # I — Lead_Time_Days > 0
    results.append(_positive(df, dataset, "Lead_Time_Days"))

    # J — Inventory_Value >= 0
    results.append(_non_negative(df, dataset, "Inventory_Value"))

    # K — Orphan SKU check
    if sku_df is not None:
        master_skus = set(sku_df["SKU"].dropna())
        inv_skus    = set(df["SKU"].dropna())
        orphan_skus = inv_skus - master_skus
        n_orphan_rows = df["SKU"].isin(orphan_skus).sum()

        if orphan_skus:
            results.append(_warning(
                dataset, "orphan_sku_check", int(n_orphan_rows),
                msg=(
                    f"{len(orphan_skus)} inventory SKU(s) have no corresponding sku_master record. "
                    f"Affected rows: {n_orphan_rows}. "
                    f"Policy: orphan_sku_treatment='{CFG.orphan_sku_treatment}'. "
                    "These rows are preserved and quarantined — NOT removed."
                ),
                detail=(
                    f"Orphan SKUs: {sorted(orphan_skus)[:20]}"
                    + (f" … (+{len(orphan_skus)-20} more)" if len(orphan_skus) > 20 else "")
                ),
            ))
        else:
            results.append(_pass(dataset, "orphan_sku_check",
                                 "All inventory SKUs are present in sku_master."))
    else:
        results.append(_warning(
            dataset, "orphan_sku_check", 0,
            msg="Orphan SKU check skipped: sku_master not provided.",
        ))

    # L — Inventory_Value basis warning
    basis = CFG.inventory_valuation_basis
    if basis == "unresolved":
        results.append(_warning(
            dataset, "inventory_value_basis", 0,
            msg=(
                "Inventory_Value valuation basis is UNRESOLVED. "
                "Per Phase 0b investigation: unit_iv does not match Cost_Price "
                "or Selling_Price in sku_master (mean errors: 137%, 165%). "
                "Inventory_Value is preserved as-is. "
                "Monetary risk calculations are BLOCKED until basis is confirmed."
            ),
            detail="Set inventory_valuation_basis in config.yaml after ERP/WMS confirmation.",
        ))
    else:
        results.append(_pass(
            dataset, "inventory_value_basis",
            f"Inventory_Value basis confirmed: '{basis}'.",
        ))

    _log_results(results)
    return results


# ===========================================================================
# Cross-dataset validators
# ===========================================================================

def validate_referential_integrity(
    sales: pd.DataFrame,
    sku:   pd.DataFrame,
    cal:   pd.DataFrame,
    inv:   pd.DataFrame,
) -> list[ValidationResult]:
    """
    Cross-table referential integrity checks.

    Checks
    ------
    1. sales.SKU ⊆ sku_master.SKU                  [PASS expected]
    2. sales.Date ⊆ calendar.date                   [PASS expected]
    3. inventory.SKU ⊆ sku_master.SKU               [WARNING expected — 150 orphans]
    4. inventory.Snapshot_Date ⊆ calendar.date      [PASS expected]
    """
    dataset = "cross_dataset"
    results: list[ValidationResult] = []

    master_skus  = set(sku["SKU"].dropna())
    cal_dates    = set(cal["date"].dropna())
    sales_dates  = set(sales["Date"].dropna())
    sales_skus   = set(sales["SKU"].dropna())
    inv_skus     = set(inv["SKU"].dropna())
    inv_dates    = set(inv["Snapshot_Date"].dropna())

    # 1 — sales.SKU ⊆ sku_master.SKU
    missing_sales_skus = sales_skus - master_skus
    if missing_sales_skus:
        results.append(_fail(
            dataset, "sales_sku_in_master",
            n_rows=int(sales["SKU"].isin(missing_sales_skus).sum()),
            msg=f"sales_daily contains {len(missing_sales_skus)} SKU(s) absent from sku_master.",
            detail=f"Missing: {sorted(missing_sales_skus)[:10]}",
            severity="CRITICAL",
        ))
    else:
        results.append(_pass(dataset, "sales_sku_in_master",
                             f"All {len(sales_skus)} sales SKUs present in sku_master."))

    # 2 — sales.Date ⊆ calendar.date
    missing_sales_dates = sales_dates - cal_dates
    if missing_sales_dates:
        results.append(_fail(
            dataset, "sales_date_in_calendar",
            n_rows=int(sales["Date"].isin(missing_sales_dates).sum()),
            msg=f"sales_daily has {len(missing_sales_dates)} date(s) absent from calendar.",
            detail=f"Sample missing dates: {sorted(str(d) for d in missing_sales_dates)[:5]}",
            severity="CRITICAL",
        ))
    else:
        results.append(_pass(dataset, "sales_date_in_calendar",
                             f"All {len(sales_dates)} sales dates present in calendar."))

    # 3 — inventory.SKU ⊆ sku_master.SKU — WARNING per policy
    orphan_skus = inv_skus - master_skus
    if orphan_skus:
        n_rows = int(inv["SKU"].isin(orphan_skus).sum())
        results.append(_warning(
            dataset, "inventory_sku_in_master", n_rows,
            msg=(
                f"inventory_snapshots contains {len(orphan_skus)} SKU(s) absent from sku_master "
                f"({n_rows} rows). Policy: '{CFG.orphan_sku_treatment}'. "
                "These rows are preserved and quarantined."
            ),
            detail=(
                f"Orphan SKUs (first 20): {sorted(orphan_skus)[:20]}"
                + (f" … +{len(orphan_skus)-20}" if len(orphan_skus) > 20 else "")
            ),
        ))
    else:
        results.append(_pass(dataset, "inventory_sku_in_master",
                             "All inventory SKUs present in sku_master."))

    # 4 — inventory.Snapshot_Date ⊆ calendar.date
    missing_inv_dates = inv_dates - cal_dates
    if missing_inv_dates:
        results.append(_fail(
            dataset, "inventory_date_in_calendar",
            n_rows=int(inv["Snapshot_Date"].isin(missing_inv_dates).sum()),
            msg=f"inventory_snapshots has {len(missing_inv_dates)} date(s) absent from calendar.",
            detail=f"Sample: {sorted(str(d) for d in missing_inv_dates)[:5]}",
            severity="CRITICAL",
        ))
    else:
        results.append(_pass(dataset, "inventory_date_in_calendar",
                             f"All {len(inv_dates)} inventory snapshot dates present in calendar."))

    _log_results(results)
    return results


def validate_panel_completeness(sales: pd.DataFrame) -> list[ValidationResult]:
    """
    Check that sales_daily forms a complete SKU × date panel.

    Notes
    -----
    Audit finding: panel is 100% complete (50 SKUs × 731 days = 36,550 rows).
    This check remains important for future data refreshes.
    """
    dataset = "sales_daily"
    results: list[ValidationResult] = []

    skus  = sales["SKU"].dropna().unique()
    dates = sales["Date"].dropna().unique()

    expected = len(skus) * len(dates)
    actual   = len(sales)

    if actual == expected:
        results.append(_pass(
            dataset, "panel_completeness",
            f"Panel is complete: {len(skus)} SKUs × {len(dates)} dates = {actual:,} rows.",
        ))
    else:
        gap = expected - actual
        results.append(_fail(
            dataset, "panel_completeness", int(gap),
            msg=(
                f"Panel has {gap:,} missing cell(s). "
                f"Expected {expected:,} rows ({len(skus)} SKUs × {len(dates)} dates), "
                f"found {actual:,}."
            ),
            severity="ERROR",
        ))

    _log_results(results)
    return results


# ===========================================================================
# Orchestrator
# ===========================================================================

def run_all_validations(data: dict[str, pd.DataFrame]) -> ValidationSuite:
    """
    Run all validations on the four loaded datasets.

    Parameters
    ----------
    data : dict with keys 'sales', 'sku', 'calendar', 'inventory'
           as returned by data_ingestion.load_all_data().

    Returns
    -------
    ValidationSuite containing all ValidationResult objects.
    """
    suite = ValidationSuite()

    logger.info("[validation] Running validate_sales…")
    suite.add_all(validate_sales(data["sales"]))

    logger.info("[validation] Running validate_sku_master…")
    suite.add_all(validate_sku_master(data["sku"]))

    logger.info("[validation] Running validate_calendar…")
    suite.add_all(validate_calendar(data["calendar"]))

    logger.info("[validation] Running validate_inventory…")
    suite.add_all(validate_inventory(data["inventory"], sku_df=data["sku"]))

    logger.info("[validation] Running validate_referential_integrity…")
    suite.add_all(validate_referential_integrity(
        data["sales"], data["sku"], data["calendar"], data["inventory"]
    ))

    logger.info("[validation] Running validate_panel_completeness…")
    suite.add_all(validate_panel_completeness(data["sales"]))

    logger.info("[validation] %s", suite.summary_line())
    return suite


# ===========================================================================
# Report generation
# ===========================================================================

def _log_results(results: list[ValidationResult]) -> None:
    """Log each check result at the appropriate log level."""
    for r in results:
        if r.status == "PASS":
            logger.info("  [%-8s %-8s] %-40s %s", r.status, r.severity, r.check_name, r.message[:80])
        elif r.status == "WARNING":
            logger.warning("  [%-8s %-8s] %-40s %s", r.status, r.severity, r.check_name, r.message[:120])
        else:
            logger.error("  [%-8s %-8s] %-40s %s", r.status, r.severity, r.check_name, r.message[:120])


def save_validation_report(
    suite:       ValidationSuite,
    report_dir:  Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Save the validation suite as a markdown report and a CSV artefact.

    Parameters
    ----------
    suite       : Completed ValidationSuite.
    report_dir  : Directory for the markdown report (default: PATHS.reports_data_quality_dir).
    metrics_dir : Directory for the CSV artefact (default: PATHS.metrics_dir).

    Returns
    -------
    Tuple of (markdown_path, csv_path).
    """
    report_dir  = Path(report_dir  or PATHS.reports_data_quality_dir)
    metrics_dir = Path(metrics_dir or PATHS.metrics_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    md_path  = report_dir  / "phase1_validation_report.md"
    csv_path = metrics_dir / "phase1_validation_results.csv"

    # ── Markdown report ────────────────────────────────────────────────────
    lines: list[str] = []
    def p(s=""): lines.append(s)

    p("# PROJECT FORESIGHT — Phase 1A Validation Report")
    p(f"> Generated: {suite.run_timestamp} UTC")
    p(f"> Policy: orphan_sku_treatment=`{CFG.orphan_sku_treatment}`  "
      f"negative_margin_treatment=`{CFG.negative_margin_treatment}`  "
      f"inventory_valuation_basis=`{CFG.inventory_valuation_basis}`")
    p()
    p("## Summary")
    p()
    p(f"| Status  | Count |")
    p(f"|---------|-------|")
    p(f"| PASS    | {suite.n_pass} |")
    p(f"| WARNING | {suite.n_warning} |")
    p(f"| FAIL    | {suite.n_fail} |")
    p(f"| **Total** | **{len(suite.results)}** |")
    p()
    if suite.has_fails:
        p("> [!CAUTION]")
        p(f"> {suite.n_fail} FAIL check(s) detected. Pipeline halted for CRITICAL failures.")
    else:
        p("> [!NOTE]")
        p("> No FAIL checks. All structural integrity checks passed.")
    p()

    # Group by status
    for status, icon in [("FAIL", "FAIL"), ("WARNING", "WARN"), ("PASS", "PASS")]:
        group = [r for r in suite.results if r.status == status]
        if not group:
            continue
        p(f"## {icon} Checks ({len(group)})")
        p()
        p(f"| Dataset | Check | Severity | Affected Rows | Message |")
        p(f"|---------|-------|----------|---------------|---------|")
        for r in group:
            msg_trunc = (r.message[:120] + "…") if len(r.message) > 120 else r.message
            p(f"| {r.dataset} | {r.check_name} | {r.severity} | {r.affected_rows} | {msg_trunc} |")
        p()

    p("---")
    p("*Phase 1A complete. No raw files were modified. No preprocessing performed.*")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("[validation] Report saved -> %s", md_path)

    # ── CSV artefact ───────────────────────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "dataset", "check_name", "status", "severity", "affected_rows", "message", "detail"
        ])
        writer.writeheader()
        for r in suite.results:
            writer.writerow({
                "dataset":       r.dataset,
                "check_name":    r.check_name,
                "status":        r.status,
                "severity":      r.severity,
                "affected_rows": r.affected_rows,
                "message":       r.message,
                "detail":        r.detail,
            })
    logger.info("[validation] CSV artefact saved -> %s", csv_path)

    return md_path, csv_path
