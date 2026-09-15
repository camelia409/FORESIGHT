"""
data_ingestion.py — Controlled Raw Data Loading
================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Provide the single, authoritative entry-point for loading all four raw CSVs.
- Enforce immutability of raw data: callers receive deep copies of loaded frames.
- Perform minimal, non-destructive type coercions (date parsing, explicit dtypes).
- Validate that required columns exist at load time — fail fast and clearly.
- Log every load event with row/column counts for full traceability.
- Never filter rows, impute values, rename columns, or modify values.
- Keep this module free of all business logic.

Public API
----------
    load_sales()         -> pd.DataFrame    (sales_daily.csv)
    load_sku_master()    -> pd.DataFrame    (sku_master.csv)
    load_calendar()      -> pd.DataFrame    (calendar.csv)
    load_inventory()     -> pd.DataFrame    (inventory_snapshots.csv)
    load_all_data()      -> dict[str, pd.DataFrame]

Column contracts (from Phase 0 audit — must match actual files exactly)
-----------------------------------------------------------------------
    sales_daily        : Date, SKU, Units_Sold, Revenue, Price, Promotion
    sku_master         : SKU, Product_Name, Category, Subcategory, Launch_Date,
                         Cost_Price, Selling_Price, Gross_Margin_Per_Unit
    calendar           : date, year, month, quarter, week, day_of_week,
                         is_weekend, season, holiday, is_holiday, promotion_event
    inventory_snapshots: Snapshot_Date, SKU, Current_Stock, On_Order,
                         Lead_Time_Days, Safety_Stock, Reorder_Point, Inventory_Value
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import CFG, PATHS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column contracts — required columns per dataset
# Derived from the actual CSV files (Phase 0 audit confirmed).
# Any change here must be accompanied by an audit update.
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "sales_daily": [
        "Date", "SKU", "Units_Sold", "Revenue", "Price", "Promotion"
    ],
    "sku_master": [
        "SKU", "Product_Name", "Category", "Subcategory", "Launch_Date",
        "Cost_Price", "Selling_Price", "Gross_Margin_Per_Unit",
    ],
    "calendar": [
        "date", "year", "month", "quarter", "week", "day_of_week",
        "is_weekend", "season", "holiday", "is_holiday", "promotion_event",
    ],
    "inventory_snapshots": [
        "Snapshot_Date", "SKU", "Current_Stock", "On_Order",
        "Lead_Time_Days", "Safety_Stock", "Reorder_Point", "Inventory_Value",
    ],
}

# Explicit dtype declarations — non-destructive; date columns handled separately
_SALES_DTYPES: dict = {
    "SKU":        "object",
    "Units_Sold": "int64",
    "Revenue":    "float64",
    "Price":      "float64",
    "Promotion":  "int64",
}
_SKU_DTYPES: dict = {
    "SKU":                   "object",
    "Product_Name":          "object",
    "Category":              "object",
    "Subcategory":           "object",
    "Cost_Price":            "float64",
    "Selling_Price":         "float64",
    "Gross_Margin_Per_Unit": "float64",
}
_CALENDAR_DTYPES: dict = {
    "year":       "int64",
    "month":      "int64",
    "week":       "int64",
    "is_weekend": "int64",
    "is_holiday": "int64",
}
_INVENTORY_DTYPES: dict = {
    "SKU":             "object",
    "Current_Stock":   "int64",
    "On_Order":        "int64",
    "Lead_Time_Days":  "int64",
    "Safety_Stock":    "int64",
    "Reorder_Point":   "int64",
    "Inventory_Value": "float64",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class IngestionError(RuntimeError):
    """Raised when a raw file cannot be loaded or fails column validation."""


def _check_columns(df: pd.DataFrame, dataset: str) -> None:
    """
    Verify that all required columns are present in the loaded DataFrame.

    Parameters
    ----------
    df      : The loaded DataFrame.
    dataset : Key into _REQUIRED_COLUMNS.

    Raises
    ------
    IngestionError if any required column is missing.
    """
    required = set(_REQUIRED_COLUMNS[dataset])
    present  = set(df.columns)
    missing  = required - present
    if missing:
        raise IngestionError(
            f"[data_ingestion] '{dataset}' is missing required columns: {sorted(missing)}\n"
            f"  Found columns: {sorted(present)}\n"
            f"  Check the raw CSV file and the _REQUIRED_COLUMNS contract."
        )


def _read_csv(
    path:      Path,
    dataset:   str,
    dtype:     dict,
    date_cols: list[str],
) -> pd.DataFrame:
    """
    Read a CSV file, coerce dtypes, parse dates, and validate required columns.

    Parameters
    ----------
    path      : Absolute path to the CSV file.
    dataset   : Logical name (used for error messages and column validation).
    dtype     : Dict of column -> dtype for non-date columns.
    date_cols : List of column names to parse as datetime64.

    Returns
    -------
    pd.DataFrame — a deep copy of the loaded data. Raw values are unchanged.

    Raises
    ------
    IngestionError : File not found or required columns missing.
    """
    if not path.exists():
        raise IngestionError(
            f"[data_ingestion] Raw file not found: {path}\n"
            f"  Ensure '{path.name}' is in data/raw/ before running the pipeline.\n"
            f"  The pipeline never generates raw files — they must be provided."
        )

    try:
        df = pd.read_csv(path, dtype=dtype, low_memory=False)
    except Exception as exc:
        raise IngestionError(
            f"[data_ingestion] Failed to read '{path.name}': {exc}"
        ) from exc

    # Parse date columns — errors='coerce' produces NaT for invalid values
    # so that the validation layer (not ingestion) can report them clearly.
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            logger.warning(
                "[data_ingestion] Date column '%s' not found in '%s'.",
                col, dataset
            )

    _check_columns(df, dataset)

    # Return a deep copy — callers cannot mutate the raw source
    result = df.copy(deep=True)

    logger.info(
        "[data_ingestion] Loaded %-28s rows=%-6d  cols=%d",
        path.name, len(result), result.shape[1]
    )
    return result


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_sales() -> pd.DataFrame:
    """
    Load data/raw/sales_daily.csv.

    Returns
    -------
    pd.DataFrame
        Columns: Date (datetime64[ns]), SKU (str), Units_Sold (int64),
                 Revenue (float64), Price (float64), Promotion (int64).

    Notes
    -----
    - Raw values are preserved exactly — no filtering, imputation, or aggregation.
    - 36,550 rows expected (50 SKUs × 731 days — confirmed complete panel).
    - Revenue = Units_Sold × Price is verified by the validation layer, not here.
    """
    return _read_csv(
        path      = PATHS.raw_sales,
        dataset   = "sales_daily",
        dtype     = _SALES_DTYPES,
        date_cols = ["Date"],
    )


def load_sku_master() -> pd.DataFrame:
    """
    Load data/raw/sku_master.csv.

    Returns
    -------
    pd.DataFrame
        Columns: SKU (str), Product_Name (str), Category (str), Subcategory (str),
                 Launch_Date (datetime64[ns]), Cost_Price (float64),
                 Selling_Price (float64), Gross_Margin_Per_Unit (float64).

    Notes
    -----
    - 50 SKU records expected (SKU001–SKU050).
    - 16 SKUs have Cost_Price > Selling_Price — KNOWN AUDIT FINDING.
      Per engineering policy (negative_margin_treatment = preserve_and_flag),
      these records are NOT corrected or filtered here.
      The negative_margin_flag is derived in the preprocessing layer.
    """
    return _read_csv(
        path      = PATHS.raw_sku_master,
        dataset   = "sku_master",
        dtype     = _SKU_DTYPES,
        date_cols = ["Launch_Date"],
    )


def load_calendar() -> pd.DataFrame:
    """
    Load data/raw/calendar.csv.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime64[ns]), year (int64), month (int64),
                 quarter (str/object), week (int64), day_of_week (str/object),
                 is_weekend (int64), season (str/object), holiday (object),
                 is_holiday (int64), promotion_event (object).

    Notes
    -----
    - 731 date rows expected (2024-01-01 through 2025-12-31).
    - NaN in 'holiday' and 'promotion_event' is STRUCTURAL ABSENCE (no event),
      NOT a data-quality error.  Per audit finding + config.calendar_nan_treatment.
      Do not impute or flag NaN in these columns as errors.
    """
    return _read_csv(
        path      = PATHS.raw_calendar,
        dataset   = "calendar",
        dtype     = _CALENDAR_DTYPES,
        date_cols = ["date"],
    )


def load_inventory() -> pd.DataFrame:
    """
    Load data/raw/inventory_snapshots.csv.

    Returns
    -------
    pd.DataFrame
        Columns: Snapshot_Date (datetime64[ns]), SKU (str),
                 Current_Stock (int64), On_Order (int64),
                 Lead_Time_Days (int64), Safety_Stock (int64),
                 Reorder_Point (int64), Inventory_Value (float64).

    Notes
    -----
    CRITICAL AUDIT FINDING — 150 ORPHAN SKUs:
        inventory_snapshots contains 200 unique SKUs (SKU001–SKU200).
        Only SKU001–SKU050 exist in sku_master.
        SKU051–SKU200 are ORPHAN SKUs with zero sales records.
        Per engineering policy (orphan_sku_treatment = quarantine):
          - All 4,800 rows are loaded and returned as-is.
          - Orphan rows are NOT deleted, filtered, or modified here.
          - Orphan detection and quarantine tagging occur in preprocessing.
          - The validation layer reports orphans as a WARNING, not a FAIL.

    INVENTORY_VALUE NOTE:
        Inventory_Value uses an independent per-unit reference price that does
        not match Cost_Price or Selling_Price in sku_master (Phase 0b finding).
        Per policy (inventory_valuation_basis = unresolved), the column is
        preserved without reinterpretation.
    """
    return _read_csv(
        path      = PATHS.raw_inventory,
        dataset   = "inventory_snapshots",
        dtype     = _INVENTORY_DTYPES,
        date_cols = ["Snapshot_Date"],
    )


def load_all_data() -> dict[str, pd.DataFrame]:
    """
    Load all four raw datasets and return them in a named dictionary.

    Returns
    -------
    dict with keys:
        'sales'     -> sales_daily DataFrame
        'sku'       -> sku_master DataFrame
        'calendar'  -> calendar DataFrame
        'inventory' -> inventory_snapshots DataFrame

    Each value is a deep copy of the raw data — callers cannot mutate the source.
    All four files must be present; raises IngestionError for any missing file.
    """
    logger.info("[data_ingestion] Loading all four raw datasets…")
    data = {
        "sales":     load_sales(),
        "sku":       load_sku_master(),
        "calendar":  load_calendar(),
        "inventory": load_inventory(),
    }
    logger.info(
        "[data_ingestion] All datasets loaded. "
        "sales=%d  sku=%d  calendar=%d  inventory=%d",
        len(data["sales"]),
        len(data["sku"]),
        len(data["calendar"]),
        len(data["inventory"]),
    )
    return data


# ---------------------------------------------------------------------------
# Backward-compatible aliases (match existing pipeline.py _stage_ingest call)
# ---------------------------------------------------------------------------
load_sales_daily        = load_sales
load_sku_master_data    = load_sku_master       # avoids name clash with function
load_calendar_data      = load_calendar
load_inventory_snapshots = load_inventory

def load_all() -> dict[str, pd.DataFrame]:
    """Alias for load_all_data() — keeps pipeline.py _stage_ingest working."""
    return load_all_data()
