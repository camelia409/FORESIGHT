"""
config.py — Central Configuration & Path Registry
==================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Load configs/config.yaml at import time.
- Resolve all project-relative paths to absolute Path objects safely.
- Expose CFG (configuration constants) and PATHS (path registry) to all modules.
- Fail clearly if the config file is missing or structurally invalid.
- Never hard-code OS-specific absolute paths.
- Keep configuration logic separate from business logic.

Usage
-----
    from src.config import CFG, PATHS, PROJECT_ROOT

    raw_sales = pd.read_csv(PATHS.raw_sales)
    horizon   = CFG.forecast_horizon_weeks
    policy    = CFG.orphan_sku_treatment      # "quarantine"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root — resolved relative to this file's location
# Never hard-code Windows paths; always anchor to this file.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]   # …/foresight/
CONFIG_FILE:  Path = PROJECT_ROOT / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# Internal YAML loader
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    """Load a YAML file. Raises FileNotFoundError with a clear message."""
    if not path.exists():
        raise FileNotFoundError(
            f"[config] Configuration file not found: {path}\n"
            f"  Expected location: foresight/configs/config.yaml\n"
            f"  Ensure the project root is correct."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"[config] config.yaml is empty or malformed: {path}"
        )
    return raw


def _require(cfg: dict, key: str) -> Any:
    """Return cfg[key]; raise KeyError with a descriptive message if absent."""
    if key not in cfg:
        raise KeyError(
            f"[config] Required key '{key}' is missing from config.yaml.\n"
            f"  Add it to: foresight/configs/config.yaml"
        )
    return cfg[key]


def _resolve(relative: str) -> Path:
    """Resolve a project-relative path string to an absolute Path."""
    return PROJECT_ROOT / relative


# ---------------------------------------------------------------------------
# Load raw YAML once at module import
# ---------------------------------------------------------------------------
_raw: dict = _load_yaml(CONFIG_FILE)


# ---------------------------------------------------------------------------
# Path Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Paths:
    """
    Centralised, immutable path registry.
    All paths are absolute Path objects derived from PROJECT_ROOT.
    No path is hard-coded with OS-specific separators.
    """
    # ── Raw (immutable) data layer ──────────────────────────────────────────
    raw_dir:            Path
    raw_sales:          Path
    raw_sku_master:     Path
    raw_calendar:       Path
    raw_inventory:      Path

    # ── Interim / processed data layers ────────────────────────────────────
    interim_dir:        Path
    processed_dir:      Path

    # ── Report directories ──────────────────────────────────────────────────
    reports_data_quality_dir: Path
    reports_eda_dir:          Path
    reports_model_dir:        Path
    reports_executive_dir:    Path

    # ── Artefact directories ────────────────────────────────────────────────
    metrics_dir:        Path
    figures_dir:        Path
    predictions_dir:    Path

    # ── Model directories ───────────────────────────────────────────────────
    models_dir:         Path
    baseline_dir:       Path
    production_dir:     Path


def _build_paths(cfg: dict) -> _Paths:
    """Construct the path registry from config.yaml, resolving all paths."""
    data_cfg      = cfg.get("data", {})
    reports_cfg   = cfg.get("reports", {})
    artifacts_cfg = cfg.get("artifacts", {})
    models_cfg    = cfg.get("models", {})

    raw_dir = _resolve(data_cfg.get("raw_dir", "data/raw"))

    return _Paths(
        raw_dir                   = raw_dir,
        raw_sales                 = raw_dir / data_cfg.get("sales_daily",         "sales_daily.csv"),
        raw_sku_master            = raw_dir / data_cfg.get("sku_master",          "sku_master.csv"),
        raw_calendar              = raw_dir / data_cfg.get("calendar",            "calendar.csv"),
        raw_inventory             = raw_dir / data_cfg.get("inventory_snapshots", "inventory_snapshots.csv"),

        interim_dir               = _resolve(data_cfg.get("interim_dir",    "data/interim")),
        processed_dir             = _resolve(data_cfg.get("processed_dir",  "data/processed")),

        reports_data_quality_dir  = _resolve(reports_cfg.get("data_quality_dir", "reports/data_quality")),
        reports_eda_dir           = _resolve(reports_cfg.get("eda_dir",          "reports/eda")),
        reports_model_dir         = _resolve(reports_cfg.get("model_dir",        "reports/model")),
        reports_executive_dir     = _resolve(reports_cfg.get("executive_dir",    "reports/executive")),

        metrics_dir               = _resolve(artifacts_cfg.get("metrics_dir",     "artifacts/metrics")),
        figures_dir               = _resolve(artifacts_cfg.get("figures_dir",     "artifacts/figures")),
        predictions_dir           = _resolve(artifacts_cfg.get("predictions_dir", "artifacts/predictions")),

        models_dir                = _resolve(models_cfg.get("dir",            "models")),
        baseline_dir              = _resolve(models_cfg.get("baseline_dir",   "models/baseline")),
        production_dir            = _resolve(models_cfg.get("production_dir", "models/production")),
    )


PATHS: _Paths = _build_paths(_raw)


# ---------------------------------------------------------------------------
# Project Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Config:
    """
    Typed, immutable project-level configuration loaded from config.yaml.

    Design rules
    ------------
    - Every attribute has an explicit type annotation.
    - Unresolved business-policy values are loaded as-is (strings) and must be
      checked by the relevant pipeline stage before use.
    - No business logic lives here — this is a pure value object.
    """
    # ── Identity ────────────────────────────────────────────────────────────
    project_name:     str
    project_version:  str

    # ── Forecasting ─────────────────────────────────────────────────────────
    forecast_horizon_weeks: int
    forecast_frequency:     str
    primary_metric:         str

    # ── Backtesting ─────────────────────────────────────────────────────────
    n_backtest_folds:  int
    min_train_weeks:   int
    backtest_step_weeks: int

    # ── Reproducibility ─────────────────────────────────────────────────────
    random_seed: int

    # ── Data quality thresholds ─────────────────────────────────────────────
    revenue_price_tol: float

    # ── Engineering policy decisions (from Phase 0b investigation) ──────────
    # These are explicit string values or "unresolved" — never None after Phase 1A.
    orphan_sku_treatment:      str   # "quarantine" | "exclude" | "backfill"
    negative_margin_treatment: str   # "preserve_and_flag" | "correct"
    inventory_valuation_basis: str   # "cost" | "selling" | "unresolved"
    panel_gap_treatment:       str   # "zero_fill" | "interpolate" | "flag_only"
    calendar_nan_treatment:    str   # "sentinel_none" | "keep_nan"
    forecasting_universe:      str   # "master_only" | "all"

    # ── Known audit facts (from Phase 0 audit — immutable constants) ────────
    # These are documented facts about the raw data, not configurable decisions.
    KNOWN_MASTER_SKU_COUNT:     int = field(default=50,   compare=False)
    KNOWN_TOTAL_SKU_COUNT:      int = field(default=200,  compare=False)
    KNOWN_ORPHAN_SKU_COUNT:     int = field(default=150,  compare=False)
    KNOWN_ORPHAN_INV_ROWS:      int = field(default=3600, compare=False)
    KNOWN_SALES_ROWS:           int = field(default=36550, compare=False)
    KNOWN_CALENDAR_DAYS:        int = field(default=731,  compare=False)
    KNOWN_INV_ROWS:             int = field(default=4800, compare=False)
    KNOWN_NEG_MARGIN_SKU_COUNT: int = field(default=16,   compare=False)

    def require_policy(self, key: str) -> str:
        """
        Return a policy value; raise RuntimeError if it is 'unresolved'.
        Use this in pipeline stages that CANNOT proceed without a confirmed value.
        """
        val = getattr(self, key, None)
        if val is None or val == "unresolved":
            raise RuntimeError(
                f"[config] Policy '{key}' is not resolved. "
                f"Set a confirmed value in configs/config.yaml before proceeding."
            )
        return val


def _build_config(cfg: dict) -> _Config:
    """Construct the typed config object from the raw YAML dictionary."""

    def _get(key: str, default: Any) -> Any:
        return cfg.get(key, default)

    return _Config(
        project_name             = _get("project_name",             "FORESIGHT"),
        project_version          = _get("project_version",          "0.2.0"),
        forecast_horizon_weeks   = int(_get("forecast_horizon_weeks",  8)),
        forecast_frequency       = _get("forecast_frequency",       "W-MON"),
        primary_metric           = _get("primary_metric",           "WAPE"),
        n_backtest_folds         = int(_get("n_backtest_folds",        12)),
        min_train_weeks          = int(_get("min_train_weeks",         52)),
        backtest_step_weeks      = int(_get("backtest_step_weeks",      1)),
        random_seed              = int(_get("random_seed",             42)),
        revenue_price_tol        = float(_get("revenue_price_tol",    0.01)),
        orphan_sku_treatment     = str(_get("orphan_sku_treatment",     "quarantine")),
        negative_margin_treatment= str(_get("negative_margin_treatment","preserve_and_flag")),
        inventory_valuation_basis= str(_get("inventory_valuation_basis","unresolved")),
        panel_gap_treatment      = str(_get("panel_gap_treatment",      "flag_only")),
        calendar_nan_treatment   = str(_get("calendar_nan_treatment",   "sentinel_none")),
        forecasting_universe     = str(_get("forecasting_universe",     "master_only")),
    )


CFG: _Config = _build_config(_raw)


# ---------------------------------------------------------------------------
# Module-level validation: catch obviously wrong configs at import time
# ---------------------------------------------------------------------------
def _validate_config(cfg: _Config) -> None:
    """Raise ValueError for obviously invalid configuration values."""
    if cfg.forecast_horizon_weeks < 1:
        raise ValueError("[config] forecast_horizon_weeks must be >= 1")
    if cfg.random_seed < 0:
        raise ValueError("[config] random_seed must be >= 0")
    if cfg.revenue_price_tol < 0:
        raise ValueError("[config] revenue_price_tol must be >= 0")
    if cfg.orphan_sku_treatment not in {"quarantine", "exclude", "backfill"}:
        raise ValueError(
            f"[config] orphan_sku_treatment='{cfg.orphan_sku_treatment}' is not valid. "
            "Use: 'quarantine' | 'exclude' | 'backfill'"
        )
    if cfg.negative_margin_treatment not in {"preserve_and_flag", "correct"}:
        raise ValueError(
            f"[config] negative_margin_treatment='{cfg.negative_margin_treatment}' is not valid. "
            "Use: 'preserve_and_flag' | 'correct'"
        )
    if cfg.inventory_valuation_basis not in {"cost", "selling", "unresolved"}:
        raise ValueError(
            f"[config] inventory_valuation_basis='{cfg.inventory_valuation_basis}' is not valid. "
            "Use: 'cost' | 'selling' | 'unresolved'"
        )


_validate_config(CFG)
logger.debug(
    "[config] Loaded. orphan=%s  neg_margin=%s  inv_val=%s  seed=%d",
    CFG.orphan_sku_treatment,
    CFG.negative_margin_treatment,
    CFG.inventory_valuation_basis,
    CFG.random_seed,
)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------
__all__ = ["CFG", "PATHS", "PROJECT_ROOT"]
