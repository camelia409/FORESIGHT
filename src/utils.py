"""
utils.py — Shared Utility Functions
====================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Provide reusable helper functions used across multiple modules.
- Keep this module free of business logic and project-specific decisions.
- Include: logging setup, date arithmetic, metric computation, file I/O
  helpers, serialization utilities, and reproducibility tools.

Functions (to be expanded progressively)
-----------------------------------------
    setup_logging(level, log_dir)       : Configure project-wide logging.
    set_seed(seed)                      : Set all random seeds for reproducibility.
    wape(y_true, y_pred)               : Compute WAPE metric.
    mae(y_true, y_pred)                : Compute MAE metric.
    rmse(y_true, y_pred)               : Compute RMSE metric.
    iso_week_start(date)               : Monday of the ISO week containing date.
    save_parquet(df, path)             : Write DataFrame to parquet with logging.
    load_parquet(path)                 : Read parquet with logging.
    save_json(obj, path)               : Write dict to JSON.
    load_json(path)                    : Read JSON to dict.
    timer(label)                       : Context manager to time code blocks.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    level:   int | str = logging.INFO,
    log_dir: Path | None = None,
    name:    str = "foresight",
) -> logging.Logger:
    """
    Configure project-wide logging with a console handler and optional file handler.

    Parameters
    ----------
    level   : logging level (e.g., logging.INFO or "DEBUG")
    log_dir : If provided, also write logs to log_dir/foresight_<timestamp>.log
    name    : Logger name

    Returns
    -------
    logging.Logger
    """
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger(name)
    root_logger.setLevel(level)

    if not root_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root_logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts  = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        fh  = logging.FileHandler(log_dir / f"{name}_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root_logger.addHandler(fh)

    return root_logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and (if available) PyTorch / TF.

    Call this once at the start of every notebook and script.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

        WAPE = sum(|y_true - y_pred|) / sum(y_true) × 100  [%]

    Parameters
    ----------
    y_true, y_pred : array-like of equal length.

    Returns
    -------
    float — WAPE in percent.  Returns NaN if sum(y_true) == 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = y_true.sum()
    if denom == 0:
        logger.warning("WAPE denominator is zero — returning NaN.")
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom * 100)


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def iso_week_start(date: pd.Timestamp | str) -> pd.Timestamp:
    """
    Return the Monday (start) of the ISO week containing the given date.

    Used to align all weekly aggregations to a consistent anchor.
    """
    ts = pd.Timestamp(date)
    return ts - pd.Timedelta(days=ts.dayofweek)  # Monday = 0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write a DataFrame to parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)
    logger.info("Saved parquet: %s  shape=%s", path, df.shape)


def load_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a parquet file with logging."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, **kwargs)
    logger.info("Loaded parquet: %s  shape=%s", path, df.shape)
    return df


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    """Serialise a JSON-serialisable object to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)
    logger.info("Saved JSON: %s", path)


def load_json(path: Path) -> Any:
    """Load a JSON file from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = "Block") -> Generator[None, None, None]:
    """
    Context manager to log the elapsed time of a code block.

    Usage
    -----
        with timer("feature engineering"):
            fe.fit_transform(df)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("[timer] %s — %.3f seconds", label, elapsed)


# ---------------------------------------------------------------------------
# Raw-file integrity helpers
# ---------------------------------------------------------------------------

import hashlib


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute the SHA-256 digest of a file without loading it fully into memory.

    Parameters
    ----------
    path       : Absolute or relative path to the file.
    chunk_size : Read chunk size in bytes (default 1 MiB).

    Returns
    -------
    str — 64-character lowercase hex digest.

    Raises
    ------
    FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[utils.file_sha256] File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    digest = h.hexdigest()
    logger.debug("[file_sha256] %s  -> %s", path.name, digest[:16] + "…")
    return digest


def hash_raw_files(raw_dir: Path) -> dict[str, str]:
    """
    Compute SHA-256 hashes for all four FORESIGHT raw CSV files.

    Parameters
    ----------
    raw_dir : Path to the data/raw/ directory.

    Returns
    -------
    dict mapping filename -> sha256 hex digest.
    All four files must be present; raises FileNotFoundError otherwise.
    """
    raw_dir = Path(raw_dir)
    file_names = [
        "sales_daily.csv",
        "sku_master.csv",
        "calendar.csv",
        "inventory_snapshots.csv",
    ]
    hashes: dict[str, str] = {}
    for fname in file_names:
        fpath = raw_dir / fname
        hashes[fname] = file_sha256(fpath)
        logger.info("[hash_raw_files] %-30s  sha256=%s", fname, hashes[fname][:16] + "…")
    return hashes


def assert_hashes_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    """
    Assert that two hash dictionaries are identical.
    Raises RuntimeError listing any changed files.

    Parameters
    ----------
    before : Hash dict captured before a pipeline operation.
    after  : Hash dict captured after a pipeline operation.
    """
    changed = [k for k in before if before[k] != after.get(k)]
    if changed:
        lines = "\n".join(
            f"  {k}: {before[k][:16]}… -> {after.get(k, 'MISSING')[:16]}…"
            for k in changed
        )
        raise RuntimeError(
            f"[utils] RAW FILE INTEGRITY VIOLATION — {len(changed)} file(s) changed:\n"
            f"{lines}\n"
            "The pipeline must never write to data/raw/."
        )
    logger.info("[assert_hashes_unchanged] All %d raw files verified unchanged.", len(before))
