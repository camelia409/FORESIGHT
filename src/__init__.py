"""
foresight.src
=============
Source package for Project FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform.

Sub-modules
-----------
config              : Central configuration, paths, and constants.
data_ingestion      : Controlled raw-data loading functions.
validation          : Schema, key, and data-quality validation.
preprocessing       : Cleaning and dataset integration (post business sign-off).
feature_engineering : Leakage-safe time-series feature construction.
baseline            : Seasonal-naive forecasting baseline.
models              : Model definitions, training helpers, and serialization.
backtesting         : Rolling-origin time-series cross-validation.
forecasting         : Forecast generation and inference-time preparation.
risk_engine         : Stockout / overstock scoring and business recommendations.
pipeline            : End-to-end orchestration of the full pipeline.
utils               : Shared utility functions used across modules.
"""
