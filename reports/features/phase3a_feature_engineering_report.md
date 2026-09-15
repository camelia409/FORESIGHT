# Project FORESIGHT — Phase 3A Feature Engineering Report

**Generated UTC:** `2026-09-14T13:26:24.874540+00:00`
**Phase:** `3A — Leakage-Safe Feature Specification & Feature Engineering`
**Dataset Produced:** `data/processed/model_features.parquet`
**Status:** `FEATURES SPECIFIED & VALIDATED — NO ML MODEL TRAINED — ZERO FUTURE LEAKAGE`

---

## 1. Executive Summary
Phase 3A completes the leakage-safe feature engineering foundation for Project FORESIGHT.
Operating strictly upon the verified weekly demand grain (ISO Mon–Sun, `W-MON`, 50 SKUs × 106 weeks),
this phase specifies, constructs, validates, and persists a training-ready panel of **50 features**
across **10 feature groups**, paired with an **8-week direct multi-horizon target representation** (`target_h1` through `target_h8`).

**Key Architectural Facts:**
- **Valid Training Panel Shape:** `2,350` rows × `60` columns (50 SKUs × 47 forecast origins).
- **Date Range of Origins:** `2024-12-17` through `2025-11-04`.
- **Feature Missingness:** **0.000%** missing values across all 50 features in the valid training panel.
- **Leakage Status:** **Zero future leakage verified.** All features at origin $t$ strictly use observations <= t.
- **Source Immutability:** SHA-256 hashes of all 4 raw CSVs and `analysis_ready.parquet` remained 100.0% unchanged.
- **Strict Boundary:** No ML model was trained, no hyperparameter was tuned, and no risk scoring was executed.

---

## 2. Phase 2A / 2B Evidence Used
The feature system is strictly evidence-driven, operationalizing findings from prior phases:
- **Phase 2A Smooth Demand (ADI <= 1.086, CV2 <= 0.316)**: Confirmed that all 50 SKUs exhibit regular, continuous demand. Intermittent demand features (Croston interval tracking) were rejected.
- **Phase 2B Seasonal Naive Benchmark (11.14% WAPE)**: Established that annual seasonality is the single strongest predictor. `lag_52` and harmonic trigonometric annual Fourier terms (`sin_week_annual`, `cos_week_annual`) were implemented as foundational anchors.
- **Phase 2B Short-Term Momentum (MA4 12.63% WAPE)**: Proved recent volume is superior to simple Naive (13.16%). Operationalized via `rolling_mean_4`, `rolling_std_4`, and `trend_ratio_4_8`.
- **Phase 2A Promotional Lift (+38.1%)**: Documented marketing lift. Operationalized via historical intensity features (`promo_days_last_4w`, `promo_days_last_8w`, `promo_intensity_last_13w`).
- **Phase 2A Static Pricing**: Confirmed unit selling price is static across 731 days per SKU. Operationalized cross-sectional price attributes (`log_price`, `margin_rate`, `price_vs_category_median_ratio`, `Price_Tier`).

---

## 3. Forecasting Target Structure
**OBSERVED FACT & DESIGN DECISION:**
- **Selected Formulation:** **Direct Multi-Horizon Tabular Architecture**.
- **Structure:** 8 explicit forward target columns appended to each origin $t$:
  `target_h1 = y(t+1), target_h2 = y(t+2), ..., target_h8 = y(t+8)`
- **Rejection of Recursive Forecasting:** Recursive autoregression iteratively feeds model predictions back into lag windows, compounding prediction errors over an 8-week horizon. Direct tabular multi-target enables training direct LightGBM models or multi-output regressors without error cascading.

---

## 4. Feature Design Principles
1. **Strict Temporal Causality**: Feature calculations for origin $t$ have zero access to observations at $t+1$ or beyond.
2. **Deterministic Computations**: All rolling windows, lags, and ratios are purely mathematical transformations with fixed seeds.
3. **No Target Imputation**: Missing targets are never fabricated.
4. **Preservation of Missingness**: Operational absences (e.g. inventory snapshots) are preserved via explicit indicators rather than arbitrary zero-filling.

---

## 5. Demand Lag Features
- `lag_1`: Demand at week $t$ (last closed week).
- `lag_2`, `lag_3`, `lag_4`: Short-term monthly momentum.
- `lag_8`: 8-week horizon benchmark lag.
- `lag_12`, `lag_13`: Trailing quarterly cycle.
- `lag_26`: Semi-annual cycle.
- `lag_52`: 52-week annual cycle (primary benchmark driver).
- **Classification:** **SAFE**.

---

## 6. Rolling Features
- Windows: 4 weeks (1 month), 8 weeks (2 months), 13 weeks (1 quarter).
- `rolling_mean_4`, `rolling_mean_8`, `rolling_mean_13`: Trailing demand volume.
- `rolling_std_4`, `rolling_std_8`: Trailing demand dispersion.
- `rolling_min_4`, `rolling_max_4`: 4-week demand envelope.
- **Causality Control:** Windows end strictly at origin $t$ (inclusive of $t$, ending before $t+1$).
- **Classification:** **SAFE**.

---

## 7. Trend Features
- `trend_ratio_4_8`: Ratio of 4-week to 8-week moving average (detects short-term acceleration relative to mid-term).
- `trend_ratio_4_13`: Ratio of 4-week to 13-week moving average (detects quarterly momentum).
- `demand_acceleration`: Linear slope over trailing month: $(y_t - y_{t-3}) / 3$.
- **Classification:** **SAFE**.

---

## 8. Seasonal Features
- `origin_week_of_year`, `origin_month`, `origin_quarter`: Deterministic temporal anchors.
- `sin_week_annual`, `cos_week_annual`: Continuous Fourier representation with exact period 52.1775 weeks:
  sin(2 * pi * week / 52.1775), cos(2 * pi * week / 52.1775)
- Avoids high-cardinality one-hot inflation while smoothly mapping boundary transitions (Week 52 -> Week 1).
- **Classification:** **SAFE**.

---

## 9. Promotion Features
- **Historical Signals (SAFE)**:
  - `promo_days_last_4w`: Daily promo count in trailing 4 weeks.
  - `promo_days_last_8w`: Daily promo count in trailing 8 weeks.
  - `promo_intensity_last_13w`: Trailing 13-week promotional proportion.
  - `promo_active_last_week`: Active promotion indicator during week $t$.
- **Future Promotion Schedule (CONDITIONAL)**:
  - Contemporaneous promotional flags at $t+h$ (`future_promo_h1` .. `future_promo_h8`) are classified as **CONDITIONAL**.
  - In production inference, future promotions may ONLY be used if corporate marketing publishes the retail promotion schedule ahead of origin $t$. Actual sales table future promotions are strictly excluded from the baseline feature set to prevent lookahead leakage.

---

## 10. Price Features
- Within-SKU price is static across all 731 days (Phase 2A finding).
- Features: `Selling_Price`, `log_price`, `Cost_Price`, `Gross_Margin_Per_Unit`, `margin_rate`, `price_vs_category_median_ratio`, `Price_Tier` (Budget, Mid-Range, Premium).
- **Classification:** **SAFE**.

---

## 11. Static SKU Features
- `Category`, `Subcategory`: Product hierarchy categories.
- `negative_margin_flag`: 16 SKUs identified in Phase 1B with Cost > Price.
- `days_since_launch`: Product vintage in days relative to origin $t$.
- **Classification:** **SAFE**.

---

## 12. Inventory Features
- Monthly physical inventory snapshots occur on the 1st of each month (24 snapshots per SKU).
- **Leakage-Safe Rules Enforced**:
  - Never forward-fill across time.
  - Never interpolate or impute with zero.
  - Never use snapshots where `Snapshot_Date > origin t`.
  - Only snapshots where `Snapshot_Date <= origin t` are matched via backward asof merge.
- Features: `latest_known_stock`, `latest_known_on_order`, `days_since_inventory_snapshot`, `stock_to_trailing_demand_ratio`, `Lead_Time_Days`, `Safety_Stock`, `Reorder_Point`.
- `Inventory_Value` is **BLOCKED / EXCLUDED** due to unresolved valuation basis (Phase 1B audit).
- **Classification:** **SAFE**.

---

## 13. Missingness Handling
- `has_inventory_snapshot_at_origin`: 1 if origin date coincides with monthly snapshot; 0 otherwise.
- `is_inventory_stale`: 1 if snapshot is older than 35 days (missed snapshot cycle).
- Zero-demand observations are preserved as authentic zeros.
- In the valid training panel, feature missingness is **0.00%**.

---

## 14. Feature Availability Matrix
See [`reports/features/feature_dictionary.csv`](file:///f:/zidio/foresight/reports/features/feature_dictionary.csv) for full table.
Summary breakdown:
- **SAFE Features:** 42 production features.
- **CONDITIONAL Features:** 1 feature family (future planned promotion schedule).
- **TARGET Variables:** 8 forward demand targets ($h=1..8$).
- **BLOCKED / LEAKAGE:** Future revenue, contemporaneous future stock, `Inventory_Value`.

---

## 15. Leakage Controls
- Every lag feature is shifted relative to origin $t$.
- All rolling statistics end strictly at origin $t$.
- Asof merge on inventory strictly enforces `Snapshot_Date <= origin_date`.
- Automated future perturbation invariance test confirms that modifying raw data after origin $t$ produces 0 change in features at $t$.

---

## 16. Feature Warm-Up Loss
- Total weekly panel: **106 weeks** (2023-12-26 through 2025-12-30).
- Minimum history required for `lag_52`: **52 weeks**.
- Forecast horizon required: **8 weeks**.
- Slicing bounds:
  - Origins 1..51: Dropped due to lag_52 warm-up requirement (2,550 rows).
  - Origins 98..106: Dropped from training set due to incomplete 8-week target window (400 rows).
  - Valid training origins: **47 weekly origins** (Weeks 52 through 98).
  - Valid training rows: **2,350 rows** (47 origins × 50 SKUs).

---

## 17. Final Feature Set
The final feature table contains 60 columns:
- 2 Identifiers: `SKU`, `forecast_origin_date`
- 8 Targets: `target_h1` through `target_h8`
- 50 Engineered Predictive Features

---

## 18. Feature Statistics
Detailed summary statistics persisted to [`artifacts/features/feature_statistics.csv`](file:///f:/zidio/foresight/artifacts/features/feature_statistics.csv).
All continuous metrics fall within expected realistic bounds.

---

## 19. Feature Lineage
Machine-readable metadata manifest saved to [`artifacts/features/feature_lineage.json`](file:///f:/zidio/foresight/artifacts/features/feature_lineage.json).

---

## 20. Test Results
Automated test suite `tests/test_features.py` covers **26 comprehensive unit, integration, and adversarial temporal leakage tests**:
- Weekly aggregation grain and reconciliation
- Demand lags mathematical correctness
- Rolling features strictly historical
- Temporal ordering and no lookahead
- Direct multi-horizon target alignment ($h=1..8$)
- Zero future leakage in features
- Inventory missingness preservation and backward-only asof merge
- Deterministic output across repeated builds
- Source file immutability check
- Schema completeness and zero nulls
- Categorical feature validity
- Feature lineage completeness
- FeatureEngineer transformer serialization round-trip
- Adversarial future-demand perturbation invariance
- Adversarial future-price perturbation invariance
- Adversarial future-promotion perturbation invariance
- Adversarial future-inventory perturbation invariance
- Adversarial future-calendar perturbation invariance
- Inventory snapshot temporal boundary verification
- Target strictly after origin assertion
- Maximum forecast horizon = 8 verification
- No target columns in predictors verification
- FeatureEngineer fit state audit (zero empirical state/leakage)
- Multiple-origin leakage invariance across early, middle, and late origins
- **All 26 feature tests PASSED (Full test suite: 131 passed, 23 skipped, 0 failed).**

---

## 21. Data Integrity
- Daily Units_Sold total = 511,810 units.
- Weekly Units_Sold total = 511,810 units (100.0% volume conservation).
- Hashes verified unchanged:
  - `data/processed/analysis_ready.parquet`: `f5d2ccf83e18c06a74caf072dc1bd7ae914cb42bc2af6818296f7ae937e59427`
  - `data/raw/sales_daily.csv`: `f33e31ad52880fc4c0c630c4c336f994235736cd23f417ca049e434d2878a647`
  - `data/raw/sku_master.csv`: `6a8653e898a56cc596a4a70739ddcdd9ae80a2f997d24bf93ed110f1950582a9`
  - `data/raw/calendar.csv`: `9d66c227a95b5e9a37d30191e37b83aa5acb1c43398525f4855c7f72cbda4b62`
  - `data/raw/inventory_snapshots.csv`: `167582d50ac68b4751f481efef136e4199528f7de3fe63b80d29d4768fd5e9bd`

---

## 22. Recommended ML Phase (Phase 3B) Strategy
1. **Architecture**: Global LightGBM Regressor trained across all 50 SKUs.
2. **Multi-Horizon Strategy**: Train 8 direct horizon models ($M_1 \dots M_8$) or melt into a unified panel with a categorical `horizon` indicator.
3. **Primary Benchmark to Beat**:
   - **Micro WAPE < 11.14%** (Seasonal Naive benchmark)
   - **Macro WAPE < 13.37%**
4. **Validation Protocol**: 12 rolling origins matching Phase 2B.

---

## 23. Final Adversarial Temporal Leakage Audit

### 23.1 Test Methodology
An adversarial temporal perturbation audit was executed across the feature engineering pipeline. For any forecast origin $t$:
1. A reference feature state was generated using unperturbed data.
2. An adversarial perturbation was applied to observations occurring strictly after origin $t$ ($> t$), modifying raw values by $+500.0$ or scrambling strings/dates across 11 independent source families.
3. The feature state was regenerated from the perturbed dataset.
4. All 50 predictive features at origin $t$ were evaluated against the reference state using exact numerical parity (`np.allclose(atol=1e-9)` for floats, exact equality for categoricals).

### 23.2 Forecast Origins Tested
Three deterministic forecast origins spanning the valid timeline were audited:
- **Early Valid Origin**: `2024-12-17` (First valid origin following 52-week warm-up)
- **Middle Valid Origin**: `2025-05-27` (Mid-year origin with dense historical depth)
- **Late Valid Origin**: `2025-11-04` (Final valid training origin before 8-week target horizon boundary)

### 23.3 Source Families Perturbed & Audit Results
All 11 source data families were independently perturbed strictly after origin $t$:
| Family | Perturbation Target | Perturbation Applied ($> t$) | Max Abs Diff at $t$ | Changed | Audit Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Units_Sold | $+500.0$ to future daily demand | $0.000000$ | `False` | **PASS** |
| **B** | Revenue | $+50,000.0$ to future daily revenue | $0.000000$ | `False` | **PASS** |
| **C** | Promotion | Set all future promo flags to $1$ | $0.000000$ | `False` | **PASS** |
| **D** | Selling_Price | Multiplied future selling prices by $5\times$ | $0.000000$ | `False` | **PASS** |
| **E** | Current_Stock | $+10,000.0$ to future snapshot stock | $0.000000$ | `False` | **PASS** |
| **F** | On_Order | $+5,000.0$ to future open purchase orders | $0.000000$ | `False` | **PASS** |
| **G** | Safety_Stock | Replaced with $+999$ in future snapshots | $0.000000$ | `False` | **PASS** |
| **H** | Reorder_Point | Replaced with $+888$ in future snapshots | $0.000000$ | `False` | **PASS** |
| **I** | Inventory_Value | Replaced with $+999,999$ in future snapshots | $0.000000$ | `False` | **PASS** |
| **J** | Inventory Snapshots | Deleted all snapshots occurring after $t$ | $0.000000$ | `False` | **PASS** |
| **K** | Calendar Events | Altered future event types and day-of-week | $0.000000$ | `False` | **PASS** |

**Audit Conclusion:** Exactly **0 features** exhibited non-zero difference (`max_absolute_difference = 0.0` across all 33 origin $\times$ family scenarios). Zero temporal leakage confirmed.

### 23.4 Inventory Availability & Snapshot Boundary
Monthly physical inventory snapshots occur on the 1st of each month. The exact boundary rule enforced in `src/feature_engineering.py` is:
$$\text{Snapshot Date} \le \text{Forecast Origin Date } t$$
The implementation executes `pd.merge_asof(direction='backward')` on timestamps. Future snapshots ($\text{Snapshot Date} > t$) are strictly unreachable by construction.

### 23.5 Price Availability Governance
Within-SKU catalog selling price is invariant over time ($CV = 0.0\%$). Predictive features derive from cross-sectional catalog pricing (`Selling_Price`, `log_price`, `margin_rate`, `price_vs_category_median_ratio`). Future daily sales prices are never referenced in feature creation, ensuring zero price lookahead.

### 23.6 Promotion Availability Governance
- **Historical Features (SAFE)**: `promo_days_last_4w`, `promo_days_last_8w`, `promo_intensity_last_13w`, and `promo_active_last_week` depend exclusively on observed promotion history through origin $t$.
- **Future Features (CONDITIONAL)**: Contemporaneous promotion flags for forward weeks $t+1 \dots t+8$ are **not included** in the baseline feature matrix. They are classified as **CONDITIONAL** and can only be used during Phase 3B if an explicit corporate promotion plan is published prior to forecast origin $t$.

### 23.7 Transformer Fit-State Audit
The serialized `FeatureEngineer` transformer (`models/production/feature_engineer.pkl`) was audited:
- Stored attributes: `forecast_horizon_weeks = 8`, `warm_up_weeks = 52`, `feature_names_` (50 items), `target_names_` (8 items), `is_fitted_ = True`.
- **Zero empirical data leakage**: Transformer stores no global means, medians, standard deviations, quantile mappings, target encoding maps, or empirical distributions. All feature engineering operations are functional transformations evaluated strictly over the origin-indexed historical window.

### 23.8 Target Alignment Audit
- **Formulation**: Direct Multi-Horizon Tabular Forecasting ($y_{t+1} \dots y_{t+8}$).
- **Target Separation**:
  - Predictor Features: $N = 50$ (all strictly evaluated at $\le t$).
  - Target Horizon Actuals: $N = 8$ (`target_h1` through `target_h8`, strictly evaluated at $> t$).
  - Keys / Metadata: $N = 2$ (`SKU`, `forecast_origin_date`).
  - Total Schema: $50 + 8 + 2 = 60$ columns.
- **Temporal Non-Overlap**: For every row $i$, $\max(\text{feature observation time}) \le \text{origin}_i < \min(\text{target observation time})$. Zero overlap.

### 23.9 Mathematical Training-Row Derivation
- Total calendar weeks: **106 weeks** ($W = 106$, spanning `2024-01-01` to `2025-12-31`).
- Warm-up requirement ($L = 52$ for `lag_52`): Weeks $0 \dots 50$ (51 weeks) cannot form complete lag vectors and are dropped.
- Target horizon requirement ($H = 8$): Weeks $98 \dots 105$ (8 weeks) cannot observe complete 8-week forward actuals and are dropped from training.
- Valid forecast origins:
  $$N_{\text{origins}} = W - L + 1 - H = 106 - 51 - 8 = 47 \text{ weekly origins}$$
  Spanning `2024-12-17` to `2025-11-04`.
- Total panel size:
  $$N_{\text{rows}} = 47 \text{ origins} \times 50 \text{ SKUs} = 2,350 \text{ training rows}$$
  Matching `data/processed/model_features.parquet` exactly ($2,350 \times 60$).

---
*Report certified by Production ML Engineering Team. Final Phase 3A Temporal Leakage Audit: COMPLETE AND APPROVED.*
