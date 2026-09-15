# FORESIGHT Production Forecasting Model Card

## 1. Model Purpose
The FORESIGHT Production Forecaster provides 8-week horizon-specific demand forecasts for 50 commercial SKUs to drive automated inventory optimization, safety stock calculation, and stockout prevention.

## 2. Forecast Target
Weekly aggregated unit sales demand across horizons:
$$y_{t+1}, y_{t+2}, \dots, y_{t+8}$$
anchored on weekly Monday forecast origins (`W-MON`).

## 3. Forecast Horizons
Direct multi-horizon forecasting for 8 weeks forward ($h=1 \dots 8$).

## 4. Training Data
- Panel: 2,350 observations (50 SKUs × 47 weekly origins).
- Historical Span: 2024-01-01 through 2025-12-31.
- Features: Exactly 50 approved SAFE features from Phase 3A Feature Engineering.

## 5. Model Architecture: Horizon-Segmented Hybrid
The production architecture is a **validation-justified Horizon-Segmented Hybrid**:
- **h=1**: Tuned Random Forest (`max_depth=8`, `min_samples_split=8`, `min_samples_leaf=3`, `max_features=0.7`)
- **h=2**: Tuned XGBoost (`objective='reg:absoluteerror'`, `learning_rate=0.04`, `max_depth=4`, `min_child_weight=2`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.5`, `reg_lambda=1.0`)
- **h=3..8**: Seasonal Naive 52-week baseline ($y_{t+h-52}$)

### Architecture Selection Rationale
In expanding-window temporal validation, Machine Learning models achieved superior precision on short horizons ($h=1$: 8.88% vs. 9.24% SN; $h=2$: 9.08% vs. 9.44% SN), while Seasonal Naive 52w outperformed all ML candidates on longer horizons ($h \ge 3$) due to annual retail demand seasonality persistence.

## 6. Final Performance vs Benchmark

| Metric | Selected Hybrid | Seasonal Naive 52w | Benchmark Difference |
| :--- | :---: | :---: | :---: |
| **Micro WAPE** | **10.45%** | **10.67%** (11.14% 12-orig) | **-0.22 pp (-2.1% relative)** |
| **Macro WAPE** | **12.63%** | **13.00%** (13.37% 12-orig) | **-0.37 pp (-2.8% relative)** |
| **MAE (units)**| **10.31** | **10.51** | **-0.20 units** |
| **RMSE (units)**| **13.48** | **13.73** | **-0.25 units** |

## 7. Operational & Latency Characteristics
- Feature Transformation: ~0.15 ms
- Model Inference: ~0.85 ms
- End-to-End Latency: < 1.5 ms per batch (50 SKUs × 8 horizons)
- Negative Predictions: Strictly 0 (bounded by non-negativity post-processing)

## 8. Leakage Governance
- Zero temporal leakage: Adversarial perturbation test on future observations produced 0.000000000000 prediction difference.
- 50 SAFE features strictly isolated from future demand and metadata identifiers.

## 9. Production Approval Status
**APPROVED FOR PRODUCTION PROMOTION**
