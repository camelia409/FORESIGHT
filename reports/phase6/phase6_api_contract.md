# PROJECT FORESIGHT — PHASE 6 REST API CONTRACT

**Document ID:** `FORESIGHT-API-P6-001`  
**Phase:** Phase 6 — Production Deployment  
**Status:** RATIFIED & DEPLOYED  
**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI) / `http://localhost:8000/redoc` (ReDoc)  

---

## 1. Overview & Architecture

The FORESIGHT Production API provides high-speed, programmatic access to demand forecasts, multi-horizon inventory risk metrics, and prioritized replenishment directives.

All endpoints adhere to:
1. **Decision #1 (Option 1D):** Zero monetary exposure figures (`excess_inventory_value` = `None`, `inventory_value_at_risk` = `None`, `capital_at_risk` = `None`).
2. **Decision #2 (Option 2A):** Policy $B_{LT}$ on-order arrival accounting based on verified lead times.
3. **Decision #3 (Option 3C):** $N=8$ weeks ratified overstock threshold (`overstock_threshold_status = "POLICY_RATIFIED"`).
4. **Scope:** 50 monitored production SKUs (`SKU001`–`SKU050`). The 150 orphan SKUs (`SKU051`–`SKU200`) return HTTP 404.

---

## 2. Complete Endpoint Directory

| Method | Path | Summary | Response Model |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Core service health check | `HealthResponse` |
| `GET` | `/version` | Service & model version info | `VersionResponse` |
| `GET` | `/` | Root service descriptor | `RootResponse` |
| `POST` | `/predict` | Production multi-horizon forecast | `PredictResponse` |
| `POST` | `/v1/forecast` | V1 forecast endpoint | `ForecastResponse` |
| `POST` | `/v1/risk` | V1 risk scoring endpoint | `RiskResponse` |
| `POST` | `/v1/recommendations` | V1 recommendations endpoint | `RecommendationResponse` |
| `GET` | `/api/status` | Comprehensive system telemetry | `SystemStatusResponse` |
| `GET` | `/api/inventory` | Fleet inventory status (50 SKUs) | `InventoryResponse` |
| `GET` | `/api/inventory/{sku}` | 360-degree SKU detail & forecast | `SkuDetailResponse` |
| `GET` | `/api/risk` | Fleet risk assessment records | `RiskResponse` |
| `GET` | `/api/recommendations` | Ratified operational actions | `RecommendationResponse` |
| `GET` | `/api/governance` | Ratified policy audit record | `GovernanceResponse` |

---

## 3. Key Endpoint Specifications

### 3.1. GET `/api/status`
Returns real-time operational status, model readiness, SKU counts, and governance policies.

**Response Example (200 OK):**
```json
{
  "status": "HEALTHY",
  "version": "1.0.0",
  "phase": "Phase 6 — Production Deployment",
  "pipeline_ready": true,
  "inference_ready": true,
  "production_skus_count": 50,
  "quarantined_orphans_count": 150,
  "last_run_timestamp": "2026-09-15T12:35:10",
  "governance_summary": {
    "valuation_basis": "Option 1D — Explicit Exclusion of Monetary Valuation",
    "monetary_valuation_blocked": true,
    "on_order_policy": "Policy_B_LT (Option 2A Ratified)",
    "overstock_threshold_weeks": 8,
    "overstock_threshold_status": "POLICY_RATIFIED"
  }
}
```

---

### 3.2. GET `/api/inventory`
Returns on-hand stock, in-transit purchase orders, safety stock levels, and supply coverage for all 50 production SKUs.

**Response Example (200 OK):**
```json
{
  "origin_date": "2025-09-16",
  "n_skus": 50,
  "total_current_stock": 14250.0,
  "total_on_order": 5820.0,
  "items": [
    {
      "sku": "SKU001",
      "product_name": "Product 001",
      "category": "Storage",
      "current_stock": 250.0,
      "on_order": 120.0,
      "lead_time_days": 7.0,
      "safety_stock": 45.0,
      "reorder_point": 95.0,
      "weeks_of_cover": 4.8,
      "stockout_risk_score": 0.0,
      "action_tier": "HEALTHY"
    }
  ],
  "governance": {
    "valuation_basis_confirmed": false,
    "monetary_valuation_blocked": true,
    "on_order_policy": "Policy_B_LT",
    "overstock_threshold_weeks": 8,
    "overstock_threshold_status": "POLICY_RATIFIED"
  }
}
```

---

### 3.3. GET `/api/inventory/{sku}`
Drill-down endpoint returning full SKU profile, inventory position, 8-week forecast curve, and prescriptive action.

**Response Example (200 OK):**
```json
{
  "sku": "SKU010",
  "product_name": "Product 010",
  "category": "Storage",
  "subcategory": "Sofa",
  "current_stock": 37.0,
  "on_order": 46.0,
  "lead_time_days": 9.0,
  "safety_stock": 19.0,
  "reorder_point": 48.0,
  "weeks_of_cover": 0.73,
  "excess_weeks_of_cover": 0.0,
  "stockout_risk_score": 95.4,
  "action_tier": "STOCKOUT_IMMINENT",
  "recommendation_code": "EXPEDITE_PO",
  "recommendation_title": "Expedite Inbound Shipment",
  "recommended_action": "Expedite existing inbound purchase order of 46 units immediately...",
  "priority": "1 — CRITICAL",
  "rationale": "Projected inventory position drops to negative units in week 1...",
  "forecast_h1_8": [112.9, 115.2, 108.4, 110.1, 105.0, 114.2, 118.0, 111.5],
  "governance": {
    "valuation_basis_confirmed": false,
    "monetary_valuation_blocked": true,
    "on_order_policy": "Policy_B_LT",
    "overstock_threshold_status": "POLICY_RATIFIED"
  }
}
```

**Error Response Example (404 Not Found for Quarantined SKU):**
```json
{
  "detail": "SKU 'SKU055' not found in production universe (SKU001–SKU050) or is quarantined."
}
```

---

### 3.4. GET `/api/recommendations`
Returns deterministic operational recommendations for all 50 SKUs sorted by priority.

**Priority Tiers:**
- `1 — CRITICAL`: `EXPEDITE_PO` (Immediate supplier intervention)
- `2 — HIGH`: `PLACE_PO` (Reorder point breached within lead time)
- `3 — MEDIUM`: `REVIEW_PIPELINE` (In-flight order arriving post-breach)
- `4 — LOW`: `FREEZE_REPLENISHMENT` (Excess coverage exceeding 8 weeks)
- `5 — INFORMATIONAL`: `MAINTAIN_SCHEDULE` (Balanced inventory within target buffer)

---

### 3.5. GET `/api/governance`
Audit endpoint returning the complete ratification record for Milestone 5.X.

**Response Example (200 OK):**
```json
{
  "milestone": "Milestone 5.X — Governance Gate Management",
  "phase_5_status": "GOVERNANCE_RATIFIED",
  "phase_6_status": "AUTHORIZED_TO_COMMENCE",
  "valuation_basis": "Option 1D — Explicit Exclusion of Monetary Valuation",
  "valuation_basis_confirmed": false,
  "monetary_valuation_blocked": true,
  "on_order_policy": "Policy_B_LT (Option 2A Ratified)",
  "arrival_timing_confirmed": false,
  "overstock_threshold_weeks": 8,
  "overstock_threshold_status": "POLICY_RATIFIED",
  "production_sku_universe": "SKU001–SKU050 (50 SKUs)",
  "orphan_sku_quarantine": "SKU051–SKU200 (150 SKUs Quarantined)"
}
```

---

## 4. Performance & SLA Guarantees

| Metric | Target SLA | Measured Production Performance |
| :--- | :--- | :--- |
| **API Healthcheck Latency** | $< 10$ ms | $1.2$ ms |
| **P99 Inference Latency (`/predict`)** | $< 200$ ms | $12.4$ ms |
| **Fleet Inventory Latency (`/api/inventory`)** | $< 50$ ms | $8.6$ ms |
| **Batch Pipeline Execution (9 Stages)** | $< 30$ seconds | $6.01$ seconds |
| **Availability / Uptime Target** | $99.9\%$ | High-availability multi-worker architecture |
