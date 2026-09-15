# MEMORANDUM

**TO:** Head of Operations, Finance Lead — NorthBay Living  
**FROM:** Data Science & Analytics Engagement Team (Project FORESIGHT)  
**DATE:** September 15, 2026  
**SUBJECT:** **Project FORESIGHT — Executive Readout: Demand & Inventory Intelligence Platform**  
**ENGAGEMENT DELIVERABLE:** D7 — Executive Readout Memo  

---

## 1. Executive Summary: What We Found & What to Do

NorthBay Living’s omnichannel business faces a common retail challenge: **simultaneous stockouts on fast-moving items and capital tied up in slow-moving surplus inventory**.

Through Project FORESIGHT, we deployed an artificial intelligence forecasting and automated inventory risk engine across NorthBay’s core product fleet. 

### Key Fleet Takeaways Across 50 Core Products:
1. **Critical Stockout Alerts (8 SKUs):** Immediate replenishment acceleration is required. Physical stock will deplete within Week 1, inside supplier lead times.
2. **Reorder Required (10 SKUs):** Reorder points are breached within supplier lead times; new purchase orders must be issued this week.
3. **Pipeline In-Flight (24 SKUs):** Inventory is safely protected by open purchase orders currently in transit.
4. **Surplus Inventory (8 SKUs):** More than 8 weeks of forward supply cover is on hand; replenishment should be frozen to protect working capital.
5. **Healthy Baseline (Median Cover = 3.8 Weeks):** The core fleet is well-balanced within the optimal 2-to-8 week supply window.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FLEET TRIAGE SUMMARY                            │
├────────────────────────────────┬───────────────────────────────────────┤
│ P1 — Critical Expedite (8)     │ Contact suppliers within 24h          │
│ P2 — Place New PO (10)         │ Issue orders within 48h               │
│ P3 — Review Pipeline (24)      │ Monitor in-flight deliveries          │
│ P4 — Freeze Surplus (8)        │ Halt replenishment; clearance review  │
│ Total Production Fleet (50)    │ 150 Incomplete Catalog SKUs Isolated  │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 2. Recommended Operational Directives

### Immediate Action List (Top Urgency SKUs)
| SKU | Product Name | Category | On-Hand Stock | Inbound On-Order | Lead Time | Projected Stockout | Immediate Directive |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **SKU010** | Product 010 | Storage | 37 units | 46 units | 9 days | Week 1 | **Expedite Inbound PO:** Stockout occurs in 7 days; split-ship or air-freight existing PO. |
| **SKU012** | Product 012 | Home Decor | 43 units | 17 units | 12 days | Week 1 | **Expedite Inbound PO:** Stockout occurs in 7 days; accelerate supplier dispatch. |
| **SKU018** | Product 018 | Furniture | 28 units | 35 units | 14 days | Week 1 | **Expedite Inbound PO:** Lead time is 14 days; physical stock depletes before delivery. |
| **SKU025** | Product 025 | Electronics | 52 units | 0 units | 7 days | Week 1 | **Emergency PO:** Zero pipeline orders exist; place emergency PO for 180 units immediately. |

### Surplus Management (Capital Protection)
- **8 SKUs hold coverage in excess of 8 weeks** (up to 24 weeks on select items).
- **Directive:** Halt automatic reordering on these items immediately. SCM and merchandising should review whether targeted promotional bundles or seasonal clearances can accelerate turnover without cannibalizing full-price lines.

---

## 3. Financial Governance & Valuation Integrity Note

The engagement brief originally proposed quantifying financial impact in monetary units. During our forensic data audit, Engineering and Data Science uncovered critical accounting discrepancies in the raw historical extracts:
- Raw `Inventory_Value` per SKU diverged by **$+136.9\%$ against Cost Price** and **$+164.9\%$ against Selling Price**.
- 16 SKUs exhibited negative gross margin flags (`Cost_Price > Selling_Price`).

### Ratified Business Decision (Option 1D):
To protect NorthBay Living leadership from making multi-million-rupee purchasing commitments on inaccurate surrogate valuations, **Finance and Management formally ratified Option 1D — Explicit Exclusion of Monetary Valuation**. 

Until NorthBay’s core ERP system delivers authoritative Weighted Average Cost (WAC) or Standard Cost feeds:
- All platform recommendations operate strictly on **physical units and weeks of supply cover**.
- No proxy rupee figures are displayed that could mislead purchasing teams.

---

## 4. Forecasting Model Accuracy & Honest Limitations

### How Well Does the AI Forecast?
We tested multiple machine learning architectures against NorthBay Living’s 2-year sales history using strict 12-fold rolling backtesting (mimicking real weekly forecasting without looking ahead):
- **Traditional Seasonal-Naive Baseline:** 32.1% WAPE (Weighted Absolute Percentage Error).
- **FORESIGHT Production Machine Learning Model:** **13.9% WAPE** on Horizon 1 and **15.4% WAPE** on Horizon 2.
- **Accuracy Improvement:** **+18.2 percentage points more accurate** than standard historical methods, cutting demand forecasting errors by more than half.

```
Forecast Error Comparison (Lower is Better):
  Traditional Baseline: [████████████████████] 32.1% WAPE
  FORESIGHT ML Engine:   [█████████           ] 13.9% WAPE  (18.2% Margin of Victory)
```

### Honest Operational Limitations:
1. **Horizon Degradation:** Machine learning models excel on 1-to-2 week tactical replenishment ($h=1,2$). Beyond 2 weeks, supply chain uncertainty increases; FORESIGHT blends trend-adjusted seasonal baselines for Weeks 3–8.
2. **Promotions & External Shocks:** Forecasts assume typical promotional lifts. Unplanned marketing flashes or competitor stockouts will require human buyer override via the dashboard.
3. **150 Quarantined SKUs:** 150 items found in warehouse records lacked product master data (no category, cost, or sales history). These have been safely quarantined and isolated so they cannot corrupt production inventory plans.

---

## 5. Tools Handed Over to NorthBay Living

1. **Automated Production Pipeline:** A single-command script (`scripts/run_pipeline.bat`) that refreshes forecasts, risk scores, and recommendations in **6 seconds**.
2. **Interactive Streamlit Dashboard:** Accessible locally or via cloud deployment (`http://localhost:8501`), providing:
   - Executive Overview & Category Risk Mix.
   - 8-Week Forward SKU Demand Curves.
   - 2D Inventory Risk Matrix & Breach Timelines.
   - Role-Based Action Center (Procurement Buyer, Inventory Controller, Warehouse Ops).
   - 360° SKU Intelligence Profiles.
3. **REST Prediction & Risk API:** Programmatic service (`http://localhost:8000`) ready for direct integration into NorthBay’s ERP and WMS.

---

## 6. Action Plan for Operations & Finance (Next 30 Days)

| Week | Target Milestone | Responsible Team |
| :---: | :--- | :--- |
| **Week 1** | Triage the 8 P1 Expedite SKUs; issue orders for the 10 P2 Reorder SKUs. | Procurement & Sourcing |
| **Week 2** | Freeze POs on the 8 P4 surplus items; review markdown potential. | Inventory Control / Merchandising |
| **Week 3** | Resolve catalog master data for the 150 quarantined orphan SKUs. | ERP / Master Data Management |
| **Week 4** | Schedule automated weekly batch scoring (Sundays at 02:00 UTC). | IT & Systems Engineering |

---
*Project FORESIGHT — Delivered by Data Science & Analytics Engineering.*
