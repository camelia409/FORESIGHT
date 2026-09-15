# Project FORESIGHT — Formal Stakeholder Decision Prompts
## Milestone 5.X: Executive Communications Collateral

**Document ID:** `FORESIGHT-GOV-STAKEHOLDER-PROMPTS`  
**Author:** Lead ML / Inventory Intelligence Engineer  
**Purpose:** Official copy-paste communication templates to send to Finance, Supply Chain Operations, Inventory Management, and the Executive Sponsor to collect formal business sign-offs.

---

## 1. Prompt #1: Finance / Inventory Valuation

**Target Recipient:** Head of Finance / Corporate Controller  
**Subject:** FORESIGHT — Approval Required: Inventory Valuation Basis  

> **Subject:** FORESIGHT — Approval Required: Inventory Valuation Basis
>
> We need your formal decision for the FORESIGHT inventory intelligence system.
>
> The current dataset contains an `Inventory_Value`-derived per-SKU value, but investigation shows it does not reliably match `Cost_Price`, `Selling_Price`, WAC, or other tested valuation hypotheses. Therefore, Engineering will not use a proxy valuation.
>
> Please select ONE:
>
> **1A — Standard Cost:** Use the authoritative ERP Standard Cost as the unit valuation basis.
>
> **1B — Weighted Average Cost (WAC):** Use the authoritative monthly GL/WAC costing feed.
>
> **1C — FIFO / Batch Cost:** Use authoritative lot/batch-level costing.
>
> **1D — Exclude Monetary Valuation:** Do not calculate inventory monetary exposure in production. FORESIGHT will operate using physical units, demand forecasts, weeks of cover, shortage/excess classifications, and recommendations only.
>
> **Recommended immediate option if no authoritative costing feed is currently available: 1D.**
>
> Please confirm:
>
> **Approved Option:** 1A / 1B / 1C / 1D  
> **Authoritative ERP/Data Source:** __________  
> **Approver Name:** __________  
> **Role:** __________  
> **Approval Date:** __________  
> **Additional Conditions:** __________  
>
> This decision is required before monetary inventory metrics can be enabled.

---

## 2. Prompt #2: Supply Chain / Operations

**Target Recipient:** VP Supply Chain Operations / Head of Procurement  
**Subject:** FORESIGHT — Approval Required: On_Order Arrival Policy  

> **Subject:** FORESIGHT — Approval Required: On_Order Arrival Policy
>
> FORESIGHT currently has `On_Order` quantities but does not contain PO placement dates, promised delivery dates, PO line IDs, or discrete delivery schedules.
>
> Observed supplier lead times are between **3 and 14 days**, so Engineering has implemented Policy B_LT:
>
> `Inventory Position = Current Stock + On_Order × arrival-within-horizon indicator − forecast demand`
>
> Under this policy:
>
> * For week 1, On_Order is included only when lead time ≤ 7 days.
> * By week 2, all observed On_Order quantities are included because observed lead times are ≤14 days.
> * No artificial delivery dates are generated.
>
> Please select ONE:
>
> **2A — Ratify Policy B_LT:** Accept this lead-time-based approach as the operational interim baseline.
>
> **2B — Require ERP PO Integration:** Require line-item purchase-order data containing expected delivery timing before production deployment.
>
> **Recommended immediate option: 2A**, provided Supply Chain accepts the lead-time assumption.
>
> Please confirm:
>
> **Approved Option:** 2A / 2B  
> **Approver Name:** __________  
> **Role:** __________  
> **Approval Date:** __________  
> **Additional Conditions:** __________  
>
> If 2B is selected, please identify the ERP/data source and expected availability of the PO line-item feed.

---

## 3. Prompt #3: Inventory Management / Planning

**Target Recipient:** Head of Inventory Planning / Merchandising  
**Subject:** FORESIGHT — Approval Required: Overstock Threshold  

> **Subject:** FORESIGHT — Approval Required: Overstock Threshold
>
> FORESIGHT needs an operational threshold defining when projected inventory should be classified as excess.
>
> Engineering evaluated three supported thresholds:
>
> **3A — 4 weeks:** Aggressive JIT  
> Fleet alert rate: approximately **52.17%**
>
> **3B — 6 weeks:** Moderate buffer  
> Fleet alert rate: approximately **36.09%**
>
> **3C — 8 weeks:** Forecast-aligned strategic buffer  
> Fleet alert rate: approximately **25.57%**
>
> Engineering recommends **3C — 8 weeks** because the production forecasting system provides an 8-week forecast horizon (`h=1..8`). It also produced no false alarms among the balanced SKU group in the tested analysis.
>
> Please select ONE:
>
> **Approved Threshold:** 4 weeks / 6 weeks / 8 weeks  
> **Approver Name:** __________  
> **Role:** __________  
> **Approval Date:** __________  
> **Additional Conditions:** __________  
>
> This threshold will govern the production excess-inventory classification.

---

## 4. Prompt #4: Executive Sponsor (Consolidated Final Authorization)

**Target Recipient:** Executive Project Sponsor  
**Subject:** FORESIGHT — Final Phase 6 Authorization Request  

> **Subject:** FORESIGHT — Final Phase 6 Authorization Request
>
> The three FORESIGHT policy decisions have been reviewed and documented:
>
> **Decision #1 — Inventory Valuation:** [1A / 1B / 1C / 1D]  
> **Decision #2 — On_Order Policy:** [2A / 2B]  
> **Decision #3 — Overstock Threshold:** [4 / 6 / 8 weeks]  
>
> Finance/Supply Chain/Inventory Management approvals are attached/documented.
>
> Please provide final authorization for Phase 6:
>
> **[ ] AUTHORIZE PHASE 6**  
> **[ ] DO NOT AUTHORIZE — additional conditions required**  
>
> Executive Sponsor: __________  
> Role: __________  
> Date: __________  
> Conditions / Comments: __________  
