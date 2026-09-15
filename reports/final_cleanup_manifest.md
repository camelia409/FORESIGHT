# PROJECT FORESIGHT — FINAL CLEANUP MANIFEST

**Date:** 2026-09-15  
**Author:** Lead ML / Inventory Intelligence Engineer  
**Status:** APPROVED & EXECUTED  

---

## 1. Overview & Deletion Policy

In accordance with Phase B and Phase E of the Project FORESIGHT Final Submission plan, this cleanup manifest catalogs all temporary, generated, and cache artifacts identified for safe removal.

### Safety Verification Protocol
Before executing any deletion:
1. **Reference Check:** Complete repository search (`grep_search`) to guarantee no source module (`src/`), test (`tests/`), API endpoint (`api/`), dashboard page (`app/`), or pipeline runner (`scripts/`) imports or calls the target file.
2. **Protected Artifact Check:** Confirmed none of the 11 protected artifacts (`data/raw/`, `models/production/`, `artifacts/risk/`, `artifacts/decision_support/`) are touched.
3. **Audit Trail Integrity:** Ensured that formal markdown reports in `reports/` and JSON/CSV artifacts in `artifacts/` documenting investigations remain intact.

---

## 2. Cleanup Action Manifest

| File / Folder | Action | Reason | Reference Check |
| :--- | :---: | :--- | :--- |
| `scratch/` | **REMOVE** | Directory of temporary test runners and brief text dumps created during operational smoke testing. | No production code or tests import from `scratch/`. |
| `investigation.py` | **REMOVE** | Ad-hoc EDA/data audit script used during preliminary data discovery. | Zero imports found in `src/`, `tests/`, `api/`, `app/`. |
| `inv_final_conclusion.py` | **REMOVE** | Temporary script verifying inventory value correlation hypotheses. | Zero imports found across project. |
| `inv_rotation_test.py` | **REMOVE** | Temporary script testing snapshot date shifts and rotations. | Zero imports found across project. |
| `inv_value_deep.py` | **REMOVE** | Ad-hoc script inspecting per-SKU inventory valuation discrepancies. | Zero imports found across project. |
| `print_findings.py` | **REMOVE** | Temporary terminal formatting script for early findings. | Zero imports found across project. |
| `scratch_calc_val_arch.py` | **REMOVE** | Temporary script for validating model architecture comparison math. | Zero imports found across project. |
| `scratch_inspect_val.py` | **REMOVE** | Temporary inspection script for validation splits. | Zero imports found across project. |
| `scratch_inv_basis_investigation.py` | **REMOVE** | One-off script for investigating Inventory_Value vs Cost_Price/Selling_Price. | Zero imports found across project. Findings preserved in `reports/risk/`. |
| `scratch_on_order_investigation.py` | **REMOVE** | One-off script analyzing On_Order behavior and delivery assumptions. | Zero imports found across project. Findings preserved in `reports/risk/`. |
| `scratch_phase4a_audit.py` | **REMOVE** | One-off script auditing Phase 4A formulas. | Zero imports found across project. |
| `scratch_phase4a_val.py` | **REMOVE** | One-off script validating Phase 4A metric outputs. | Zero imports found across project. |
| `scratch_phase4a_verify.py` | **REMOVE** | Temporary verification script for Phase 4A risk engine. | Zero imports found across project. |
| `write_investigation_report.py` | **REMOVE** | One-off script that generated early draft markdown for investigation report. | Zero imports found across project. Target report committed. |
| `.pytest_cache/` | **REMOVE** | Pytest test execution cache directory. | Generated runtime cache; excluded in `.gitignore`. |
| `api/__pycache__/` | **REMOVE** | Python bytecode cache directory. | Standard compiled bytecode; excluded in `.gitignore`. |
| `app/__pycache__/` | **REMOVE** | Python bytecode cache directory. | Standard compiled bytecode; excluded in `.gitignore`. |
| `app/pages/__pycache__/` | **REMOVE** | Python bytecode cache directory. | Standard compiled bytecode; excluded in `.gitignore`. |
| `src/__pycache__/` | **REMOVE** | Python bytecode cache directory. | Standard compiled bytecode; excluded in `.gitignore`. |
| `tests/__pycache__/` | **REMOVE** | Python bytecode cache directory. | Standard compiled bytecode; excluded in `.gitignore`. |

---

## 3. Preserved Essential Artifacts Summary

The following core components have been audited and explicitly **PRESERVED**:

1. **Source Code (`src/`):**
   - `preprocessing.py` (Data ingestion & cleaning pipeline)
   - `features.py` (Feature engineering & lag generation)
   - `models.py` (Forecasting model architecture & inference)
   - `risk_engine.py` (Inventory risk scoring engine)
   - `decision_support.py` (Operational recommendation & priority engine)
   - `production_pipeline.py` (End-to-end batch pipeline DAG)
   - `evaluate.py` (Evaluation metrics: WAPE, MAPE, RMSE, Coverage)
2. **API Service (`api/`):**
   - `main.py`, `inference.py`, `schemas.py`
3. **Interactive Dashboard (`app/`):**
   - `streamlit_app.py`, `data_loader.py`, and all 5 page modules in `app/pages/`
4. **Test Suite (`tests/`):**
   - All 15 test suites encompassing 257 unit and integration tests
5. **Operational Scripts (`scripts/`):**
   - `run_api.bat`/`.sh`, `run_dashboard.bat`/`.sh`, `run_pipeline.bat`/`.sh`
6. **Core Protected Artifacts (11/11):**
   - All 11 baseline artifacts verified bitwise identical via SHA-256
7. **Reports & Governance (`reports/`, `artifacts/governance/`):**
   - All Phase 1–6 formal documentation, governance ratification records, and executive readout memo.
