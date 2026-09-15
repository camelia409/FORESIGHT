# PROJECT FORESIGHT — PHASE 6 OPERATIONS RUNBOOK

**Document ID:** `FORESIGHT-OPS-P6-002`  
**Phase:** Phase 6 — Production Deployment & Ongoing Operations  
**Status:** ACTIVE & APPROVED  
**Audience:** Tier 1/2/3 Operations, System Administrators, Supply Chain System Integrators  

---

## 1. Operational Cadence & Schedule

Project FORESIGHT operates on a weekly automated batch scoring cadence with continuous on-demand REST API serving and interactive dashboard access.

```
[ Weekly Snapshot Delivery ] (Sundays 00:00 UTC)
             │
             ▼
[ Automated Pipeline Execution ] (Sundays 02:00 UTC)
             │
             ├─ Ingests new inventory positions
             ├─ Runs 8-week ML demand forecasts
             ├─ Projects forward inventory position
             ├─ Generates prioritized action queue (P1–P5)
             └─ Emits audit manifest and CSV/Parquet artifacts
             │
             ▼
[ Business Users Review & Act ] (Monday 08:00 Local)
             ├─ Procurement buyers execute P1 Expedites & P2 Reorders
             ├─ Merchandising controllers review P4 Overstock freezes
             └─ Operations monitors inbound deliveries
```

---

## 2. Standard Operating Procedures (SOPs)

### SOP-001: Triggering Manual Pipeline Run
When fresh inventory snapshots are delivered mid-week or following data corrections:
```bash
# Windows
f:\zidio\foresight\scripts\run_pipeline.bat

# Linux
/opt/foresight/scripts/run_pipeline.sh --origin latest
```
Verify completion in `artifacts/phase6/pipeline_manifest.json` (`"status": "COMPLETED"`).

---

### SOP-002: Service Health Inspection & Restart
If API or Dashboard becomes unresponsive:
```bash
# Check Docker service health
docker-compose ps

# Restart services cleanly
docker-compose restart

# Tail API service logs
docker-compose logs -f --tail=100 foresight-api
```

For native host deployments:
```bash
# Inspect process status
Get-Process -Name python | Select-Object Id, ProcessName, CPU, WorkingSet

# Terminate and relaunch via scripts
scripts\run_api.bat
scripts\run_dashboard.bat
```

---

### SOP-003: Quarantined SKU Inquiry Resolution
If business users inquire why an SKU is missing from recommendations:
1. Check SKU ID. Only `SKU001` through `SKU050` are part of the active production fleet.
2. `SKU051` through `SKU200` are quarantined orphan SKUs identified in Phase 0 audit as lacking catalog metadata in `sku_master.csv`.
3. To admit quarantined SKUs into production:
   - Business must provide master catalog attributes (`Product_Name`, `Category`, `Subcategory`, `Cost_Price`, `Selling_Price`, `Lead_Time_Days`).
   - SCM team must resolve negative margin flags.
   - Retrain models and promote via formal Phase 3B gate.

---

### SOP-004: Validating Upstream Artifact Integrity
To confirm that model files and datasets have not suffered bitwise drift or silent corruption:
```bash
python -c "
import hashlib
from pathlib import Path
for p in ['models/production/models/random_forest_h1.joblib', 'models/production/models/xgboost_h2.joblib', 'data/processed/analysis_ready.parquet']:
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    print(f'{p}: {h[:16]}...')
"
```
Compare against authoritative hashes in `reports/governance/phase5_final_gate_status.md`.

---

## 3. Incident Management & Troubleshooting

| Alert / Symptom | Potential Cause | Remediation Procedure |
| :--- | :--- | :--- |
| **`GET /health` returns 503** | Model weights missing or corrupted | Verify files exist in `models/production/models/`. Restart API service to trigger reload. |
| **`GET /api/inventory/{sku}` returns 404** | SKU is an orphan (`SKU051`–`SKU200`) or non-existent | Consult SOP-003. Confirm SKU is within `SKU001`–`SKU050`. |
| **Pipeline fails at `STAGE_INGEST`** | `inventory_snapshots.csv` or `sku_master.csv` missing | Confirm raw files are present in `data/raw/`. Check file permissions. |
| **Dashboard shows blank charts** | Latest recommendations parquet not generated | Execute `scripts/run_pipeline.bat` to produce `artifacts/phase6/production_latest_recommendations.parquet`. |
| **Zero monetary figures displayed** | Expected behavior under Option 1D | Do NOT attempt to fix. Decision #1 strictly excludes monetary valuation metrics until ERP WAC/Standard Cost integration. |

---

## 4. Disaster Recovery & Backup

- **Artifact Backups:** `artifacts/` and `reports/` should be backed up following every batch pipeline run to immutable object storage (AWS S3, GCP Cloud Storage, or Azure Blob Storage).
- **Model Weights:** Frozen production models in `models/production/models/` are immutable and backed up in version control / model registry.
- **Recovery Time Objective (RTO):** $< 15$ minutes to full service restoration.
- **Recovery Point Objective (RPO):** 0 data loss for analytical outputs (regenerable in 6 seconds via pipeline DAG).
