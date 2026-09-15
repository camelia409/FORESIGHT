# PROJECT FORESIGHT — PHASE 6 DEPLOYMENT RUNBOOK

**Document ID:** `FORESIGHT-OPS-P6-001`  
**Phase:** Phase 6 — Production Deployment  
**Status:** PRODUCTION READY  
**Audience:** DevOps, Cloud Engineering, System Administrators, Site Reliability Engineers (SRE)  

---

## 1. Deployment Overview

This runbook specifies the step-by-step procedures to deploy, configure, verify, and operate the Project FORESIGHT production environment using either native Python host execution or multi-container Docker orchestration.

### Services Summary
| Service Name | Port | Process | Description |
| :--- | :--- | :--- | :--- |
| **`foresight-api`** | `8000` | `uvicorn api.main:app` | High-performance FastAPI REST prediction and recommendation engine |
| **`foresight-dashboard`** | `8501` | `streamlit run app/streamlit_app.py` | Multi-page Streamlit Executive & Operational Command Center |
| **`foresight-pipeline`** | Batch | `python -m src.production_pipeline` | 9-stage automated batch scoring and decision support DAG |

---

## 2. Prerequisites & System Requirements

### Hardware Requirements
- **CPU:** 4 vCPUs minimum (8 vCPUs recommended for high-volume concurrent inference).
- **RAM:** 8 GB minimum (16 GB recommended).
- **Disk:** 10 GB available SSD storage for model artifacts, parquets, and logs.

### Software Requirements
- **Python:** 3.10 to 3.13 (Python 3.11/3.13 recommended).
- **Container Runtime:** Docker Engine 24.0+ & Docker Compose 2.20+ (for containerized deployments).
- **OS:** Linux (Ubuntu 22.04 LTS / Debian 12 / RHEL 9), macOS, or Windows Server 2022.

---

## 3. Deployment Option A: Docker Compose (Recommended)

### Step 1: Clone and Prepare Environment
```bash
git clone <repository_url> foresight
cd foresight
```

### Step 2: Build Containers
```bash
docker-compose build
```

### Step 3: Launch Services in Background
```bash
docker-compose up -d
```

### Step 4: Verify Container Health
```bash
docker-compose ps
```
Both `foresight-production-api` and `foresight-production-dashboard` should report status `healthy` or `running`.

### Step 5: Smoke Test API and Dashboard
- **API Healthcheck:**
  ```bash
  curl -f http://localhost:8000/health
  # Response: {"status":"healthy","inference_ready":true,"version":"1.0.0"}
  ```
- **API Documentation:** Visit `http://localhost:8000/docs` in your browser.
- **Executive Dashboard:** Visit `http://localhost:8501` in your browser.

---

## 4. Deployment Option B: Native Host Process

### Step 1: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Production Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run the Production Batch Pipeline
```bash
# Windows
scripts\run_pipeline.bat

# Linux / macOS
chmod +x scripts/*.sh
./scripts/run_pipeline.sh
```

### Step 4: Launch the REST API Service
```bash
# Windows
scripts\run_api.bat

# Linux / macOS
./scripts/run_api.sh
```
The API is available at `http://localhost:8000`.

### Step 5: Launch the Streamlit Dashboard
```bash
# Windows
scripts\run_dashboard.bat

# Linux / macOS
./scripts/run_dashboard.sh
```
The Dashboard is available at `http://localhost:8501`.

---

## 5. Automated Pipeline Execution & Scheduling

The batch pipeline orchestrator (`src/production_pipeline.py`) should be scheduled to run weekly or daily following the ingestion of fresh inventory snapshots.

### Linux Crontab (Weekly Sunday Run at 02:00 UTC)
```cron
0 2 * * 0 cd /opt/foresight && /opt/foresight/.venv/bin/python -m src.production_pipeline >> /var/log/foresight_pipeline.log 2>&1
```

### Windows Task Scheduler
Create a Basic Task executing:
- **Program/Script:** `python.exe` (in virtual environment)
- **Add arguments:** `-m src.production_pipeline --origin latest`
- **Start in:** `F:\zidio\foresight`

---

## 6. Verification & Post-Deployment Checklist

- [x] Pre-flight SHA-256 integrity check passes for all 11 core artifacts.
- [x] Pipeline DAG executes all 9 stages in `< 10` seconds.
- [x] Manifest artifact generated at `artifacts/phase6/pipeline_manifest.json` with status `COMPLETED`.
- [x] `GET /health` returns `{"status":"healthy","inference_ready":true}`.
- [x] `GET /api/status` returns `production_skus_count: 50`, `quarantined_orphans_count: 150`.
- [x] `GET /api/recommendations` returns 50 prioritized SKU actions with `monetary_valuation_blocked: true`.
- [x] Streamlit dashboard loads all 5 analytical pages without errors or missing data warnings.
- [x] Full test suite passes: `257 passed, 10 skipped, 0 failed`.
