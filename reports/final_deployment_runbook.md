# PROJECT FORESIGHT — FINAL DEPLOYMENT RUNBOOK

**Document:** `FORESIGHT-RUNBOOK-FINAL-001`  
**Target Environments:** Streamlit Community Cloud (Dashboard) & Render (REST API)  
**Status:** READY FOR DEPLOYMENT  
**Docker Status:** Container-free / Native Cloud PaaS  

---

## 1. Platforms Selected

To achieve maximum reliability without complex infrastructure or container overhead, the project utilizes standard native Python cloud PaaS platforms:

1. **REST Scoring API:** Render (Free / Starter Python Web Service)
2. **Operations Dashboard:** Streamlit Community Cloud (Free Hosted Python App)

---

## 2. Build & Start Commands

| Component | Platform | Build Command | Start Command |
| :--- | :--- | :--- | :--- |
| **REST Scoring API** | Render / Railway | `pip install -r requirements.txt` | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| **Operations Dashboard** | Streamlit Cloud / Render | `pip install -r requirements.txt` | `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |

---

## 3. Environment Variables & Ports

### REST API Configuration
- `PYTHON_VERSION`: `3.11.9`
- `PORT`: Assigned automatically by platform (default: `8000`)
- `ENVIRONMENT`: `production`

### Dashboard Configuration
- `PYTHON_VERSION`: `3.11.9`
- `PORT`: Assigned automatically by platform (default: `8501`)
- `STREAMLIT_SERVER_HEADLESS`: `true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: `false`

---

## 4. Endpoints & URL Patterns

### REST API
- **Base URL Pattern:** `https://foresight-api-[team].onrender.com`
- **Health Check Endpoint:** `GET /health` (Response: `{"status":"healthy","inference_ready":true}`)
- **Interactive Documentation:** `GET /docs` (Swagger UI)
- **Status Endpoint:** `GET /api/status`

### Dashboard
- **URL Pattern:** `https://[github-user]-foresight-app.streamlit.app`
- **Health Check Path:** `GET /_stcore/health`

---

## 5. Exact Step-by-Step Deployment Procedures

### Procedure A: Deploying the REST API to Render
1. Push the clean `foresight` repository to GitHub:
   ```bash
   git add .
   git commit -m "chore: prepare final submission and deployment configuration"
   git push origin main
   ```
2. Log in to [dashboard.render.com](https://dashboard.render.com).
3. Click **New +** $\rightarrow$ **Blueprint** (or **Web Service**).
4. Connect the GitHub repository.
5. Render detects [`render.yaml`](file:///f:/zidio/foresight/render.yaml) automatically:
   - Sets service `foresight-api`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - Health Path: `/health`
6. Click **Apply** to deploy.

### Procedure B: Deploying the Dashboard to Streamlit Community Cloud
1. Ensure repository is pushed to GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io) using GitHub SSO.
3. Click **New app**.
4. Configure application settings:
   - **Repository:** `<your-github-username>/foresight`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **Python version:** `3.11`
5. Click **Deploy**.

---

## 6. Post-Deployment Smoke Tests

Once remote deployment completes, execute these verification commands against your live URLs:

```bash
# Set your deployed API base URL
export API_URL="https://foresight-api-[team].onrender.com"

# 1. Verify API Health
curl -s -f "$API_URL/health"
# Expected: {"status":"healthy","inference_ready":true}

# 2. Verify System Telemetry & Governance Status
curl -s -f "$API_URL/api/status"
# Expected: status="HEALTHY", production_skus_count=50, quarantined_orphans_count=150

# 3. Verify Live Inference
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"origin_date": "2025-09-01", "skus": ["SKU001"]}'
# Expected: 8 forecast records returned

# 4. Verify Orphan SKU Quarantine (404 Rejection)
curl -s -w "%{http_code}\n" -o /dev/null "$API_URL/api/inventory/SKU051"
# Expected: 404

# 5. Verify Dashboard
curl -s -I "https://[github-user]-foresight-app.streamlit.app" | grep "HTTP/"
# Expected: HTTP/2 200 or HTTP/1.1 200
```
