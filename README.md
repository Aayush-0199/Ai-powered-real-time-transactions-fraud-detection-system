# FraudGuard AI: Real-Time Transaction Fraud Detection System

FraudGuard AI is an enterprise-grade, real-time transaction surveillance and fraud detection platform. It combines supervised gradient boosting, unsupervised anomaly detection, Graph Neural Networks (GNN), and local SHAP explainability into a multi-layered scoring pipeline. 

Instead of relying on rigid rule-based systems or opaque black-box models, FraudGuard AI computes a composite risk score for every incoming transaction, highlights the exact behavioral drivers behind each decision, and provides security analysts with an interactive live dashboard for threat monitoring and incident response.

---

## Architecture Overview

```
                         [ Live Transaction Stream ]
                   (Automated In-Process Background Engine)
                                     │
                                     ▼
                      [ POST /api/analyze (Flask API) ]
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   [ XGBoost Model ]        [ PyTorch GNN ]           [ Isolation Forest ]
 (Supervised Patterns)    (Graph Fraud Rings)       (Unsupervised Outlier)
      Weight: 0.45              Weight: 0.30              Weight: 0.25
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     ▼
                       [ Behavioral Anomaly Boosts ]
                 (Login Spikes, Geo-Hops, Velocity Jumps)
                                     │
                                     ▼
                         [ Composite Risk Score ]
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
        [ SHAP Explainer ]                   [ Ring Buffer (1,000 Txns) ]
   (Top Feature Impact Scores)                          │
                  │                                     ▼
                  └───────────────┬─────────────────────┘
                                  ▼
                     [ Real-Time Web Dashboard ]
           • Live Stream Table with Sticky Headers & Pagination
           • Zero-Drift Locked Velocity & Distribution Charts
           • Interactive Vis-Network Graph Visualizer
           • FinCEN SAR PDF Compliance Report Generation
           • One-Click Account Freeze / Incident Containment
```

### Why a Multi-Model Ensemble?
1. **XGBoost (Supervised - 45%)**: Captures non-linear relationships and historical fraud signatures (e.g., transaction-amount-to-balance ratios, suspicious merchant categories).
2. **Graph Neural Network (GNN - 30%)**: Identifies syndicated fraud rings, shared device IDs, and cyclic transaction routing across accounts.
3. **Isolation Forest (Unsupervised - 25%)**: Detects zero-day anomalies and abnormal behavior without needing prior fraud labels.
4. **SHAP (SHapley Additive exPlanations)**: Delivers instant, mathematically sound feature attributions for compliance (FinCEN SAR) and analyst verification.

---

## Core Capabilities

- **Automated In-Process Simulation**: The live transaction simulator starts automatically in the background when the server launches—no separate terminal or activation command required.
- **Sub-100ms Inference Pipeline**: Evaluates transactions against the full ensemble in real time.
- **Structured Zero-Reflow Table & Pagination**: Fixed-bounds table container with sticky column headers and interactive pagination controls (`10/15/25/50/100` rows per page) preventing infinite downward layout creep.
- **In-Memory Ring Buffer**: Thread-safe rolling buffer holding the last 1,000 transactions for instantaneous UI polling without database lag.
- **Dedicated Threat Alerts Feed**: In-page live threat stream highlighting critical and elevated risk transactions (`RiskScore > 0.70`).
- **FinCEN SAR Compliance Generator**: One-click generation and download of formatted Suspicious Activity Report (SAR) PDFs with cryptographic verification hashes.
- **One-Click Incident Containment**: Freeze compromised accounts instantly via the API or dashboard to intercept all subsequent transactions.
- **Network Graph Visualizer**: Interactive Vis-Network topology showing relationships between sender accounts, target merchants, and shared devices.

---

## Project Structure

```
├── app.py                     # Central Flask API server, scoring engine & built-in auto-simulator
├── simulate_transactions.py   # Standalone CLI transaction streamer (optional)
├── requirements.txt           # Production dependencies
├── start.bat                  # 1-Click Windows startup launcher
├── start.sh                   # Linux / Container startup script
├── Dockerfile                 # Container image specification
├── render.yaml                # Cloud deployment configuration
│
├── data/
│   ├── bank_transactions_data_2.csv # Base dataset for training & baseline statistics
│   └── customer_profiles.json       # Historical profiles for risk tiering
│
├── trained_models/
│   ├── xgboost.pkl            # Trained XGBoost classifier
│   ├── isolation_forest.pkl   # Fitted Isolation Forest model
│   └── shap_explainer.pkl     # TreeExplainer model for local feature attribution
│
├── models/
│   └── automl/
│       └── trainer.py         # Automated self-healing model evaluation and retrainer
│
├── graph_models/
│   ├── data_loader.py         # Dynamic bipartite transaction graph builder
│   ├── gnn_model.py           # Graph Convolutional Network (GCN) architecture & fallback
│   └── gnn_model.pt           # PyTorch Geometric GNN weights
│
├── profiling/
│   └── builder.py             # Customer profile & spending baseline builder
│
├── drift/
│   └── detector.py            # Concept drift monitoring (Wasserstein distance)
│
├── reporting/
│   └── generator.py           # FinCEN SAR PDF report generator (ReportLab)
│
└── templates/
    └── dashboard.html         # Real-time enterprise monitoring dashboard (HTML/CSS/JS)
```

---

## Getting Started

### Prerequisites
- **Python**: Version 3.10 or 3.11 recommended.
- **Git**: Installed and available in your terminal.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aayush-0199/Ai-powered-real-time-transactions-fraud-detection-system.git
cd Ai-powered-real-time-transactions-fraud-detection-system
```

---

### Step 2: Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Start the Application

You only need to run the server. **Transaction simulation starts automatically in the background on startup.**

**Option A (Windows 1-Click Launcher):**
```cmd
start.bat
```

**Option B (Direct Python Command):**
```bash
python app.py
```

---

### Step 5: Open the Dashboard

Open your web browser and navigate to:
```
http://localhost:5000
```

The live transactions table, risk charts, threat alerts feed, and network graph will immediately stream data automatically.

*(Note: `python simulate_transactions.py` is completely optional and only needed if you wish to run a dedicated external CLI streaming monitor).*

---

## API Reference

### 1. Analyze Transaction
`POST /api/analyze`

Submits a single transaction to the multi-model ensemble for instant scoring and ring buffer storage.

**Request Payload:**
```json
{
  "AccountID": "AC00128",
  "TransactionAmount": 12500.00,
  "TransactionDuration": 18,
  "LoginAttempts": 6,
  "AccountBalance": 15000.00,
  "TransactionType": "Wire Transfer",
  "Location": "Moscow",
  "DeviceID": "DVC_SUSP_01",
  "MerchantID": "M102"
}
```

**Response:**
```json
{
  "transaction_id": "SIM000104",
  "account_id": "AC00128",
  "risk_score": 0.8425,
  "is_fraud": true,
  "is_high_risk": true,
  "is_blocked": false,
  "status": "FLAGGED",
  "xgb_score": 0.9945,
  "iso_score": 0.8710,
  "gnn_score": 0.5860,
  "shap_drivers": [
    { "feature": "LoginAttempts", "value": 6, "shap_value": 0.556 },
    { "feature": "TransactionAmount", "value": 12500, "shap_value": 0.389 }
  ],
  "timestamp": "2026-09-18T00:15:00.123456+00:00"
}
```

---

### 2. Fetch Live Dashboard Statistics
`GET /api/stats`

Returns aggregated metrics from the rolling 1,000-transaction buffer.

---

### 3. Fetch Recent Transactions
`GET /api/transactions/live?limit=100`

Returns the most recent transactions stored in the in-memory ring buffer (newest first).

---

### 4. Incident Response: Account Freeze / Unfreeze
- `POST /api/accounts/freeze` — Freeze an account to intercept all subsequent transactions.
  ```json
  { "account_id": "AC00128", "reason": "Suspicious wire transfer activity" }
  ```
- `POST /api/accounts/unfreeze` — Unfreeze an account.
  ```json
  { "account_id": "AC00128" }
  ```
- `GET /api/accounts/frozen` — List all currently frozen accounts.

---

### 5. Simulator Controls
- `GET /api/simulator/status` — Get live background simulator status and streaming statistics.
- `POST /api/simulator/toggle` — Pause or resume the automatic transaction stream.

---

### 6. Compliance SAR Report Generation
`POST /api/reports/sar`

Generates and downloads a complete FinCEN-compliant Suspicious Activity Report (SAR) PDF for a flagged transaction with SHAP explainability and audit trail.

---

### 7. Reset Transaction Ring Buffer
`POST /api/reset`

Clears the in-memory ring buffer and resets counters to zero. Useful during testing or demonstrations.

---

## Troubleshooting

### 1. Port 5000 is already in use
If another application or previous Python process is occupying port 5000:
- **Windows (PowerShell)**:
  ```powershell
  Get-NetTCPConnection -LocalPort 5000 | Select-Object OwningProcess, State
  Stop-Process -Id <PID> -Force
  ```
- **macOS / Linux**:
  ```bash
  lsof -i :5000
  kill -9 <PID>
  ```

### 2. Missing pre-trained model files
Ensure the `trained_models/` directory contains `xgboost.pkl`, `isolation_forest.pkl`, and `shap_explainer.pkl`. If missing, `AutoMLTrainer` in `models/automl/trainer.py` can automatically regenerate baseline models from `data/bank_transactions_data_2.csv`.

### 3. Dashboard visual caching
If you make changes to HTML/CSS and don't see updates in the browser, perform a hard refresh (**Ctrl + F5** on Windows/Linux or **Cmd + Shift + R** on Mac).
