# FraudShield AI: Real-Time Transaction Fraud Detection System

FraudShield AI is an enterprise-grade, real-time transaction surveillance and fraud detection platform. It combines supervised gradient boosting, unsupervised anomaly detection, Graph Neural Networks (GNN), and local SHAP explainability into a multi-layered scoring pipeline. 

Instead of relying on rigid rule-based systems or opaque black-box models, FraudShield AI computes a composite risk score for every incoming transaction, highlights the exact behavioral drivers behind each decision, and provides security analysts with an interactive live dashboard for threat monitoring and incident response.

---

## Architecture Overview

```
                         [ Live Transaction Stream ]
                                     │
                                     ▼
                      [ POST /api/analyze (Flask API) ]
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   [ XGBoost Model ]       [ Isolation Forest ]        [ PyTorch GNN ]
 (Supervised Patterns)    (Unsupervised Outlier)    (Graph Fraud Rings)
      Weight: 0.45              Weight: 0.35              Weight: 0.20
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
           • Live Stream Table & In-Page Threat Feed
           • Zero-Drift Locked Velocity & Distribution Charts
           • Interactive Vis-Network Graph Visualizer
           • One-Click Account Freeze / Containment
```

### Why a Multi-Model Ensemble?
1. **XGBoost (Supervised)**: Captures non-linear relationships and historical fraud signatures (e.g., transaction-amount-to-balance ratios, suspicious merchant categories).
2. **Isolation Forest (Unsupervised)**: Detects zero-day anomalies and abnormal behavior without needing prior fraud labels.
3. **Graph Neural Network (GNN)**: Identifies syndicated fraud rings, shared device IDs, and cyclic transaction routing across accounts.
4. **SHAP (SHapley Additive exPlanations)**: Delivers instant, mathematically sound feature attributions for compliance (FinCEN SAR) and analyst verification.

---

## Core Capabilities

- **Sub-100ms Inference Pipeline**: Evaluates transactions against the full ensemble in real time.
- **In-Memory Ring Buffer**: Thread-safe rolling buffer holding the last 1,000 transactions for instantaneous UI polling without database lag.
- **Dedicated Threat Alerts Feed**: Replaced intrusive popups with a live, in-page alerts feed that aggregates high-risk transactions (`RiskScore > 0.70`).
- **One-Click Incident Containment**: Freeze compromised accounts instantly via the API or dashboard to intercept all subsequent transactions.
- **Zero-Reflow Dashboard**: Fixed-dimension container architecture for Chart.js prevents downward layout creep during high-frequency live stream polling.
- **Network Graph Visualizer**: Interactive Vis-Network topology showing relationships between sender accounts, target merchants, and shared devices.

---

## Project Structure

```
├── app.py                     # Central Flask API server, scoring engine & route handlers
├── simulate_transactions.py   # Real-time transaction generator & HTTP streaming engine
├── requirements.txt           # Production dependencies
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
│   ├── gnn_model.pt           # PyTorch Geometric GNN weights
│   └── automl/
│       └── trainer.py         # Automated model evaluation and retrainer
│
├── graph_models/
│   ├── data_loader.py         # Dynamic bipartite transaction graph builder
│   ├── gnn_model.py           # Graph Convolutional Network (GCN) architecture
│   └── train_gnn.py           # Script to train GNN on transaction topologies
│
├── profiling/
│   └── builder.py             # Customer profile & spending baseline builder
│
├── drift/
│   └── detector.py            # Concept drift monitoring (Wasserstein distance)
│
└── templates/
    └── dashboard.html         # Modern fintech monitoring dashboard (HTML/CSS/JS)
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

### Step 4: Start the Flask Backend Server

```bash
python app.py
```

You will see the Flask server boot up and bind to port `5000`:
```
 * Running on http://127.0.0.1:5000
 * Running on http://localhost:5000
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

### Step 5: Start the Live Transaction Stream

Open a **second terminal window**, activate your virtual environment, and run the streaming simulator:

**On Windows (PowerShell):**
```powershell
cd "c:\path\to\Ai-powered-real-time-transactions-fraud-detection-system"
.\venv\Scripts\Activate.ps1
python simulate_transactions.py
```

**On macOS / Linux:**
```bash
cd /path/to/Ai-powered-real-time-transactions-fraud-detection-system
source venv/bin/activate
python simulate_transactions.py
```

The simulator will stream realistic transactions directly to `http://localhost:5000/api/analyze` at 1.2-second intervals, introducing realistic fraud scenarios (high-value transfers, login spikes, foreign location jumps, card testing).

---

## API Reference

### 1. Analyze Transaction
`POST /api/analyze`

Submits a single transaction to the multi-model ensemble for instant scoring and ring buffer storage.

**Request Payload:**
```json
{
  "AccountID": "ACC_44921",
  "TransactionAmount": 12500.00,
  "TransactionDuration": 4,
  "LoginAttempts": 5,
  "AccountBalance": 18200.00,
  "TransactionType": "Debit",
  "Channel": "Online",
  "Location": "Moscow",
  "DeviceID": "DEV_8829",
  "MerchantID": "MERCH_102",
  "CustomerOccupation": "Engineer",
  "TransactionDate": "2026-09-18 00:15:00",
  "PreviousTransactionDate": "2026-09-18 00:14:50"
}
```

**Response:**
```json
{
  "transaction_id": "SIM000104",
  "account_id": "ACC_44921",
  "risk_score": 0.8425,
  "is_fraud": true,
  "is_high_risk": true,
  "is_blocked": false,
  "status": "FLAGGED",
  "xgb_score": 0.9945,
  "iso_score": 0.8710,
  "gnn_score": 0.5860,
  "shap_drivers": [
    { "feature": "LoginAttempts", "value": 5, "shap_value": 5.56 },
    { "feature": "TransactionAmount", "value": 12500, "shap_value": 0.89 }
  ],
  "timestamp": "2026-09-18T00:15:00.123456+00:00"
}
```

---

### 2. Fetch Live Dashboard Statistics
`GET /api/stats`

Returns aggregated metrics from the rolling 1,000-transaction buffer.

**Response:**
```json
{
  "total": 450,
  "flagged": 42,
  "high_risk": 18,
  "blocked": 3,
  "cleared": 408,
  "avg_risk": 0.2841,
  "fraud_rate": 9.33,
  "frozen_count": 3
}
```

---

### 3. Fetch Recent Transactions
`GET /api/transactions/live?limit=50`

Returns the most recent transactions stored in the in-memory ring buffer (newest first).

---

### 4. Freeze an Account
`POST /api/account/<account_id>/freeze`

Instantly locks an account to prevent any further outgoing transactions.

**Request Payload:**
```json
{
  "reason": "Repeated high-value international wire anomalies"
}
```

---

### 5. Reset Transaction Ring Buffer
`POST /api/reset`

Clears the in-memory ring buffer and resets counters to zero. Useful during testing or demonstrations.

---

## Manual Testing & Verification

You can quickly verify that the scoring pipeline is responsive using PowerShell or curl:

**PowerShell:**
```powershell
$payload = @{
    AccountID = "TEST_ACC_01"
    TransactionAmount = 14500
    LoginAttempts = 6
    TransactionDuration = 3
    Location = "Moscow"
    Channel = "Online"
    TransactionType = "Debit"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/api/analyze" -Method Post -Body $payload -ContentType "application/json"
```

**cURL:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "AccountID": "TEST_ACC_01",
    "TransactionAmount": 14500,
    "LoginAttempts": 6,
    "TransactionDuration": 3,
    "Location": "Moscow",
    "Channel": "Online",
    "TransactionType": "Debit"
  }'
```

---

## Troubleshooting

### 1. Port 5000 is already in use
If another application or previous Python process is occupying port 5000:
- **Windows (PowerShell)**:
  ```powershell
  # Find PID listening on 5000
  Get-NetTCPConnection -LocalPort 5000 | Select-Object OwningProcess, State
  # Terminate process by PID
  Stop-Process -Id <PID> -Force
  ```
- **macOS / Linux**:
  ```bash
  lsof -i :5000
  kill -9 <PID>
  ```

### 2. Missing pre-trained model files
Ensure the `trained_models/` directory contains `xgboost.pkl`, `isolation_forest.pkl`, and `shap_explainer.pkl`. If missing, `AutoMLTrainer` in `models/automl/trainer.py` can regenerate baseline models from `data/bank_transactions_data_2.csv`.

### 3. Dashboard visual caching
If you make changes to HTML/CSS and don't see updates in the browser, perform a hard refresh (**Ctrl + F5** on Windows/Linux or **Cmd + Shift + R** on Mac).

---

## License

This project is released under the [MIT License](LICENSE).
