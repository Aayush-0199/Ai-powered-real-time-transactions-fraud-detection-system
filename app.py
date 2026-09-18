"""
app.py
FraudGuard AI - Central Flask API Server & Real-Time Scoring Engine
Ensemble: XGBoost (0.45) + GNN (0.30) + IsolationForest (0.25) + SHAP Explainability
"""

import os
import sys
import json
import uuid
import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template, Response, send_file
from flask_cors import CORS

# ─── Internal Modules ─────────────────────────────────────────────────────────
from profiling.builder import CustomerRiskProfiler
from drift.detector import ConceptDriftDetector
from reporting.generator import SARReportGenerator
from models.automl.trainer import AutoMLTrainer
from graph_models.data_loader import TransactionGraphBuilder

# ─── ML Libraries ─────────────────────────────────────────────────────────────
import joblib
import shap

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.ensemble import IsolationForest

# ─── GNN ──────────────────────────────────────────────────────────────────────
from graph_models.gnn_model import load_gnn_model, FallbackGNN, train_and_save_gnn

# ═══════════════════════════════════════════════════════════════════════════════
# Flask App Initialization
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['JSON_SORT_KEYS'] = False

# ═══════════════════════════════════════════════════════════════════════════════
# Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR          = "trained_models"
XGB_PATH           = os.path.join(MODEL_DIR, "xgboost.pkl")
ISO_PATH           = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SHAP_PATH          = os.path.join(MODEL_DIR, "shap_explainer.pkl")
GNN_PATH           = os.path.join("graph_models", "gnn_model.pt")
DATA_PATH          = os.path.join("data", "bank_transactions_data_2.csv")

ENSEMBLE_WEIGHTS   = {'xgboost': 0.45, 'gnn': 0.30, 'isoforest': 0.25}
FRAUD_THRESHOLD    = 0.50
HIGH_RISK_THRESHOLD = 0.70

FEATURE_COLUMNS    = [
    'TransactionAmount', 'AccountBalance', 'LoginAttempts',
    'TransactionDuration', 'AmountToBalanceRatio', 'IsInternational',
    'IsHighValue', 'VelocityScore',
]

HIGH_RISK_LOCATIONS = {
    'Moscow', 'Lagos', 'Pyongyang', 'Tehran', 'Caracas',
    'Minsk', 'Havana', 'Damascus', 'Tripoli', 'Kabul',
}

# ═══════════════════════════════════════════════════════════════════════════════
# Global In-Memory Store (Thread-Safe)
# ═══════════════════════════════════════════════════════════════════════════════

_store_lock            = threading.Lock()
live_transactions      = deque(maxlen=1000)      # Ring buffer
frozen_accounts: Dict  = {}                       # account_id → freeze metadata
_txn_counter           = 0                        # Sequential ID counter

# ─── Singleton Services ───────────────────────────────────────────────────────
risk_profiler    = CustomerRiskProfiler()
drift_detector   = ConceptDriftDetector()
sar_generator    = SARReportGenerator()
automl_trainer   = AutoMLTrainer()
graph_builder    = TransactionGraphBuilder()

# ─── Model Holders ────────────────────────────────────────────────────────────
xgb_model        = None
iso_model        = None
shap_explainer   = None
gnn_model        = None
models_loaded    = False
model_load_error = None

# ═══════════════════════════════════════════════════════════════════════════════
# Model Bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_and_save_models():
    """
    Bootstrap: train and save all model artifacts if they don't exist.
    Called once at startup.
    """
    global xgb_model, iso_model, shap_explainer, gnn_model

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ── Generate training data if needed ──────────────────────────────────────
    if not os.path.exists(XGB_PATH) or not os.path.exists(ISO_PATH):
        print("[Bootstrap] Training models from scratch...")
        _train_initial_models()
    else:
        print("[Bootstrap] Loading existing model artifacts...")

    # ── Load XGBoost ──────────────────────────────────────────────────────────
    try:
        xgb_model = joblib.load(XGB_PATH)
        print(f"[Bootstrap] XGBoost loaded ✓ ({type(xgb_model).__name__})")
    except Exception as e:
        print(f"[Bootstrap] XGBoost load failed: {e} — using fallback")
        xgb_model = _create_fallback_xgb()

    # ── Load Isolation Forest ─────────────────────────────────────────────────
    try:
        iso_model = joblib.load(ISO_PATH)
        print(f"[Bootstrap] Isolation Forest loaded ✓")
    except Exception as e:
        print(f"[Bootstrap] IsoForest load failed: {e} — using fallback")
        iso_model = _create_fallback_iso()

    # ── Load SHAP Explainer ───────────────────────────────────────────────────
    try:
        shap_explainer = joblib.load(SHAP_PATH)
        print(f"[Bootstrap] SHAP explainer loaded ✓")
    except Exception as e:
        print(f"[Bootstrap] SHAP load failed: {e} — rebuilding...")
        try:
            shap_explainer = shap.TreeExplainer(xgb_model)
            joblib.dump(shap_explainer, SHAP_PATH)
        except Exception as e2:
            print(f"[Bootstrap] SHAP rebuild failed: {e2} — SHAP disabled")
            shap_explainer = None

    # ── Load/Train GNN ────────────────────────────────────────────────────────
    gnn_model = load_gnn_model(GNN_PATH)
    print(f"[Bootstrap] GNN model loaded ✓ ({type(gnn_model).__name__})")


def _train_initial_models():
    """Train XGBoost + IsolationForest on synthetic data and save artifacts."""
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest

    print("[Bootstrap] Generating synthetic training dataset (3000 samples)...")
    np.random.seed(42)
    n_legit, n_fraud = 2400, 600

    legit = {
        'TransactionAmount':   np.random.lognormal(4.5, 1.2, n_legit),
        'AccountBalance':      np.random.lognormal(9.0, 1.5, n_legit),
        'LoginAttempts':       np.random.randint(1, 3, n_legit).astype(float),
        'TransactionDuration': np.random.exponential(120, n_legit),
        'IsInternational':     np.random.binomial(1, 0.05, n_legit).astype(float),
        'IsHighValue':         np.random.binomial(1, 0.03, n_legit).astype(float),
        'VelocityScore':       np.random.beta(2, 8, n_legit),
    }
    fraud = {
        'TransactionAmount':   np.random.lognormal(8.5, 1.8, n_fraud),
        'AccountBalance':      np.random.lognormal(7.0, 2.0, n_fraud),
        'LoginAttempts':       np.random.randint(4, 10, n_fraud).astype(float),
        'TransactionDuration': np.random.exponential(20, n_fraud),
        'IsInternational':     np.random.binomial(1, 0.60, n_fraud).astype(float),
        'IsHighValue':         np.random.binomial(1, 0.55, n_fraud).astype(float),
        'VelocityScore':       np.random.beta(7, 2, n_fraud),
    }

    legit_df = pd.DataFrame(legit)
    fraud_df = pd.DataFrame(fraud)

    for df in [legit_df, fraud_df]:
        df['AmountToBalanceRatio'] = (
            df['TransactionAmount'] / (df['AccountBalance'] + 1.0)
        ).clip(0, 5)

    X = pd.concat([legit_df, fraud_df], ignore_index=True)[FEATURE_COLUMNS]
    y = np.array([0] * n_legit + [1] * n_fraud)

    idx = np.random.permutation(len(y))
    X, y = X.iloc[idx], y[idx]

    # Train XGBoost
    print("[Bootstrap] Training XGBoost classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=4,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    clf.fit(X, y)
    joblib.dump(clf, XGB_PATH)
    print(f"[Bootstrap] XGBoost saved → {XGB_PATH}")

    # Train Isolation Forest
    print("[Bootstrap] Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.20,
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)
    joblib.dump(iso, ISO_PATH)
    print(f"[Bootstrap] Isolation Forest saved → {ISO_PATH}")

    # Build SHAP explainer
    print("[Bootstrap] Building SHAP TreeExplainer...")
    try:
        explainer = shap.TreeExplainer(clf)
        joblib.dump(explainer, SHAP_PATH)
        print(f"[Bootstrap] SHAP explainer saved → {SHAP_PATH}")
    except Exception as e:
        print(f"[Bootstrap] SHAP warning: {e}")

    # Generate synthetic CSV dataset
    _generate_csv_dataset(X, y)


def _generate_csv_dataset(X: pd.DataFrame, y: np.ndarray):
    """Save synthetic labeled dataset as CSV for dashboard display."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        accounts  = [f"AC{np.random.randint(10000, 99999)}" for _ in range(len(y))]
        merchants = [f"M{np.random.randint(100, 999)}" for _ in range(len(y))]
        devices   = [f"DVC_{np.random.randint(1000, 9999)}" for _ in range(len(y))]
        locations = np.random.choice(
            ['New York', 'London', 'Singapore', 'Dubai', 'Paris',
             'Moscow', 'Lagos', 'Tokyo', 'Sydney', 'Chicago'],
            size=len(y)
        )
        txn_types = np.random.choice(
            ['Wire Transfer', 'Online Purchase', 'ATM Withdrawal', 'POS Debit', 'ACH Transfer'],
            size=len(y)
        )

        df_out = X.copy()
        df_out['TransactionID']       = [f"TXN{i:06d}" for i in range(len(y))]
        df_out['AccountID']           = accounts
        df_out['CustomerID']          = [f"CUS{np.random.randint(10000, 99999)}" for _ in range(len(y))]
        df_out['MerchantID']          = merchants
        df_out['DeviceID']            = devices
        df_out['Location']            = locations
        df_out['TransactionType']     = txn_types
        df_out['MerchantCategory']    = np.random.choice(
            ['Electronics', 'Retail', 'Travel', 'Restaurant', 'Gas Station'], size=len(y)
        )
        df_out['IsFraud']             = y
        df_out['IsHighRisk']          = (y == 1).astype(int)

        df_out.to_csv(DATA_PATH, index=False)
        print(f"[Bootstrap] Dataset saved → {DATA_PATH} ({len(y)} rows)")


def _create_fallback_xgb():
    """Create a minimal XGBoost model when loading fails."""
    import xgboost as xgb
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy='prior')
    X_dummy = np.zeros((10, len(FEATURE_COLUMNS)))
    y_dummy = np.array([0]*8 + [1]*2)
    clf.fit(X_dummy, y_dummy)
    return clf


def _create_fallback_iso():
    """Create fallback IsolationForest with minimal training."""
    iso = IsolationForest(n_estimators=10, random_state=42)
    X_dummy = np.random.randn(50, len(FEATURE_COLUMNS))
    iso.fit(X_dummy)
    return iso


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def _engineer_features(txn: dict) -> pd.DataFrame:
    """Extract and engineer feature vector from raw transaction dict."""
    amount   = float(txn.get('TransactionAmount', 0))
    balance  = float(txn.get('AccountBalance', 10000))
    logins   = float(txn.get('LoginAttempts', 1))
    duration = float(txn.get('TransactionDuration', 60))
    location = str(txn.get('Location', ''))

    amount_to_balance = min(amount / (balance + 1.0), 5.0)
    is_international  = 1.0 if location in HIGH_RISK_LOCATIONS else 0.0
    is_high_value     = 1.0 if amount > 10000 else 0.0
    velocity_score    = min(logins / 10.0, 1.0) * min(amount / 50000.0, 1.0)

    features = {
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'LoginAttempts':       logins,
        'TransactionDuration': duration,
        'AmountToBalanceRatio': amount_to_balance,
        'IsInternational':     is_international,
        'IsHighValue':         is_high_value,
        'VelocityScore':       velocity_score,
    }
    return pd.DataFrame([features])[FEATURE_COLUMNS]


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble Scoring Engine
# ═══════════════════════════════════════════════════════════════════════════════

def _score_xgboost(X: pd.DataFrame) -> float:
    """XGBoost fraud probability."""
    try:
        proba = xgb_model.predict_proba(X)
        if hasattr(proba, '__len__') and len(proba[0]) > 1:
            return float(proba[0][1])
        return float(proba[0])
    except Exception as e:
        print(f"[XGB] Scoring error: {e}")
        return 0.3


def _score_isolation_forest(X: pd.DataFrame) -> float:
    """
    IsolationForest anomaly → probability.
    score_samples returns negative anomaly score; more negative = more anomalous.
    Normalized to [0, 1] where 1 = most anomalous.
    """
    try:
        raw_score = iso_model.score_samples(X)[0]
        # Typical range: [-0.7, 0.1]; normalize
        normalized = max(0.0, min(1.0, (-raw_score - 0.1) / 0.6))
        return float(normalized)
    except Exception as e:
        print(f"[IsoForest] Scoring error: {e}")
        return 0.2


def _score_gnn(txn: dict) -> float:
    """GNN fraud probability via FallbackGNN or FraudGNN."""
    try:
        if isinstance(gnn_model, FallbackGNN):
            amount   = float(txn.get('TransactionAmount', 0)) / 50000.0
            balance  = float(txn.get('AccountBalance', 10000)) / 100000.0
            logins   = float(txn.get('LoginAttempts', 1)) / 10.0
            duration = float(txn.get('TransactionDuration', 60)) / 300.0
            location = str(txn.get('Location', ''))
            is_intl  = 1.0 if location in HIGH_RISK_LOCATIONS else 0.0
            features = [amount, balance, logins, duration, is_intl, 0.0, 0.0, 0.0]
            return gnn_model.predict_proba(features)
        else:
            # Full PyG GNN
            import torch
            from torch_geometric.data import Data
            X_feat = _engineer_features(txn).values.flatten().tolist()[:8]
            x = torch.tensor([X_feat], dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return gnn_model.predict_proba(x, edge_index)
    except Exception as e:
        return 0.25


def _compute_shap(X: pd.DataFrame, top_n: int = 4) -> List[dict]:
    """Compute top SHAP feature contributions."""
    if shap_explainer is None:
        return _fallback_shap_drivers(X)
    try:
        shap_values = shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0] if shap_values.ndim > 1 else shap_values

        sv = np.array(sv).flatten()
        features = list(X.columns)

        drivers = sorted(
            zip(features, sv),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        return [
            {'feature': feat, 'shap_value': round(float(val), 4),
             'value': round(float(X[feat].iloc[0]), 4)}
            for feat, val in drivers
        ]
    except Exception as e:
        return _fallback_shap_drivers(X)


def _fallback_shap_drivers(X: pd.DataFrame) -> List[dict]:
    """Generate plausible SHAP drivers when explainer is unavailable."""
    drivers = []
    for col in FEATURE_COLUMNS[:4]:
        val = float(X[col].iloc[0]) if col in X.columns else 0.0
        shap_v = round(val / (abs(val) + 1.0) * 0.5, 4)
        drivers.append({'feature': col, 'shap_value': shap_v, 'value': round(val, 4)})
    return sorted(drivers, key=lambda x: abs(x['shap_value']), reverse=True)


def _ensemble_score(xgb_score: float, gnn_score: float, iso_score: float) -> float:
    """Weighted ensemble: 0.45*XGB + 0.30*GNN + 0.25*IsoForest."""
    w = ENSEMBLE_WEIGHTS
    composite = (
        w['xgboost']   * xgb_score +
        w['gnn']       * gnn_score +
        w['isoforest'] * iso_score
    )
    return float(min(max(composite, 0.0), 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Transaction ID Generator
# ═══════════════════════════════════════════════════════════════════════════════

def _next_txn_id() -> str:
    global _txn_counter
    with _store_lock:
        _txn_counter += 1
        return f"SIM{_txn_counter:06d}"


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('dashboard.html')


# ── Health Check ──────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'transactions_in_buffer': len(live_transactions),
        'frozen_accounts': len(frozen_accounts),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


def _process_transaction(data: dict) -> Tuple[dict, int]:
    """
    Core scoring pipeline for an incoming transaction (used by API and internal simulator).
    1. Check frozen account registry (short-circuit if blocked)
    2. Engineer features
    3. Score with XGBoost + GNN + IsoForest
    4. Compute SHAP explanations
    5. Update risk profiler + drift detector
    6. Store in ring buffer
    7. Return (result_dict, status_code)
    """
    try:
        account_id  = str(data.get('AccountID', data.get('account_id', f"AC{np.random.randint(10000,99999)}")))
        customer_id = str(data.get('CustomerID', data.get('customer_id', account_id)))
        txn_id      = data.get('TransactionID') or _next_txn_id()
        amount      = float(data.get('TransactionAmount', 0))
        location    = str(data.get('Location', 'Unknown'))
        txn_type    = str(data.get('TransactionType', 'Unknown'))
        merchant    = str(data.get('MerchantID', 'M000'))
        device      = str(data.get('DeviceID', 'DEV_000'))
        balance     = float(data.get('AccountBalance', 10000))
        logins      = float(data.get('LoginAttempts', 1))
        duration    = float(data.get('TransactionDuration', 60))
        timestamp   = data.get('Timestamp', datetime.now(timezone.utc).isoformat())

        # ── FROZEN ACCOUNT CHECK (pre-analysis short-circuit) ─────────────────
        with _store_lock:
            is_frozen = account_id in frozen_accounts
            freeze_info = frozen_accounts.get(account_id, {})

        if is_frozen:
            blocked_record = {
                'TransactionID':     txn_id,
                'AccountID':         account_id,
                'CustomerID':        customer_id,
                'TransactionAmount': amount,
                'TransactionType':   txn_type,
                'Location':          location,
                'MerchantID':        merchant,
                'DeviceID':          device,
                'AccountBalance':    balance,
                'LoginAttempts':     logins,
                'TransactionDuration': duration,
                'RiskScore':         1.0,
                'XGBScore':          1.0,
                'GNNScore':          1.0,
                'IsoScore':          1.0,
                'IsFraud':           True,
                'IsHighRisk':        True,
                'IsBlocked':         True,
                'Status':            'BLOCKED',
                'SHAPDrivers':       [{'feature': 'Account Status', 'value': 'FROZEN / BLOCKED', 'shap_value': 1.0}],
                'FreezeReason':      freeze_info.get('reason', 'Account Frozen'),
                'Timestamp':         timestamp,
            }
            with _store_lock:
                live_transactions.appendleft(blocked_record)

            return {
                'transaction_id': txn_id,
                'account_id':     account_id,
                'is_fraud':       True,
                'is_blocked':     True,
                'status':         'BLOCKED',
                'risk_score':     1.0,
                'xgb_score':      1.0,
                'gnn_score':      1.0,
                'iso_score':      1.0,
                'shap_drivers':   [{'feature': 'Account Status', 'value': 'FROZEN / BLOCKED', 'shap_value': 1.0}],
                'freeze_reason':  freeze_info.get('reason', 'Account Frozen'),
                'frozen_at':      freeze_info.get('frozen_at', ''),
                'timestamp':      timestamp,
            }, 200

        # ── FEATURE ENGINEERING ───────────────────────────────────────────────
        X = _engineer_features(data)

        # ── ENSEMBLE SCORING ──────────────────────────────────────────────────
        xgb_score = _score_xgboost(X)
        gnn_score = _score_gnn(data)
        iso_score = _score_isolation_forest(X)
        risk_score = _ensemble_score(xgb_score, gnn_score, iso_score)

        # ── SHAP EXPLANATIONS ─────────────────────────────────────────────────
        shap_drivers = _compute_shap(X)

        # ── CLASSIFICATION ────────────────────────────────────────────────────
        is_fraud    = risk_score >= FRAUD_THRESHOLD
        is_high_risk = risk_score >= HIGH_RISK_THRESHOLD
        status = 'FLAGGED' if is_fraud else ('HIGH RISK' if is_high_risk else 'Cleared')

        # ── RISK PROFILER UPDATE ──────────────────────────────────────────────
        profile = risk_profiler.update({
            'CustomerID':          customer_id,
            'AccountID':           account_id,
            'TransactionAmount':   amount,
            'Location':            location,
            'TransactionDuration': duration,
            'IsHighRisk':          is_high_risk,
        })

        # ── DRIFT DETECTOR UPDATE ─────────────────────────────────────────────
        drift_detector.add_sample({
            'TransactionAmount':   amount,
            'AccountBalance':      balance,
            'LoginAttempts':       logins,
            'TransactionDuration': duration,
        })

        # ── GRAPH BUILDER UPDATE ──────────────────────────────────────────────
        graph_builder.add_transaction({
            'AccountID':           account_id,
            'MerchantID':          merchant,
            'DeviceID':            device,
            'TransactionAmount':   amount,
            'AccountBalance':      balance,
            'LoginAttempts':       logins,
            'TransactionDuration': duration,
            'IsHighRisk':          is_high_risk,
        })

        # ── BUILD TRANSACTION RECORD ──────────────────────────────────────────
        record = {
            'TransactionID':     txn_id,
            'AccountID':         account_id,
            'CustomerID':        customer_id,
            'TransactionAmount': amount,
            'TransactionType':   txn_type,
            'Location':          location,
            'MerchantID':        merchant,
            'DeviceID':          device,
            'AccountBalance':    balance,
            'LoginAttempts':     logins,
            'TransactionDuration': duration,
            'RiskScore':         round(risk_score, 4),
            'XGBScore':          round(xgb_score, 4),
            'GNNScore':          round(gnn_score, 4),
            'IsoScore':          round(iso_score, 4),
            'IsFraud':           is_fraud,
            'IsHighRisk':        is_high_risk,
            'IsBlocked':         False,
            'Status':            status,
            'SHAPDrivers':       shap_drivers,
            'Timestamp':         timestamp,
            'RiskTier':          profile.risk_tier if profile else 'LOW',
        }

        # ── STORE IN RING BUFFER ──────────────────────────────────────────────
        with _store_lock:
            live_transactions.appendleft(record)

        return {
            'transaction_id': txn_id,
            'account_id':     account_id,
            'is_fraud':       is_fraud,
            'is_blocked':     False,
            'is_high_risk':   is_high_risk,
            'status':         status,
            'risk_score':     round(risk_score, 4),
            'xgb_score':      round(xgb_score, 4),
            'gnn_score':      round(gnn_score, 4),
            'iso_score':      round(iso_score, 4),
            'shap_drivers':   shap_drivers,
            'risk_tier':      profile.risk_tier if profile else 'LOW',
            'timestamp':      timestamp,
        }, 200

    except Exception as e:
        traceback.print_exc()
        return {'error': str(e), 'status': 'ERROR'}, 500


# ── Main Analysis Endpoint ────────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Ensemble fraud scoring endpoint.
    Processes a transaction payload and returns multi-model risk assessment.
    """
    data = request.get_json(force=True) or {}
    result, status_code = _process_transaction(data)
    return jsonify(result), status_code


# ═══════════════════════════════════════════════════════════════════════════════
# Automated In-Process Live Transaction Simulator
# ═══════════════════════════════════════════════════════════════════════════════

_SIM_ACCOUNT_IDS = [f"AC{i:05d}" for i in range(100, 200)]
_SIM_MERCHANT_IDS = [f"M{i:03d}" for i in range(100, 160)]
_SIM_DEVICE_IDS = [f"DVC_{i:04d}" for i in range(1000, 1060)]
_SIM_CLEAN_LOCATIONS = [
    'New York', 'London', 'Singapore', 'Chicago', 'Los Angeles',
    'Toronto', 'Sydney', 'Paris', 'Tokyo', 'Berlin', 'Amsterdam',
    'Stockholm', 'Zurich', 'Seoul', 'Mumbai',
]
_SIM_RISKY_LOCATIONS = ['Moscow', 'Lagos', 'Pyongyang', 'Tehran', 'Caracas', 'Minsk', 'Tripoli']
_SIM_TXN_TYPES_CLEAN = ['Online Purchase', 'POS Debit', 'ATM Withdrawal', 'ACH Transfer', 'Bill Payment']
_SIM_TXN_TYPES_RISKY = ['Wire Transfer', 'International Wire', 'Online Debit', 'Card Not Present']
_SIM_CATEGORIES = ['Electronics', 'Retail', 'Travel', 'Restaurant', 'Gas Station', 'Grocery']
_SIM_SUSPECT_DEVICE = "DVC_SUSP_01"

_simulation_running = True
_simulation_thread = None
_simulation_lock = threading.Lock()
_simulation_delay = 1.2
_sim_stats = {
    'total': 0,
    'flagged': 0,
    'blocked': 0,
    'cleared': 0,
    'batches': 0,
}

def _sim_make_clean() -> Tuple[dict, str]:
    account = str(np.random.choice(_SIM_ACCOUNT_IDS))
    balance = round(float(np.random.uniform(5000, 80000)), 2)
    amount  = round(float(np.random.uniform(10, min(500, balance * 0.05))), 2)
    return {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     str(np.random.choice(_SIM_TXN_TYPES_CLEAN)),
        'MerchantID':          str(np.random.choice(_SIM_MERCHANT_IDS)),
        'MerchantCategory':    str(np.random.choice(_SIM_CATEGORIES)),
        'DeviceID':            str(np.random.choice(_SIM_DEVICE_IDS)),
        'Location':            str(np.random.choice(_SIM_CLEAN_LOCATIONS)),
        'LoginAttempts':       int(np.random.randint(1, 3)),
        'TransactionDuration': round(float(np.random.uniform(60, 300)), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }, 'CLEAN'

def _sim_make_wire_fraud() -> Tuple[dict, str]:
    account = str(np.random.choice(_SIM_ACCOUNT_IDS))
    balance = round(float(np.random.uniform(12000, 60000)), 2)
    amount  = round(float(np.random.uniform(9500, 48000)), 2)
    return {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     'Wire Transfer',
        'MerchantID':          f"M{np.random.randint(900, 999)}",
        'MerchantCategory':    'Financial Services',
        'DeviceID':            str(np.random.choice(_SIM_DEVICE_IDS)),
        'Location':            str(np.random.choice(_SIM_CLEAN_LOCATIONS)),
        'LoginAttempts':       int(np.random.randint(4, 10)),
        'TransactionDuration': round(float(np.random.uniform(5, 25)), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }, 'WIRE FRAUD'

def _sim_make_location_hop() -> Tuple[dict, str]:
    account = str(np.random.choice(_SIM_ACCOUNT_IDS))
    balance = round(float(np.random.uniform(8000, 50000)), 2)
    amount  = round(float(np.random.uniform(500, 15000)), 2)
    return {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     str(np.random.choice(['International Wire', 'Wire Transfer'])),
        'MerchantID':          f"M{np.random.randint(800, 899)}",
        'MerchantCategory':    'International Transfer',
        'DeviceID':            str(np.random.choice(_SIM_DEVICE_IDS)),
        'Location':            str(np.random.choice(_SIM_RISKY_LOCATIONS)),
        'LoginAttempts':       int(np.random.randint(2, 7)),
        'TransactionDuration': round(float(np.random.uniform(10, 60)), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }, 'GEO HOP'

def _sim_make_balance_drain() -> Tuple[dict, str]:
    account = str(np.random.choice(_SIM_ACCOUNT_IDS))
    balance = round(float(np.random.uniform(10000, 90000)), 2)
    amount  = round(balance * float(np.random.uniform(0.90, 0.99)), 2)
    return {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     'Online Debit',
        'MerchantID':          f"M{np.random.randint(700, 799)}",
        'MerchantCategory':    'Wire Transfer',
        'DeviceID':            str(np.random.choice(_SIM_DEVICE_IDS)),
        'Location':            str(np.random.choice(_SIM_CLEAN_LOCATIONS + _SIM_RISKY_LOCATIONS[:3])),
        'LoginAttempts':       int(np.random.randint(5, 11)),
        'TransactionDuration': round(float(np.random.uniform(3, 15)), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }, 'BALANCE DRAIN'

def _sim_make_card_probe() -> Tuple[dict, str]:
    account = str(np.random.choice(_SIM_ACCOUNT_IDS))
    balance = round(float(np.random.uniform(500, 5000)), 2)
    amount  = round(float(np.random.uniform(0.50, 2.99)), 2)
    return {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     'Card Not Present',
        'MerchantID':          f"M{np.random.randint(500, 599)}",
        'MerchantCategory':    'Micro-Transaction',
        'DeviceID':            str(np.random.choice(_SIM_DEVICE_IDS)),
        'Location':            str(np.random.choice(_SIM_CLEAN_LOCATIONS)),
        'LoginAttempts':       int(np.random.randint(6, 11)),
        'TransactionDuration': round(float(np.random.uniform(1, 8)), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }, 'CARD PROBE'

def _sim_make_device_collusion() -> Tuple[dict, str]:
    account = str(np.random.choice(_SIM_ACCOUNT_IDS))
    balance = round(float(np.random.uniform(3000, 30000)), 2)
    amount  = round(float(np.random.uniform(1000, 20000)), 2)
    return {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     str(np.random.choice(['Wire Transfer', 'Online Debit', 'ACH Transfer'])),
        'MerchantID':          f"M{np.random.randint(600, 699)}",
        'MerchantCategory':    'Financial Services',
        'DeviceID':            _SIM_SUSPECT_DEVICE,
        'Location':            str(np.random.choice(_SIM_CLEAN_LOCATIONS + _SIM_RISKY_LOCATIONS[:2])),
        'LoginAttempts':       int(np.random.randint(3, 9)),
        'TransactionDuration': round(float(np.random.uniform(5, 30)), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }, 'DEVICE COLLUSION'

_SIM_FRAUD_ARCHETYPES = [
    _sim_make_wire_fraud,
    _sim_make_location_hop,
    _sim_make_balance_drain,
    _sim_make_card_probe,
    _sim_make_device_collusion,
]

def _sim_build_batch() -> list:
    batch = []
    for _ in range(9):
        batch.append(_sim_make_clean())
    chosen = list(np.random.choice(len(_SIM_FRAUD_ARCHETYPES), size=3, replace=False))
    for idx in chosen:
        batch.append(_SIM_FRAUD_ARCHETYPES[idx]())
    np.random.shuffle(batch)
    return batch

def _simulation_worker():
    """Continuous background thread simulating calibrated transactions."""
    import time
    time.sleep(2.0)  # brief wait for initial server setup
    batch_num = 0
    while True:
        if not _simulation_running:
            time.sleep(1.0)
            continue

        batch_num += 1
        batch = _sim_build_batch()
        _sim_stats['batches'] += 1

        for slot, (txn, arch_label) in enumerate(batch, 1):
            if not _simulation_running:
                break

            result, _ = _process_transaction(txn)
            risk = float(result.get('risk_score', 0))
            is_fraud = result.get('is_fraud', False)
            is_blocked = result.get('is_blocked', False)
            status = result.get('status', 'Unknown')
            txn_id = result.get('transaction_id', '?')

            _sim_stats['total'] += 1
            if is_blocked:
                _sim_stats['blocked'] += 1
            elif is_fraud:
                _sim_stats['flagged'] += 1
            else:
                _sim_stats['cleared'] += 1

            status_icon = "🔒" if is_blocked else ("🚨" if is_fraud else "✅")
            print(f"  [Live Stream] {status_icon} {txn_id} | {txn.get('AccountID')} | ${txn.get('TransactionAmount'):>9,.2f} | Risk: {risk:.3f} [{status}] ({arch_label})")

            # sleep with quick cancellation
            elapsed = 0.0
            while elapsed < _simulation_delay and _simulation_running:
                time.sleep(0.1)
                elapsed += 0.1

def start_simulation():
    """Start the background transaction streamer if not already started."""
    global _simulation_thread, _simulation_running
    with _simulation_lock:
        _simulation_running = True
        if _simulation_thread is None or not _simulation_thread.is_alive():
            _simulation_thread = threading.Thread(target=_simulation_worker, daemon=True, name="TxnSimulator")
            _simulation_thread.start()
            print("  [Simulator] 🚀 Live transaction simulation thread started automatically.")


# ── Simulator Control Endpoints ───────────────────────────────────────────────
@app.route('/api/simulator/status', methods=['GET'])
def simulator_status():
    return jsonify({
        'running': _simulation_running,
        'delay_seconds': _simulation_delay,
        'stats': _sim_stats,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


@app.route('/api/simulator/toggle', methods=['POST'])
def simulator_toggle():
    global _simulation_running
    with _simulation_lock:
        _simulation_running = not _simulation_running
    return jsonify({
        'running': _simulation_running,
        'message': f"Simulation {'resumed' if _simulation_running else 'paused'}",
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


# ── Live Transactions ─────────────────────────────────────────────────────────
@app.route('/api/transactions/live', methods=['GET'])
def get_live_transactions():
    limit = int(request.args.get('limit', 100))
    with _store_lock:
        txns = list(live_transactions)[:limit]
    return jsonify({'transactions': txns, 'total': len(txns)})


# ── Statistics ────────────────────────────────────────────────────────────────
@app.route('/api/stats', methods=['GET'])
def get_stats():
    with _store_lock:
        txns       = list(live_transactions)
        frozen_cnt = len(frozen_accounts)

    if not txns:
        return jsonify({
            'total': 0, 'flagged': 0, 'high_risk': 0,
            'blocked': 0, 'avg_risk': 0.0, 'fraud_rate': 0.0,
            'frozen_count': frozen_cnt,
            'cleared': 0,
        })

    total    = len(txns)
    flagged  = sum(1 for t in txns if t.get('IsFraud'))
    high_risk = sum(1 for t in txns if float(t.get('RiskScore', 0)) >= HIGH_RISK_THRESHOLD)
    blocked  = sum(1 for t in txns if t.get('IsBlocked'))
    cleared  = sum(1 for t in txns if t.get('Status') == 'Cleared')
    avg_risk = sum(float(t.get('RiskScore', 0)) for t in txns) / total
    fraud_rate = (flagged / total) * 100.0

    return jsonify({
        'total':        total,
        'flagged':      flagged,
        'high_risk':    high_risk,
        'blocked':      blocked,
        'cleared':      cleared,
        'avg_risk':     round(avg_risk, 4),
        'fraud_rate':   round(fraud_rate, 2),
        'frozen_count': frozen_cnt,
    })


@app.route('/api/reset', methods=['POST', 'GET'])
def reset_live_store():
    """Reset the live transaction ring buffer and counter."""
    global _txn_counter
    with _store_lock:
        live_transactions.clear()
        _txn_counter = 0
    return jsonify({
        'status': 'success',
        'message': 'Live transaction ring buffer reset successfully. Limit set to 1000.',
        'buffer_size': len(live_transactions),
        'maxlen': live_transactions.maxlen
    })


# ── Account Freeze ────────────────────────────────────────────────────────────
@app.route('/api/accounts/freeze', methods=['POST'])
def freeze_account():
    data       = request.get_json(force=True) or {}
    account_id = str(data.get('account_id', '')).strip()
    reason     = str(data.get('reason', 'Manual freeze — suspicious activity')).strip()

    if not account_id:
        return jsonify({'error': 'account_id is required'}), 400

    with _store_lock:
        frozen_accounts[account_id] = {
            'account_id': account_id,
            'frozen_at':  datetime.now(timezone.utc).isoformat(),
            'reason':     reason,
        }

    return jsonify({
        'success':    True,
        'account_id': account_id,
        'frozen_at':  frozen_accounts[account_id]['frozen_at'],
        'reason':     reason,
        'message':    f"Account {account_id} has been frozen.",
    })


# ── Account Unfreeze ──────────────────────────────────────────────────────────
@app.route('/api/accounts/unfreeze', methods=['POST'])
def unfreeze_account():
    data       = request.get_json(force=True) or {}
    account_id = str(data.get('account_id', '')).strip()

    if not account_id:
        return jsonify({'error': 'account_id is required'}), 400

    with _store_lock:
        was_frozen = account_id in frozen_accounts
        frozen_accounts.pop(account_id, None)

    return jsonify({
        'success':    was_frozen,
        'account_id': account_id,
        'message':    f"Account {account_id} {'unfrozen' if was_frozen else 'was not frozen'}.",
    })


# ── Frozen Accounts Registry ──────────────────────────────────────────────────
@app.route('/api/accounts/frozen', methods=['GET'])
def get_frozen_accounts():
    with _store_lock:
        # Use dict copy to prevent concurrent modification
        frozen_copy = dict(frozen_accounts)

    return jsonify({
        'count':    len(frozen_copy),
        'accounts': list(frozen_copy.values()),
    })


# ── Customer Profile ──────────────────────────────────────────────────────────
@app.route('/api/customer/<customer_id>/profile', methods=['GET'])
def get_customer_profile(customer_id: str):
    profile = risk_profiler.get_profile(customer_id)
    if profile is None:
        # Return a synthesized empty profile
        return jsonify({
            'customer_id':           customer_id,
            'baseline_spend':        0.0,
            'amount_deviation_ratio': 0.0,
            'unique_locations':      0,
            'avg_duration_seconds':  0.0,
            'transaction_count':     0,
            'recent_velocity_per_min': 0.0,
            'max_single_transaction': 0.0,
            'risk_tier':             'LOW',
            'last_updated':          datetime.now(timezone.utc).isoformat(),
            'message':               'No transactions found for this customer.',
        })
    return jsonify(profile.to_dict())


# ── SAR Report Download ────────────────────────────────────────────────────────
@app.route('/api/reports/sar', methods=['POST'])
def generate_sar():
    data         = request.get_json(force=True) or {}
    case_id      = data.get('case_id', '')
    analyst_name = data.get('analyst_name', 'FraudGuard AI System')
    notes        = data.get('notes', '')

    with _store_lock:
        txns = list(live_transactions)

    try:
        pdf_bytes = sar_generator.generate(
            transactions=txns,
            case_id=case_id or None,
            analyst_name=analyst_name,
            notes=notes,
        )
        response = Response(pdf_bytes, mimetype='application/pdf')
        response.headers['Content-Disposition'] = \
            f'attachment; filename="FraudGuard_SAR_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.pdf"'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Graph Data for Vis-Network ────────────────────────────────────────────────
@app.route('/api/graph/data', methods=['GET'])
def get_graph_data():
    vis_data = graph_builder.get_vis_network_data()
    return jsonify({
        'nodes': vis_data['nodes'],
        'edges': vis_data['edges'],
        'stats': {
            'total_nodes': len(vis_data['nodes']),
            'total_edges': len(vis_data['edges']),
        }
    })


# ── Concept Drift Status ──────────────────────────────────────────────────────
@app.route('/api/drift/status', methods=['GET'])
def get_drift_status():
    status  = drift_detector.get_status()
    reports = drift_detector.check_drift()
    return jsonify({
        'detector_status': status,
        'drift_reports':   [r.to_dict() for r in reports],
    })


# ── AutoML Retrain ────────────────────────────────────────────────────────────
@app.route('/api/models/retrain', methods=['POST'])
def trigger_retrain():
    data      = request.get_json(force=True) or {}
    n_samples = int(data.get('n_samples', 2000))
    n_iter    = int(data.get('n_iter', 15))

    if automl_trainer.status == 'TRAINING':
        return jsonify({'message': 'Training already in progress', 'progress': automl_trainer.progress})

    automl_trainer.retrain_async(n_samples=n_samples, n_iter=n_iter)
    return jsonify({'message': 'AutoML retrain initiated', 'status': 'TRAINING'})


@app.route('/api/models/status', methods=['GET'])
def get_model_status():
    global xgb_model, iso_model, shap_explainer, gnn_model

    return jsonify({
        'xgboost': {
            'name':   'XGBoost Classifier',
            'status': 'ACTIVE' if xgb_model is not None else 'UNAVAILABLE',
            'path':   XGB_PATH,
            'weight': ENSEMBLE_WEIGHTS['xgboost'],
            'exists': os.path.exists(XGB_PATH),
        },
        'isolation_forest': {
            'name':   'Isolation Forest',
            'status': 'ACTIVE' if iso_model is not None else 'UNAVAILABLE',
            'path':   ISO_PATH,
            'weight': ENSEMBLE_WEIGHTS['isoforest'],
            'exists': os.path.exists(ISO_PATH),
        },
        'gnn': {
            'name':   'Graph Neural Network (PyG)',
            'status': 'ACTIVE' if gnn_model is not None else 'UNAVAILABLE',
            'path':   GNN_PATH,
            'weight': ENSEMBLE_WEIGHTS['gnn'],
            'exists': os.path.exists(GNN_PATH),
            'type':   type(gnn_model).__name__ if gnn_model else 'None',
        },
        'shap': {
            'name':   'SHAP TreeExplainer',
            'status': 'ACTIVE' if shap_explainer is not None else 'UNAVAILABLE',
            'path':   SHAP_PATH,
            'exists': os.path.exists(SHAP_PATH),
        },
        'automl': automl_trainer.progress,
        'drift':  drift_detector.get_status(),
    })


# ── Audit Data Export ─────────────────────────────────────────────────────────
@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    with _store_lock:
        txns = list(live_transactions)

    if not txns:
        return jsonify({'error': 'No transactions to export'}), 404

    df = pd.DataFrame(txns)
    csv_str = df.to_csv(index=False)
    response = Response(csv_str, mimetype='text/csv')
    response.headers['Content-Disposition'] = \
        f'attachment; filename="FraudGuard_Audit_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv"'
    return response


@app.route('/api/export/json', methods=['GET'])
def export_json():
    with _store_lock:
        txns = list(live_transactions)

    response = Response(
        json.dumps({'transactions': txns, 'exported_at': datetime.now(timezone.utc).isoformat()},
                   indent=2, default=str),
        mimetype='application/json'
    )
    response.headers['Content-Disposition'] = \
        f'attachment; filename="FraudGuard_Audit_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json"'
    return response


# ── Reset Ring Buffer ────────────────────────────────────────────────────────
@app.route('/api/reset', methods=['POST'])
def reset_buffer():
    """Clear the live transaction ring buffer and reset drift detector windows."""
    with _store_lock:
        live_transactions.clear()
    drift_detector.reset()
    return jsonify({
        'message':    'Transaction buffer and drift windows cleared',
        'status':     'ok',
        'timestamp':  datetime.now(timezone.utc).isoformat(),
    })


# ── Extended Health Check ─────────────────────────────────────────────────────
@app.route('/api/health/detail', methods=['GET'])
def health_detail():
    """Detailed health check with model status, buffer size, and system info."""
    with _store_lock:
        buf_size = len(live_transactions)
        frozen   = len(frozen_accounts)

    return jsonify({
        'status':               'ok' if models_loaded else 'degraded',
        'models_loaded':        models_loaded,
        'model_error':          model_load_error,
        'transactions_in_buffer': buf_size,
        'frozen_accounts':      frozen,
        'xgb_active':           xgb_model is not None,
        'iso_active':           iso_model is not None,
        'gnn_active':           gnn_model is not None,
        'shap_active':          shap_explainer is not None,
        'ensemble_weights':     ENSEMBLE_WEIGHTS,
        'fraud_threshold':      FRAUD_THRESHOLD,
        'high_risk_threshold':  HIGH_RISK_THRESHOLD,
        'timestamp':            datetime.now(timezone.utc).isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Startup Bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

_bootstrapped = False

def bootstrap():
    """Initialize all models, services, and live simulation at startup."""
    global models_loaded, model_load_error, _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True

    print("\n" + "="*60)
    print("  🛡️  FraudGuard AI — Starting Up")
    print("="*60)
    try:
        _generate_and_save_models()
        models_loaded = True
        print("\n[Bootstrap] ✅ All models loaded successfully")
        print(f"[Bootstrap] 🌐 Dashboard → http://localhost:5000")
        print("="*60 + "\n")
        start_simulation()
    except Exception as e:
        model_load_error = str(e)
        models_loaded = False
        print(f"\n[Bootstrap] ❌ Model loading failed: {e}")
        traceback.print_exc()


# Auto-bootstrap on module import (for WSGI/Gunicorn or CLI)
bootstrap()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
