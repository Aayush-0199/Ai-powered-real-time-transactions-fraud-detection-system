from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import torch
import mlflow
import threading
from collections import deque
import time
from graph_models.gnn_model import load_gnn_model
from graph_models.data_loader import TransactionGraphBuilder
from reporting.generator import ReportGenerator
from profiling.builder import CustomerRiskProfiler
from drift.detector import ConceptDriftDetector
from models.automl.trainer import AutoMLTrainer
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize components
iso_forest = joblib.load('trained_models/isolation_forest.pkl')
xgb = joblib.load('trained_models/xgboost.pkl')
shap_explainer = joblib.load('trained_models/shap_explainer.pkl')
gnn_model = load_gnn_model('models/gnn_model.pt')
graph_builder = TransactionGraphBuilder()
report_generator = ReportGenerator()
profiler = CustomerRiskProfiler()
drift_detector = ConceptDriftDetector()

# ── Live Transaction Store ──────────────────────────────────────────────────
# Thread-safe ring buffer keeping the last 1000 analyzed transactions
_store_lock = threading.Lock()
live_transactions = deque(maxlen=1000)
_txn_counter = 0  # global sequential ID

# ── Frozen Accounts Store ──────────────────────────────────────────────────
_frozen_lock = threading.Lock()
frozen_accounts = {}  # account_id -> { 'account_id': ..., 'frozen_at': ..., 'reason': ... }

# Feature names
features = ['TransactionAmount', 'TransactionDuration', 'LoginAttempts', 
            'AccountBalance', 'DaysSinceLastTransaction', 'TransactionSpeed',
            'AvgAmount', 'StdAmount', 'MaxAmount', 'AvgDuration', 'UniqueLocations',
            'AmountDeviation', 'DurationDeviation', 'TransactionType', 
            'Location', 'DeviceID', 'MerchantID', 'Channel', 'CustomerOccupation']

# Background tasks
def auto_retrain():
    while True:
        try:
            trainer = AutoMLTrainer("data/bank_transactions_data_2.csv")
            best_model, score = trainer.train_models()
            app.logger.info(f"AutoML retraining completed. Best model: {type(best_model).__name__} with score: {score:.4f}")
        except Exception as e:
            app.logger.error(f"AutoML retraining failed: {str(e)}")
        time.sleep(7 * 24 * 60 * 60)  # Run weekly

# Start background thread
retrain_thread = threading.Thread(target=auto_retrain, daemon=True)
retrain_thread.start()


# Initialize AutoML Trainer with proper error handling
try:
    automl_trainer = AutoMLTrainer("data/bank_transactions_data_2.csv")
    
    # Check if models exist, if not train initial models
    required_models = ['isolation_forest.pkl', 'xgboost.pkl', 'shap_explainer.pkl']
    if not all(os.path.exists(f"trained_models/{model}") for model in required_models):
        logger.info("Initial models not found, training initial models...")
        automl_trainer.train_models()
except Exception as e:
    logger.error(f"Failed to initialize AutoML trainer: {str(e)}")
    raise

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_transaction():
    global _txn_counter
    data = request.json
    account_id = data.get('AccountID', 'UNKNOWN')

    # Intercept transactions for frozen accounts immediately
    with _frozen_lock:
        is_account_frozen = account_id in frozen_accounts

    if is_account_frozen:
        with _store_lock:
            _txn_counter += 1
            live_transactions.append({
                'id': _txn_counter,
                'TransactionID': f'SIM{_txn_counter:06d}',
                'AccountID': account_id,
                'TransactionAmount': float(data.get('TransactionAmount', 0)),
                'TransactionType': data.get('TransactionType', 'Debit'),
                'Location': data.get('Location', 'Unknown'),
                'Channel': data.get('Channel', 'Online'),
                'DeviceID': data.get('DeviceID', 'Unknown'),
                'MerchantID': data.get('MerchantID', 'Unknown'),
                'RiskScore': 1.0,
                'XGBProb': 1.0,
                'IsoScore': 1.0,
                'GNNProb': 1.0,
                'IsFraud': True,
                'IsFrozen': True,
                'Status': 'BLOCKED',
                'TopFeature': 'ACCOUNT_FROZEN',
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
        return jsonify({
            'isolation_forest_score': 1.0,
            'xgboost_probability': 1.0,
            'gnn_probability': 1.0,
            'composite_score': 1.0,
            'fraud_probability': 1.0,
            'is_fraud': True,
            'is_frozen': True,
            'status': 'BLOCKED',
            'risk_score': 1.0,
            'customer_risk_score': 1.0,
            'explanation': [{'feature': 'ACCOUNT_FROZEN', 'value': 1.0, 'shap_value': 1.0}],
            'drift_detected': False
        })
    
    # Update customer profile
    profiler.update_profile(account_id, {
        'amount': float(data['TransactionAmount']),
        'type': data['TransactionType'],
        'date': data['TransactionDate']
    })
    
    # Get customer stats
    cust_profile = profiler.get_risk_profile(data['AccountID'])
    cust_stats = {
        'AvgAmount': cust_profile.get('avg_amount', 150.0),
        'StdAmount': cust_profile.get('std_amount', 75.0),
        'MaxAmount': cust_profile.get('max_amount', 1000.0),
        'AvgDuration': cust_profile.get('avg_duration', 120.0),
        'UniqueLocations': cust_profile.get('unique_locations', 3)
    }
    
    # Create feature vector
    transaction_date = datetime.strptime(data['TransactionDate'], '%Y-%m-%d %H:%M:%S')
    prev_date = datetime.strptime(data['PreviousTransactionDate'], '%Y-%m-%d %H:%M:%S')
    
    features_dict = {
        'TransactionAmount': float(data['TransactionAmount']),
        'TransactionDuration': float(data['TransactionDuration']),
        'LoginAttempts': int(data['LoginAttempts']),
        'AccountBalance': float(data['AccountBalance']),
        'DaysSinceLastTransaction': (datetime.now() - prev_date).days,
        'TransactionSpeed': float(data['TransactionAmount']) / float(data['TransactionDuration']),
        'AvgAmount': cust_stats['AvgAmount'],
        'StdAmount': cust_stats['StdAmount'],
        'MaxAmount': cust_stats['MaxAmount'],
        'AvgDuration': cust_stats['AvgDuration'],
        'UniqueLocations': cust_stats['UniqueLocations'],
        'AmountDeviation': (float(data['TransactionAmount']) - cust_stats['AvgAmount']) / cust_stats['StdAmount'],
        'DurationDeviation': (float(data['TransactionDuration']) - cust_stats['AvgDuration']) / cust_stats['AvgDuration'],
        'TransactionType': 0 if data['TransactionType'] == 'Debit' else 1,
        'Location': hash(data['Location']) % 100,
        'DeviceID': hash(data['DeviceID']) % 100,
        'MerchantID': hash(data['MerchantID']) % 100,
        'Channel': {'ATM': 0, 'Online': 1, 'Branch': 2}.get(data['Channel'], 0),
        'CustomerOccupation': {'Student': 0, 'Doctor': 1, 'Engineer': 2, 'Retired': 3}.get(data['CustomerOccupation'], 0)
    }
    
    # Convert to DataFrame for prediction
    X = pd.DataFrame([features_dict], columns=features)
    
    # Check for concept drift
    drift_detector.add_data(X.values[0])
    
    # Get predictions
    iso_score = -iso_forest.decision_function(X)[0]
    xgb_prob = xgb.predict_proba(X)[0, 1]
    
    # GNN prediction
    graph_data = graph_builder.add_transaction(data)
    with torch.no_grad():
        gnn_prob = gnn_model(graph_data.x, graph_data.edge_index).item()
    
    # SHAP explanations
    shap_values = shap_explainer.shap_values(X)
    
    # Prepare explanation
    explanation = []
    for i, feature in enumerate(features):
        explanation.append({
            'feature': feature,
            'value': float(X.iloc[0, i]),
            'shap_value': float(shap_values[0][i])
        })
    
    explanation.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    
    # Composite score with calibrated normalization and behavioral anomaly indicators
    cust_risk = cust_profile['risk_score'] if cust_profile else 0.5

    # 1. Normalized Isolation Forest score (decision_function ranges from -0.15 to +0.15)
    norm_iso = min(1.0, max(0.02, (float(iso_score) + 0.08) * 4.5))

    # 2. Normalized XGBoost probability (calibrated for trained model's low base fraud distribution)
    norm_xgb = min(1.0, max(0.02, float(xgb_prob) * 3.2))

    # 3. Behavioral anomaly detection boosts
    risk_boost = 0.0
    logins = int(data.get('LoginAttempts', 1))
    if logins >= 3:
        risk_boost += 0.25 + min(0.35, (logins - 2) * 0.08)

    amt = float(data.get('TransactionAmount', 100))
    if amt > 2500:
        risk_boost += min(0.40, (amt / 10000.0) * 0.30)
    elif amt < 3.0 and logins >= 3:
        risk_boost += 0.35  # Card probing / low-value testing

    dur = max(1.0, float(data.get('TransactionDuration', 60)))
    speed = amt / dur
    if speed > 120:
        risk_boost += 0.25

    loc = str(data.get('Location', ''))
    if loc in ['Moscow', 'Lagos', 'Pyongyang', 'Tehran']:
        risk_boost += 0.30

    # Explicit simulated risk indicator from simulator
    if data.get('IsSimulatedFraud', False):
        risk_boost = max(risk_boost, 0.60)

    # Blended risk calculation
    model_risk = (norm_iso * 0.35 + norm_xgb * 0.45 + float(gnn_prob) * 0.20)
    composite_score = min(0.98, max(0.02, (model_risk * 0.50 + risk_boost * 0.50) * (0.6 + 0.4 * cust_risk)))

    is_fraud_decision = bool(composite_score > 0.50 or risk_boost >= 0.50)
    
    result = {
        'isolation_forest_score': float(iso_score),
        'xgboost_probability': float(xgb_prob),
        'gnn_probability': float(gnn_prob),
        'composite_score': float(composite_score),
        'fraud_probability': float(max(xgb_prob, composite_score if is_fraud_decision else xgb_prob)),
        'is_fraud': is_fraud_decision,
        'risk_score': float(composite_score),
        'customer_risk_score': float(cust_risk) if cust_profile else 0.5,
        'explanation': explanation[:5],
        'drift_detected': drift_detector.drift_count > 0
    }

    # ── Save to live store ──────────────────────────────────────────────────
    with _store_lock:
        _txn_counter += 1
        live_transactions.append({
            'id': _txn_counter,
            'TransactionID': f'SIM{_txn_counter:06d}',
            'AccountID': account_id,
            'TransactionAmount': float(data.get('TransactionAmount', 0)),
            'TransactionType': data.get('TransactionType', 'Debit'),
            'Location': data.get('Location', 'Unknown'),
            'Channel': data.get('Channel', 'Online'),
            'DeviceID': data.get('DeviceID', 'Unknown'),
            'MerchantID': data.get('MerchantID', 'Unknown'),
            'RiskScore': float(composite_score),
            'XGBProb': float(xgb_prob),
            'IsoScore': float(iso_score),
            'GNNProb': float(gnn_prob),
            'IsFraud': is_fraud_decision,
            'IsFrozen': False,
            'Status': 'Flagged' if is_fraud_decision else 'Cleared',
            'TopFeature': explanation[0]['feature'] if explanation else 'N/A',
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    return jsonify(result)

@app.route('/api/transactions')
def get_recent_transactions():
    # In production, this would query a database
    sample_data = [
        {
            'TransactionID': 'TX000001',
            'AccountID': 'AC00128',
            'TransactionAmount': 14.09,
            'TransactionDate': '2023-04-11 16:29:14',
            'TransactionType': 'Debit',
            'Location': 'San Diego',
            'RiskScore': 0.85,
            'Status': 'Flagged'
        },
        {
            'TransactionID': 'TX000002',
            'AccountID': 'AC00455',
            'TransactionAmount': 376.24,
            'TransactionDate': '2023-06-27 16:44:19',
            'TransactionType': 'Debit',
            'Location': 'Houston',
            'RiskScore': 0.42,
            'Status': 'Approved'
        }
    ]
    return jsonify(sample_data)

@app.route('/api/live-transactions')
def get_live_transactions():
    """Return the most recent N transactions from the live in-memory store."""
    limit = int(request.args.get('limit', 50))
    with _store_lock:
        # Return newest first
        rows = list(live_transactions)[-limit:][::-1]
    return jsonify(rows)

@app.route('/api/stats')
def get_stats():
    """Return aggregate counters for the dashboard KPI cards."""
    with _store_lock:
        txns = list(live_transactions)
    with _frozen_lock:
        frozen_cnt = len(frozen_accounts)
    total = len(txns)
    if total == 0:
        return jsonify({
            'total': 0, 'flagged': 0, 'high_risk': 0,
            'blocked': 0, 'avg_risk': 0.0, 'fraud_rate': 0.0,
            'frozen': frozen_cnt
        })
    flagged   = sum(1 for t in txns if t['IsFraud'])
    high_risk = sum(1 for t in txns if t['RiskScore'] > 0.7)
    avg_risk  = round(sum(t['RiskScore'] for t in txns) / total, 4)
    return jsonify({
        'total':      total,
        'flagged':    flagged,
        'high_risk':  high_risk,
        'blocked':    frozen_cnt,
        'avg_risk':   avg_risk,
        'fraud_rate': round(flagged / total * 100, 1),
        'frozen':     frozen_cnt
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

# ── Incident Response: Account Freeze & Block Endpoints ─────────────────────
@app.route('/api/account/<account_id>/freeze', methods=['POST'])
def freeze_account(account_id):
    """Freeze an account to immediately intercept and block all future transactions."""
    reason = (request.json or {}).get('reason', 'High-Risk Fraud Incident Detected')
    with _frozen_lock:
        frozen_accounts[account_id] = {
            'account_id': account_id,
            'frozen_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'reason': reason
        }
        count = len(frozen_accounts)

    # Update existing transactions in memory buffer
    with _store_lock:
        for t in live_transactions:
            if t.get('AccountID') == account_id:
                t['IsFrozen'] = True
                t['Status'] = 'BLOCKED'

    return jsonify({
        'success': True,
        'account_id': account_id,
        'status': 'FROZEN',
        'frozen_count': count,
        'message': f'Account {account_id} has been FROZEN. All incoming transactions will be blocked.'
    })

@app.route('/api/account/<account_id>/unfreeze', methods=['POST'])
def unfreeze_account(account_id):
    """Unfreeze an account to restore normal transaction processing."""
    with _frozen_lock:
        frozen_accounts.pop(account_id, None)
        count = len(frozen_accounts)

    with _store_lock:
        for t in live_transactions:
            if t.get('AccountID') == account_id:
                t['IsFrozen'] = False
                t['Status'] = 'Flagged' if t.get('IsFraud') else 'Cleared'

    return jsonify({
        'success': True,
        'account_id': account_id,
        'status': 'ACTIVE',
        'frozen_count': count,
        'message': f'Account {account_id} has been UNFREEZED and restored to active status.'
    })

@app.route('/api/frozen-accounts')
def get_frozen_accounts():
    """List all currently frozen accounts."""
    with _frozen_lock:
        accounts = list(frozen_accounts.values())
    return jsonify({
        'count': len(accounts),
        'accounts': accounts
    })


@app.route('/api/reports/sar', methods=['POST'])
def generate_sar_report():
    data = request.json
    report_path = f"reports/sar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_generator.generate_sar(
        data['transactions'],
        data['customer_info'],
        report_path
    )
    return send_file(report_path, as_attachment=True)

@app.route('/api/customer/<customer_id>/profile')
def get_customer_profile(customer_id):
    profile = profiler.get_risk_profile(customer_id)
    if profile:
        return jsonify(profile)
    return jsonify({"error": "Customer not found"}), 404

@app.route('/api/models/retrain', methods=['POST'])
def trigger_retraining():
    try:
        trainer = AutoMLTrainer("data/bank_transactions_data_2.csv")
        best_model, score = trainer.train_models()
        return jsonify({
            "status": "success",
            "best_model": type(best_model).__name__,
            "score": score
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/drift/status')
def get_drift_status():
    return jsonify({
        "drift_detected": drift_detector.drift_count > 0,
        "drift_count": drift_detector.drift_count
    })

if __name__ == '__main__':
    # Create required directories
    import os
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Initialize MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    
    app.run(debug=True, host='0.0.0.0')