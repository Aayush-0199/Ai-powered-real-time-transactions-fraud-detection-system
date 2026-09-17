"""
Automated Model Retraining & Evaluation Engine
==============================================
Evaluates and benchmarks XGBoost, Random Forest, and Isolation Forest
against live and historical transaction streams.
"""

import os
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class AutoMLTrainer:
    def __init__(self, data_path="data/bank_transactions_data_2.csv", experiment_name="fraud_detection"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            self._init_mlflow()

    def _init_mlflow(self):
        """Initialize MLflow tracking if installed."""
        if not MLFLOW_AVAILABLE:
            return
        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            if not mlflow.get_experiment_by_name(self.experiment_name):
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            self.logger.warning(f"MLflow initialization failed, continuing with local logging: {str(e)}")

    def _generate_fraud_labels(self, df):
        """Generate synthetic fraud labels based on known anomalous transaction patterns."""
        conditions = [
            (df['TransactionAmount'] > 500) & (df['TransactionType'] == 'Debit'),
            (df['LoginAttempts'] > 2),
            (df['Channel'] == 'Online') & (df['TransactionDuration'] < 30),
            (df['Location'].astype(str).str.contains('Foreign|Moscow|Lagos', case=False, na=False))
        ]
        
        fraud_labels = np.zeros(len(df))
        for condition in conditions:
            fraud_labels[condition] = 1
            
        if sum(fraud_labels) == 0:
            fraud_labels[:min(3, len(df))] = 1
            
        return fraud_labels

    def _prepare_data(self, df):
        """Prepare feature matrix and target vector for training."""
        if 'is_fraud' not in df.columns:
            self.logger.warning("'is_fraud' column not found, generating heuristic labels")
            df['is_fraud'] = self._generate_fraud_labels(df)
        
        df['TransactionSpeed'] = df['TransactionAmount'] / df['TransactionDuration'].replace(0, 1)
        
        feature_cols = [
            'TransactionAmount', 'TransactionDuration', 'LoginAttempts',
            'AccountBalance', 'TransactionSpeed', 'CustomerAge'
        ]
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].fillna(0)
        y = df['is_fraud']
        
        return X, y

    def preprocess_data(self):
        """Load and preprocess the transaction dataset."""
        try:
            df = pd.read_csv(self.data_path)
            date_cols = ['TransactionDate', 'PreviousTransactionDate']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            if 'TransactionDate' in df.columns and 'PreviousTransactionDate' in df.columns:
                df['DaysSinceLastTransaction'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.days.fillna(0)
            
            return self._prepare_data(df)
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def train_models(self):
        """Train and evaluate ensemble candidates, saving the highest scoring model."""
        try:
            X, y = self.preprocess_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models = {
                "xgboost": XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss'),
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "isolation_forest": IsolationForest(contamination=float(max(0.01, min(0.2, y_train.mean()))), random_state=42)
            }
            
            best_score = 0
            best_model = None
            best_name = None
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    
                    if name == "isolation_forest":
                        y_pred = -model.decision_function(X_test)
                    else:
                        y_pred = model.predict_proba(X_test)[:, 1]
                    
                    score = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else 0.5
                    self.logger.info(f"Model {name} ROC-AUC: {score:.4f}")
                    
                    if MLFLOW_AVAILABLE:
                        try:
                            with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                                mlflow.log_metric("roc_auc", score)
                        except Exception:
                            pass
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_name = name
                        
                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue
            
            if best_model is None:
                raise RuntimeError("All model training attempts failed")
                
            model_path = f"models/{best_name}.pkl"
            joblib.dump(best_model, model_path)
            self.logger.info(f"Saved best model ({best_name}) to {model_path} with score {best_score:.4f}")
            
            return best_model, best_score
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def retrain_schedule(self, interval_days=7):
        """Schedule periodic retraining in the background."""
        while True:
            try:
                self.logger.info("Starting scheduled retraining cycle...")
                self.train_models()
                self.logger.info(f"Retraining complete. Next evaluation in {interval_days} days.")
            except Exception as e:
                self.logger.error(f"Retraining failed: {str(e)}")
            finally:
                time.sleep(interval_days * 24 * 60 * 60)