import pandas as pd
from datetime import datetime
from typing import Dict, Any
import json
import threading

class CustomerRiskProfiler:
    def __init__(self, storage_path="data/customer_profiles.json"):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        try:
            with open(storage_path, 'r') as f:
                self.profiles = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.profiles = {}
    
    def update_profile(self, customer_id: str, transaction: Dict[str, Any]):
        with self._lock:
            if customer_id not in self.profiles:
                self.profiles[customer_id] = {
                    "first_seen": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat(),
                    "transaction_count": 0,
                    "total_amount": 0,
                    "risk_score": 0.5,  # Default medium risk
                    "behavior_pattern": {},
                    "flags": []
                }
            
            profile = self.profiles[customer_id]
            profile['last_activity'] = datetime.now().isoformat()
            profile['transaction_count'] += 1
            profile['total_amount'] += transaction['amount']
            
            # Update behavior patterns (simplified)
            tx_type = transaction['type']
            profile['behavior_pattern'].setdefault(tx_type, 0)
            profile['behavior_pattern'][tx_type] += 1
            
            # Calculate risk score (simplified)
            amount_deviation = self._calculate_amount_deviation(customer_id, transaction['amount'])
            freq_deviation = self._calculate_frequency_deviation(customer_id)
            
            profile['risk_score'] = min(0.9, 0.3 + amount_deviation * 0.4 + freq_deviation * 0.3)
            
            self._save_profiles()

    def _calculate_amount_deviation(self, customer_id, amount):
        """Calculate deviation from customer's typical transaction amount"""
        profile = self.profiles[customer_id]
        avg_amount = profile['total_amount'] / profile['transaction_count']
        return min(1.0, abs(amount - avg_amount) / (avg_amount + 1e-6))
    
    def _calculate_frequency_deviation(self, customer_id):
        """Calculate deviation from customer's typical transaction frequency"""
        return 0.5  # Placeholder
    
    def _save_profiles(self):
        try:
            data_to_save = dict(self.profiles)
            with open(self.storage_path, 'w') as f:
                json.dump(data_to_save, f)
        except Exception:
            pass
    
    def get_risk_profile(self, customer_id):
        with self._lock:
            return self.profiles.get(customer_id, None)






        