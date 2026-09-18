"""
profiling/builder.py
FraudGuard AI - Thread-Safe Customer Risk Profiler
Computes velocity, deviation, geo-footprint, and spend baseline per customer
"""

import threading
import time
from collections import defaultdict, deque
from typing import Dict, Optional, List
from datetime import datetime, timezone


class CustomerRiskProfile:
    """Immutable snapshot of a customer's risk profile at a point in time."""

    __slots__ = [
        'customer_id', 'baseline_spend', 'amount_deviation_ratio',
        'unique_locations', 'avg_duration', 'transaction_count',
        'recent_velocity', 'max_amount', 'risk_tier', 'last_updated'
    ]

    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.baseline_spend = 0.0
        self.amount_deviation_ratio = 0.0
        self.unique_locations = 0
        self.avg_duration = 0.0
        self.transaction_count = 0
        self.recent_velocity = 0.0         # transactions per minute in last 5 min
        self.max_amount = 0.0
        self.risk_tier = 'LOW'             # LOW / MEDIUM / HIGH / CRITICAL
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            'customer_id': self.customer_id,
            'baseline_spend': round(self.baseline_spend, 2),
            'amount_deviation_ratio': round(self.amount_deviation_ratio, 4),
            'unique_locations': self.unique_locations,
            'avg_duration_seconds': round(self.avg_duration, 1),
            'transaction_count': self.transaction_count,
            'recent_velocity_per_min': round(self.recent_velocity, 3),
            'max_single_transaction': round(self.max_amount, 2),
            'risk_tier': self.risk_tier,
            'last_updated': self.last_updated,
        }


class _CustomerState:
    """Internal mutable state per customer — not exposed directly."""

    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.amounts: deque = deque(maxlen=200)
        self.locations: set = set()
        self.durations: deque = deque(maxlen=200)
        self.timestamps: deque = deque(maxlen=200)   # Unix timestamps
        self.transaction_count: int = 0
        self.max_amount: float = 0.0


class CustomerRiskProfiler:
    """
    Thread-safe customer risk profiler.
    
    Tracks per-customer:
      - Historical baseline spend (rolling mean)
      - Amount deviation ratio (z-score proxy)
      - Unique location count (geo-footprint)
      - Average transaction duration
      - Recent velocity (txn/min over last 5 minutes)
      - Risk tier classification
    
    CRITICAL: Uses threading.Lock() and dict(self._states) copy to prevent
    RuntimeError: dictionary changed size during iteration in concurrent writes.
    """

    VELOCITY_WINDOW_SECONDS = 300  # 5 minutes
    RISK_THRESHOLDS = {
        'deviation_high': 3.0,
        'velocity_high': 5.0,    # >5 txn/min
        'location_high': 5,      # >5 unique cities
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._states: Dict[str, _CustomerState] = {}
        self._profile_cache: Dict[str, CustomerRiskProfile] = {}

    def _get_or_create_state(self, customer_id: str) -> _CustomerState:
        """NOT thread-safe — must be called within lock."""
        if customer_id not in self._states:
            self._states[customer_id] = _CustomerState(customer_id)
        return self._states[customer_id]

    def update(self, transaction: dict) -> CustomerRiskProfile:
        """
        Process a new transaction and update the customer's risk profile.
        Thread-safe — uses internal lock.
        """
        customer_id = str(transaction.get('CustomerID', transaction.get('AccountID', 'UNKNOWN')))
        amount = float(transaction.get('TransactionAmount', 0))
        location = str(transaction.get('Location', 'Unknown'))
        duration = float(transaction.get('TransactionDuration', 60))
        now_ts = time.time()

        with self._lock:
            state = self._get_or_create_state(customer_id)

            # Update rolling state
            state.amounts.append(amount)
            state.locations.add(location)
            state.durations.append(duration)
            state.timestamps.append(now_ts)
            state.transaction_count += 1
            state.max_amount = max(state.max_amount, amount)

            # Compute aggregates
            amounts_list = list(state.amounts)
            baseline = float(sum(amounts_list) / len(amounts_list)) if amounts_list else 0.0

            # Deviation ratio: how much does this txn deviate from baseline?
            if baseline > 0 and len(amounts_list) > 1:
                import statistics
                try:
                    std = statistics.stdev(amounts_list)
                    deviation_ratio = abs(amount - baseline) / (std + 1e-9)
                except Exception:
                    deviation_ratio = 0.0
            else:
                deviation_ratio = 0.0

            avg_duration = float(sum(state.durations) / len(state.durations)) if state.durations else 60.0

            # Recent velocity: count txns in last VELOCITY_WINDOW_SECONDS
            cutoff = now_ts - self.VELOCITY_WINDOW_SECONDS
            recent_txns = sum(1 for ts in state.timestamps if ts >= cutoff)
            velocity = recent_txns / (self.VELOCITY_WINDOW_SECONDS / 60.0)

            # Build profile snapshot
            profile = CustomerRiskProfile(customer_id)
            profile.baseline_spend = baseline
            profile.amount_deviation_ratio = deviation_ratio
            profile.unique_locations = len(state.locations)
            profile.avg_duration = avg_duration
            profile.transaction_count = state.transaction_count
            profile.recent_velocity = velocity
            profile.max_amount = state.max_amount
            profile.risk_tier = self._classify_tier(deviation_ratio, velocity, len(state.locations))
            profile.last_updated = __import__('datetime').datetime.now(
                __import__('datetime').timezone.utc
            ).isoformat()

            # Cache profile (using dict copy to avoid iteration bugs)
            self._profile_cache = dict(self._profile_cache)
            self._profile_cache[customer_id] = profile

        return profile

    def _classify_tier(self, deviation: float, velocity: float, locations: int) -> str:
        """Classify customer risk tier based on profile metrics."""
        score = 0
        if deviation > self.RISK_THRESHOLDS['deviation_high']:
            score += 2
        elif deviation > 1.5:
            score += 1

        if velocity > self.RISK_THRESHOLDS['velocity_high']:
            score += 2
        elif velocity > 2.0:
            score += 1

        if locations > self.RISK_THRESHOLDS['location_high']:
            score += 1

        if score >= 4:
            return 'CRITICAL'
        elif score >= 3:
            return 'HIGH'
        elif score >= 1:
            return 'MEDIUM'
        return 'LOW'

    def get_profile(self, customer_id: str) -> Optional[CustomerRiskProfile]:
        """Return cached profile for a customer (thread-safe read)."""
        with self._lock:
            # Use dict copy to prevent concurrent modification
            cache_copy = dict(self._profile_cache)
        return cache_copy.get(customer_id)

    def get_all_profiles(self) -> List[dict]:
        """Return all customer profiles as list of dicts (thread-safe)."""
        with self._lock:
            cache_copy = dict(self._profile_cache)
        return [p.to_dict() for p in cache_copy.values()]

    def get_customer_ids(self) -> List[str]:
        """Return list of all tracked customer IDs."""
        with self._lock:
            return list(self._states.keys())

    def reset(self) -> None:
        """Clear all state (for testing/resets)."""
        with self._lock:
            self._states.clear()
            self._profile_cache.clear()
