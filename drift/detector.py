"""
drift/detector.py
FraudGuard AI - Concept Drift Detector
Uses two-sample Kolmogorov-Smirnov test to detect statistical drift in transaction distributions
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
from scipy import stats
import numpy as np


class DriftWindow:
    """Maintains a sliding window of samples for KS-test comparison."""

    def __init__(self, name: str, window_size: int = 200):
        self.name = name
        self.window_size = window_size
        self._samples: deque = deque(maxlen=window_size)

    def add(self, value: float) -> None:
        self._samples.append(float(value))

    def get_samples(self) -> List[float]:
        return list(self._samples)

    def is_ready(self, min_samples: int = 50) -> bool:
        return len(self._samples) >= min_samples

    def __len__(self):
        return len(self._samples)


class DriftReport:
    """Result container for a single drift check."""

    def __init__(self):
        self.feature: str = ''
        self.ks_statistic: float = 0.0
        self.p_value: float = 1.0
        self.drift_detected: bool = False
        self.severity: str = 'NONE'        # NONE / MILD / MODERATE / SEVERE
        self.reference_mean: float = 0.0
        self.current_mean: float = 0.0
        self.mean_shift_pct: float = 0.0
        self.timestamp: str = ''

    def to_dict(self) -> dict:
        return {
            'feature': self.feature,
            'ks_statistic': round(self.ks_statistic, 4),
            'p_value': round(self.p_value, 6),
            'drift_detected': self.drift_detected,
            'severity': self.severity,
            'reference_mean': round(self.reference_mean, 2),
            'current_mean': round(self.current_mean, 2),
            'mean_shift_pct': round(self.mean_shift_pct, 2),
            'timestamp': self.timestamp,
        }


class ConceptDriftDetector:
    """
    Streaming concept drift detector using the Kolmogorov-Smirnov (KS) two-sample test.
    
    Maintains two windows per feature:
      - Reference window: "normal" baseline distribution (older data)
      - Current window: recent streaming data
    
    KS-test compares CDFs — if p-value < threshold, drift is detected.
    
    Monitored features:
      - TransactionAmount
      - AccountBalance
      - LoginAttempts
      - TransactionDuration
    """

    MONITORED_FEATURES = [
        'TransactionAmount',
        'AccountBalance',
        'LoginAttempts',
        'TransactionDuration',
    ]
    SIGNIFICANCE_LEVEL = 0.05      # p < 0.05 → drift detected
    MIN_SAMPLES = 50               # Minimum samples before testing
    REFERENCE_WINDOW_SIZE = 300    # Historical baseline window
    CURRENT_WINDOW_SIZE = 100      # Recent streaming window

    def __init__(self):
        self._lock = threading.Lock()
        self._reference_windows: Dict[str, DriftWindow] = {}
        self._current_windows: Dict[str, DriftWindow] = {}
        self._drift_history: deque = deque(maxlen=100)
        self._total_samples: int = 0
        self._drift_count: int = 0

        # Initialize windows for each monitored feature
        for feature in self.MONITORED_FEATURES:
            self._reference_windows[feature] = DriftWindow(
                f"ref_{feature}", self.REFERENCE_WINDOW_SIZE
            )
            self._current_windows[feature] = DriftWindow(
                f"cur_{feature}", self.CURRENT_WINDOW_SIZE
            )

    def add_sample(self, transaction: dict) -> None:
        """
        Ingest a new transaction sample into both windows.
        Thread-safe.
        """
        with self._lock:
            self._total_samples += 1
            for feature in self.MONITORED_FEATURES:
                value = float(transaction.get(feature, 0))
                self._reference_windows[feature].add(value)
                self._current_windows[feature].add(value)

    def check_drift(self, feature: Optional[str] = None) -> List[DriftReport]:
        """
        Run KS-test on one or all monitored features.
        Returns list of DriftReport objects.
        Thread-safe.
        """
        reports = []
        features_to_check = [feature] if feature else self.MONITORED_FEATURES

        with self._lock:
            for feat in features_to_check:
                if feat not in self.MONITORED_FEATURES:
                    continue

                ref_win = self._reference_windows[feat]
                cur_win = self._current_windows[feat]

                report = DriftReport()
                report.feature = feat
                report.timestamp = __import__('datetime').datetime.now(
                    __import__('datetime').timezone.utc
                ).isoformat()

                # Need sufficient samples in both windows
                if not ref_win.is_ready(self.MIN_SAMPLES) or not cur_win.is_ready(30):
                    report.drift_detected = False
                    report.severity = 'INSUFFICIENT_DATA'
                    reports.append(report)
                    continue

                ref_samples = ref_win.get_samples()
                cur_samples = cur_win.get_samples()

                # Two-sample KS test
                try:
                    ks_stat, p_value = stats.ks_2samp(ref_samples, cur_samples)
                except Exception as e:
                    report.drift_detected = False
                    report.severity = 'ERROR'
                    reports.append(report)
                    continue

                ref_mean = float(np.mean(ref_samples))
                cur_mean = float(np.mean(cur_samples))
                mean_shift = ((cur_mean - ref_mean) / (ref_mean + 1e-9)) * 100.0

                drift_detected = p_value < self.SIGNIFICANCE_LEVEL

                # Severity classification
                severity = 'NONE'
                if drift_detected:
                    if ks_stat > 0.5 or p_value < 0.001:
                        severity = 'SEVERE'
                    elif ks_stat > 0.3 or p_value < 0.01:
                        severity = 'MODERATE'
                    else:
                        severity = 'MILD'
                    self._drift_count += 1

                report.ks_statistic = ks_stat
                report.p_value = p_value
                report.drift_detected = drift_detected
                report.severity = severity
                report.reference_mean = ref_mean
                report.current_mean = cur_mean
                report.mean_shift_pct = mean_shift

                reports.append(report)

                # Log to history
                if drift_detected:
                    self._drift_history.append(report.to_dict())

        return reports

    def get_status(self) -> dict:
        """Return overall drift detector status summary."""
        with self._lock:
            all_reports = []
            for feat in self.MONITORED_FEATURES:
                ref_win = self._reference_windows[feat]
                cur_win = self._current_windows[feat]
                all_reports.append({
                    'feature': feat,
                    'reference_samples': len(ref_win),
                    'current_samples': len(cur_win),
                    'ready': ref_win.is_ready(self.MIN_SAMPLES),
                })

            return {
                'total_samples_processed': self._total_samples,
                'drift_events_detected': self._drift_count,
                'monitored_features': all_reports,
                'recent_drift_history': list(self._drift_history)[-5:],
            }

    def reset_reference(self, feature: Optional[str] = None) -> None:
        """
        Reset reference window (simulate model update / concept reset).
        Copies current window to reference.
        """
        with self._lock:
            features = [feature] if feature else self.MONITORED_FEATURES
            for feat in features:
                if feat in self._current_windows:
                    cur_samples = self._current_windows[feat].get_samples()
                    self._reference_windows[feat] = DriftWindow(
                        f"ref_{feat}", self.REFERENCE_WINDOW_SIZE
                    )
                    for s in cur_samples:
                        self._reference_windows[feat].add(s)

    def reset(self) -> None:
        """
        Full reset — clears all windows and resets counters.
        Used by /api/reset to wipe all streaming state.
        """
        with self._lock:
            self._total_samples = 0
            self._drift_count   = 0
            self._drift_history.clear()
            for feature in self.MONITORED_FEATURES:
                self._reference_windows[feature] = DriftWindow(
                    f"ref_{feature}", self.REFERENCE_WINDOW_SIZE
                )
                self._current_windows[feature] = DriftWindow(
                    f"cur_{feature}", self.CURRENT_WINDOW_SIZE
                )
