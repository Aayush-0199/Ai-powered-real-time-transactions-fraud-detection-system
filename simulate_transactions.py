"""
Real-Time Transaction Simulator
================================
Simulates live transaction streaming to the Fraud Detection Dashboard
without requiring Kafka or Docker. Streams transactions directly via HTTP
to the running Flask app at http://localhost:5000/api/analyze
"""

import requests
import pandas as pd
import random
import time
import json
import sys
import os
from datetime import datetime, timedelta

# ANSI Color Codes
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
BG_RED  = "\033[41m"
BG_GREEN= "\033[42m"

# Config
FLASK_URL      = "http://localhost:5000/api/analyze"
DATA_FILE      = "data/bank_transactions_data_2.csv"
INTERVAL_SECS  = 1.5
BURST_CHANCE   = 0.07
FRAUD_BOOST    = 0.15

FRAUD_PATTERNS = [
    {
        "name": "High-Value Rapid Transfer",
        "overrides": {
            "TransactionAmount": lambda: round(random.uniform(9500, 50000), 2),
            "LoginAttempts": lambda: random.randint(5, 10),
            "TransactionDuration": lambda: random.randint(2, 8),
            "Channel": "Online",
        }
    },
    {
        "name": "Suspicious Location Hop",
        "overrides": {
            "Location": lambda: random.choice(["Moscow", "Lagos", "Pyongyang", "Tehran"]),
            "TransactionAmount": lambda: round(random.uniform(3000, 15000), 2),
            "LoginAttempts": lambda: random.randint(3, 8),
            "Channel": "Online",
        }
    },
    {
        "name": "Repeated Low-Value Probing",
        "overrides": {
            "TransactionAmount": lambda: round(random.uniform(0.01, 1.99), 2),
            "LoginAttempts": lambda: random.randint(4, 9),
            "TransactionDuration": lambda: random.randint(1, 5),
        }
    },
    {
        "name": "Odd-Hours High Withdrawal",
        "overrides": {
            "TransactionAmount": lambda: round(random.uniform(2000, 8000), 2),
            "AccountBalance": lambda: round(random.uniform(50, 200), 2),
            "LoginAttempts": lambda: random.randint(3, 7),
        }
    },
]

stats = {
    "total": 0,
    "fraud": 0,
    "normal": 0,
    "errors": 0,
    "injected_patterns": 0,
    "start_time": datetime.now(),
}


def print_banner():
    os.system("cls" if os.name == "nt" else "clear")
    print("""
\033[1m\033[96m+==================================================================+
|     AI-POWERED FRAUD DETECTION -- LIVE TRANSACTION STREAM       |
|        Streaming to: http://localhost:5000                       |
|        Press Ctrl+C to stop at any time.                        |
+==================================================================+\033[0m
""")


def print_stats():
    elapsed = max((datetime.now() - stats["start_time"]).seconds, 1)
    rate = stats["total"] / elapsed
    fraud_pct = (stats["fraud"] / stats["total"] * 100) if stats["total"] > 0 else 0
    print(f"\n\033[1m\033[94m{'='*80}\033[0m")
    print(f"  \033[97mTotal: {stats['total']}\033[0m  "
          f"\033[92mNormal: {stats['normal']}\033[0m  "
          f"\033[91mFraud: {stats['fraud']} ({fraud_pct:.1f}%)\033[0m  "
          f"\033[93mErrors: {stats['errors']}\033[0m  "
          f"\033[2mRate: {rate:.2f} txn/sec  Elapsed: {elapsed}s\033[0m")
    print(f"  \033[96mInjected Fraud Patterns: {stats['injected_patterns']}\033[0m")
    print(f"\033[1m\033[94m{'='*80}\033[0m\n")


def build_payload(row, fraud_pattern=None):
    now = datetime.now()
    prev = now - timedelta(days=random.randint(1, 30))
    payload = {
        "TransactionAmount": float(row.get("TransactionAmount", 100.0)),
        "TransactionDuration": float(row.get("TransactionDuration", 60.0)),
        "LoginAttempts": int(row.get("LoginAttempts", 1)),
        "AccountBalance": float(row.get("AccountBalance", 5000.0)),
        "TransactionDate": now.strftime("%Y-%m-%d %H:%M:%S"),
        "PreviousTransactionDate": prev.strftime("%Y-%m-%d %H:%M:%S"),
        "TransactionType": str(row.get("TransactionType", "Debit")),
        "Location": str(row.get("Location", "New York")),
        "DeviceID": str(row.get("DeviceID", "DVC0001")),
        "MerchantID": str(row.get("MerchantID", "MRCH001")),
        "Channel": str(row.get("Channel", "Online")),
        "CustomerOccupation": str(row.get("CustomerOccupation", "Engineer")),
        "AccountID": str(row.get("AccountID", "AC00001")),
    }
    if fraud_pattern:
        for key, val_fn in fraud_pattern["overrides"].items():
            payload[key] = val_fn() if callable(val_fn) else val_fn
    return payload


def send_transaction(payload, fraud_pattern_name=None):
    stats["total"] += 1
    txn_num = stats["total"]
    try:
        resp = requests.post(FLASK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        fraud_score = result.get("fraud_probability", result.get("risk_score", 0))
        is_fraud    = result.get("is_fraud", fraud_score > 0.5)
        explanation = result.get("explanation", [])
        top_feature = explanation[0].get("feature", "N/A") if explanation else "N/A"

        if is_fraud:
            stats["fraud"] += 1
            flag = "\033[41m\033[97m\033[1m FRAUD \033[0m"
            score_color = "\033[91m"
        else:
            stats["normal"] += 1
            flag = "\033[42m\033[97m\033[1m LEGIT \033[0m"
            score_color = "\033[92m"

        pattern_tag = f"  \033[95m[{fraud_pattern_name}]\033[0m" if fraud_pattern_name else ""
        print(
            f"  \033[2m#{txn_num:>4}\033[0m  {flag}  "
            f"\033[97m${payload['TransactionAmount']:>10.2f}\033[0m  "
            f"Risk:{score_color}\033[1m{fraud_score:.3f}\033[0m  "
            f"\033[96m{payload['AccountID']:<9}\033[0m"
            f"\033[93m{payload['Location']:<16}\033[0m  "
            f"\033[2mtop:{top_feature}\033[0m"
            f"{pattern_tag}"
        )
    except requests.exceptions.ConnectionError:
        stats["errors"] += 1
        print(f"  \033[91m#{txn_num:>4}  ERROR: Cannot connect to {FLASK_URL} -- is the app running?\033[0m")
    except Exception as e:
        stats["errors"] += 1
        print(f"  \033[91m#{txn_num:>4}  ERROR: {e}\033[0m")


def main():
    print_banner()
    if not os.path.exists(DATA_FILE):
        print(f"\033[91mERROR: Cannot find {DATA_FILE}. Run from the project root.\033[0m")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE).dropna(subset=["TransactionAmount", "AccountID"])
    total_rows = len(df)
    print(f"  \033[92mLoaded {total_rows} real transactions from dataset.\033[0m")
    print(f"  \033[96mStreaming at {1/INTERVAL_SECS:.1f} txn/sec with {FRAUD_BOOST*100:.0f}% fraud injection.\033[0m\n")
    print(f"\033[1m\033[94m{'─'*80}\033[0m")
    print(f"  \033[2m{'#':>4}  {'STATUS':<8} {'AMOUNT':>12}  {'RISK':>7}  {'ACCOUNT':<9}{'LOCATION':<16}  TOP FEATURE\033[0m")
    print(f"\033[1m\033[94m{'─'*80}\033[0m")

    idx = 0
    try:
        while True:
            row = df.iloc[idx % total_rows].to_dict()
            idx += 1
            fraud_pattern = None
            if random.random() < FRAUD_BOOST:
                fraud_pattern = random.choice(FRAUD_PATTERNS)
                stats["injected_patterns"] += 1
            send_transaction(build_payload(row, fraud_pattern),
                             fraud_pattern["name"] if fraud_pattern else None)

            if stats["total"] % 20 == 0:
                print_stats()

            if random.random() < BURST_CHANCE:
                burst = random.choice(FRAUD_PATTERNS)
                count = random.randint(3, 5)
                print(f"\n  \033[95m\033[1m*** BURST ATTACK: {burst['name']} x{count} ***\033[0m")
                for _ in range(count):
                    r = df.sample(1).iloc[0].to_dict()
                    send_transaction(build_payload(r, burst), burst["name"])
                    stats["injected_patterns"] += 1
                    time.sleep(0.3)
                print()

            time.sleep(INTERVAL_SECS)

    except KeyboardInterrupt:
        print(f"\n\n\033[1m\033[93mSimulation stopped by user.\033[0m")
        print_stats()
        elapsed = (datetime.now() - stats["start_time"]).seconds
        print(f"  \033[96mSession: {elapsed}s -- Check your dashboard at http://localhost:5000\033[0m\n")


if __name__ == "__main__":
    main()
