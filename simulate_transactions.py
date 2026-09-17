"""
Real-Time Transaction Simulator
================================
Simulates live transaction streaming to the Fraud Detection Dashboard
without requiring Kafka or Docker. Streams transactions directly via HTTP
to the running Flask app at http://localhost:5000/api/analyze

Calibration Rule:
In each window of 10 transactions, between 2 and 5 transactions (at least 2, at most 5)
are guaranteed to be risky/fraudulent with realistic anomaly signatures.
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
BG_BLUE = "\033[44m"

# Config
FLASK_URL      = "http://localhost:5000/api/analyze"
DATA_FILE      = "data/bank_transactions_data_2.csv"
INTERVAL_SECS  = 1.2
BATCH_SIZE     = 10   # Evaluate in calibrated windows of 10
MIN_RISKY      = 2    # At least 2 risky per 10
MAX_RISKY      = 5    # At most 5 risky per 10

FRAUD_PATTERNS = [
    {
        "name": "High-Value Rapid Wire",
        "overrides": {
            "TransactionAmount": lambda: round(random.uniform(9500, 48000), 2),
            "LoginAttempts": lambda: random.randint(4, 9),
            "TransactionDuration": lambda: random.randint(2, 7),
            "Channel": "Online",
            "TransactionType": "Debit",
        }
    },
    {
        "name": "Suspicious Location Hop",
        "overrides": {
            "Location": lambda: random.choice(["Moscow", "Lagos", "Pyongyang", "Tehran"]),
            "TransactionAmount": lambda: round(random.uniform(4500, 18000), 2),
            "LoginAttempts": lambda: random.randint(3, 7),
            "TransactionDuration": lambda: random.randint(5, 15),
            "Channel": "Online",
        }
    },
    {
        "name": "Account Takeover / Balance Drain",
        "overrides": {
            "TransactionAmount": lambda: round(random.uniform(6000, 22000), 2),
            "AccountBalance": lambda: round(random.uniform(25, 120), 2),
            "LoginAttempts": lambda: random.randint(5, 10),
            "TransactionDuration": lambda: random.randint(3, 9),
            "Channel": "Online",
        }
    },
    {
        "name": "Automated Velocity Card Probing",
        "overrides": {
            "TransactionAmount": lambda: round(random.uniform(0.50, 2.99), 2),
            "LoginAttempts": lambda: random.randint(5, 11),
            "TransactionDuration": lambda: random.randint(1, 4),
            "Channel": "Online",
        }
    },
    {
        "name": "Shared Device Collusion Ring",
        "overrides": {
            "DeviceID": lambda: random.choice(["DVC_SUSP_01", "DVC_SUSP_02", "DVC_COLLUDE_X"]),
            "TransactionAmount": lambda: round(random.uniform(5500, 16000), 2),
            "LoginAttempts": lambda: random.randint(4, 8),
            "Channel": "Online",
        }
    }
]

stats = {
    "total": 0,
    "fraud": 0,
    "normal": 0,
    "errors": 0,
    "injected_patterns": 0,
    "batch_count": 0,
    "start_time": datetime.now(),
}


def print_banner():
    os.system("cls" if os.name == "nt" else "clear")
    print(f"""
{BOLD}{CYAN}+==================================================================+
|     AI-POWERED FRAUD DETECTION -- REAL-TIME RISK STREAM          |
|        Target: http://localhost:5000                             |
|        Calibration: 2 to 5 Risky Transactions Per 10 Batch       |
|        Press Ctrl+C to stop at any time.                        |
+==================================================================+{RESET}
""")


def print_stats():
    elapsed = max((datetime.now() - stats["start_time"]).seconds, 1)
    rate = stats["total"] / elapsed
    fraud_pct = (stats["fraud"] / stats["total"] * 100) if stats["total"] > 0 else 0
    print(f"\n{BOLD}{BLUE}{'='*82}{RESET}")
    print(f"  {WHITE}Total Streamed: {stats['total']}{RESET}  "
          f"{GREEN}Legit: {stats['normal']}{RESET}  "
          f"{RED}Risky/Fraud: {stats['fraud']} ({fraud_pct:.1f}%){RESET}  "
          f"{YELLOW}Batches Completed: {stats['batch_count']}{RESET}  "
          f"{DIM}Rate: {rate:.2f} txn/s | Elapsed: {elapsed}s{RESET}")
    print(f"{BOLD}{BLUE}{'='*82}{RESET}\n")


def build_payload(row, fraud_pattern=None, is_risky=False):
    now = datetime.now()
    prev = now - timedelta(days=random.randint(1, 28))
    
    if is_risky:
        # Generate risky transaction signature
        payload = {
            "TransactionAmount": round(random.uniform(5500, 32000), 2),
            "TransactionDuration": random.randint(2, 9),
            "LoginAttempts": random.randint(4, 8),
            "AccountBalance": round(random.uniform(100, 3500), 2),
            "TransactionDate": now.strftime("%Y-%m-%d %H:%M:%S"),
            "PreviousTransactionDate": prev.strftime("%Y-%m-%d %H:%M:%S"),
            "TransactionType": "Debit",
            "Location": random.choice(["Moscow", "Lagos", "New York", "Chicago", "London"]),
            "DeviceID": random.choice(["DVC_SUSP_01", "DVC_SUSP_02", "DVC9012"]),
            "MerchantID": random.choice(["MRCH_HIGH_RISK", "MRCH8821", "MRCH4410"]),
            "Channel": "Online",
            "CustomerOccupation": random.choice(["Student", "Doctor", "Engineer", "Retired"]),
            "AccountID": f"AC{random.randint(100, 999):05d}",
            "IsSimulatedFraud": True
        }
        if fraud_pattern:
            for key, val_fn in fraud_pattern["overrides"].items():
                payload[key] = val_fn() if callable(val_fn) else val_fn
    else:
        # Generate clean legitimate transaction
        payload = {
            "TransactionAmount": float(row.get("TransactionAmount", random.uniform(15.0, 280.0))),
            "TransactionDuration": max(35, int(row.get("TransactionDuration", random.randint(45, 180)))),
            "LoginAttempts": 1,
            "AccountBalance": float(row.get("AccountBalance", random.uniform(2500.0, 15000.0))),
            "TransactionDate": now.strftime("%Y-%m-%d %H:%M:%S"),
            "PreviousTransactionDate": prev.strftime("%Y-%m-%d %H:%M:%S"),
            "TransactionType": str(row.get("TransactionType", "Debit")),
            "Location": str(row.get("Location", random.choice(["New York", "San Francisco", "Austin", "Boston"]))),
            "DeviceID": str(row.get("DeviceID", f"DVC{random.randint(1000, 9999)}")),
            "MerchantID": str(row.get("MerchantID", f"MRCH{random.randint(100, 899)}")),
            "Channel": str(row.get("Channel", random.choice(["Online", "ATM", "Branch"]))),
            "CustomerOccupation": str(row.get("CustomerOccupation", "Engineer")),
            "AccountID": str(row.get("AccountID", f"AC{random.randint(100, 899):05d}")),
            "IsSimulatedFraud": False
        }

    return payload


def send_transaction(payload, fraud_pattern_name=None, slot_info=""):
    stats["total"] += 1
    txn_num = stats["total"]
    try:
        resp = requests.post(FLASK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        fraud_score = result.get("composite_score", result.get("risk_score", 0))
        is_fraud    = result.get("is_fraud", fraud_score > 0.5)
        explanation = result.get("explanation", [])
        top_feature = explanation[0].get("feature", "N/A") if explanation else "N/A"

        if is_fraud:
            stats["fraud"] += 1
            flag = f"{BG_RED}{WHITE}{BOLD} RISKY/FRAUD {RESET}"
            score_color = RED
        else:
            stats["normal"] += 1
            flag = f"{BG_GREEN}{WHITE}{BOLD} LEGIT/CLEAR {RESET}"
            score_color = GREEN

        pattern_tag = f"  {MAGENTA}[{fraud_pattern_name}]{RESET}" if fraud_pattern_name else ""
        slot_tag = f"{DIM}({slot_info}){RESET}" if slot_info else ""
        print(
            f"  {DIM}#{txn_num:>4}{RESET} {slot_tag:<8} {flag}  "
            f"{WHITE}${payload['TransactionAmount']:>9.2f}{RESET}  "
            f"Risk:{score_color}{BOLD}{fraud_score:.3f}{RESET}  "
            f"{CYAN}{payload['AccountID']:<8}{RESET} "
            f"{YELLOW}{payload['Location']:<14}{RESET} "
            f"{DIM}top:{top_feature}{RESET}"
            f"{pattern_tag}"
        )
    except requests.exceptions.ConnectionError:
        stats["errors"] += 1
        print(f"  {RED}#{txn_num:>4}  ERROR: Cannot connect to {FLASK_URL} -- is the Flask app running?{RESET}")
    except Exception as e:
        stats["errors"] += 1
        print(f"  {RED}#{txn_num:>4}  ERROR: {e}{RESET}")


def main():
    print_banner()
    if not os.path.exists(DATA_FILE):
        print(f"{RED}ERROR: Cannot find {DATA_FILE}. Run from the project root.{RESET}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE).dropna(subset=["TransactionAmount", "AccountID"])
    total_rows = len(df)
    print(f"  {GREEN}Loaded {total_rows} base transaction profiles.{RESET}")
    print(f"  {CYAN}Streaming calibrated at {1/INTERVAL_SECS:.1f} txn/s with 2 to 5 risky transactions per 10-batch.{RESET}\n")
    print(f"{BOLD}{BLUE}{'─'*84}{RESET}")
    print(f"  {DIM}{'#':>4}  {'SLOT':<6} {'STATUS':<13} {'AMOUNT':>10}  {'RISK':>7}  {'ACCOUNT':<8} {'LOCATION':<14} TOP INDICATOR{RESET}")
    print(f"{BOLD}{BLUE}{'─'*84}{RESET}")

    idx = 0
    batch_num = 0

    try:
        while True:
            # ── Form a new batch of 10 transactions ───────────────────────
            batch_num += 1
            stats["batch_count"] = batch_num
            # Calibrate: choose between 2 and 5 risky transactions in this batch
            num_risky = random.randint(MIN_RISKY, MAX_RISKY)
            risky_positions = set(random.sample(range(BATCH_SIZE), num_risky))

            print(f"\n  {BOLD}{BG_BLUE}{WHITE} BATCH #{batch_num} -- Calibrated: {num_risky} of 10 Transactions Scheduled as Risky ({num_risky*10}%) {RESET}")

            for slot in range(BATCH_SIZE):
                row = df.iloc[idx % total_rows].to_dict()
                idx += 1
                
                is_risky_slot = (slot in risky_positions)
                fraud_pattern = None

                if is_risky_slot:
                    fraud_pattern = random.choice(FRAUD_PATTERNS)
                    stats["injected_patterns"] += 1

                payload = build_payload(row, fraud_pattern, is_risky=is_risky_slot)
                slot_label = f"{slot+1}/10"
                send_transaction(payload, fraud_pattern["name"] if fraud_pattern else None, slot_info=slot_label)

                time.sleep(INTERVAL_SECS)

            # Print cumulative statistics at the end of each 10-batch
            print_stats()

    except KeyboardInterrupt:
        print(f"\n\n{BOLD}{YELLOW}Simulation paused by user.{RESET}")
        print_stats()
        elapsed = (datetime.now() - stats["start_time"]).seconds
        print(f"  {CYAN}Check real-time graphs and alerts at http://localhost:5000{RESET}\n")


if __name__ == "__main__":
    main()
