"""
simulate_transactions.py
FraudGuard AI - Calibrated Live Transaction Streamer
Batch-window calibration: 12 txns/batch, exactly 3 risky (25.0% fraud rate)
5 fraud archetypes with realistic patterns
"""

import time
import random
import requests
import json
from datetime import datetime, timezone
from typing import Tuple

try:
    from colorama import Fore, Back, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

API_URL          = "http://localhost:5000/api/analyze"
DELAY_SECONDS    = 1.2          # Interval between transactions
BATCH_SIZE       = 12           # Transactions per window
RISKY_PER_BATCH  = 3            # Exactly 3 risky per 12-txn batch = 25.0%
CLEAN_PER_BATCH  = BATCH_SIZE - RISKY_PER_BATCH   # 9

# ─── Pools ────────────────────────────────────────────────────────────────────
ACCOUNT_IDS = [f"AC{i:05d}" for i in range(100, 200)]
MERCHANT_IDS = [f"M{i:03d}" for i in range(100, 160)]
DEVICE_IDS = [f"DVC_{i:04d}" for i in range(1000, 1060)]
CLEAN_LOCATIONS = [
    'New York', 'London', 'Singapore', 'Chicago', 'Los Angeles',
    'Toronto', 'Sydney', 'Paris', 'Tokyo', 'Berlin', 'Amsterdam',
    'Stockholm', 'Zurich', 'Seoul', 'Mumbai',
]
RISKY_LOCATIONS = ['Moscow', 'Lagos', 'Pyongyang', 'Tehran', 'Caracas', 'Minsk', 'Tripoli']
TXN_TYPES_CLEAN = ['Online Purchase', 'POS Debit', 'ATM Withdrawal', 'ACH Transfer', 'Bill Payment']
TXN_TYPES_RISKY = ['Wire Transfer', 'International Wire', 'Online Debit', 'Card Not Present']
CATEGORIES      = ['Electronics', 'Retail', 'Travel', 'Restaurant', 'Gas Station', 'Grocery']

SUSPECT_DEVICE = "DVC_SUSP_01"   # Shared device collusion ring

# ─── Running Stats ────────────────────────────────────────────────────────────
stats = {
    'total':     0,
    'flagged':   0,
    'blocked':   0,
    'cleared':   0,
    'batches':   0,
    'errors':    0,
    'start_time': time.time(),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Color Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def c(text: str, color: str) -> str:
    if not COLORS_AVAILABLE:
        return text
    colors = {
        'red':     Fore.RED,
        'green':   Fore.GREEN,
        'yellow':  Fore.YELLOW,
        'cyan':    Fore.CYAN,
        'magenta': Fore.MAGENTA,
        'white':   Fore.WHITE,
        'blue':    Fore.BLUE,
        'bright':  Style.BRIGHT,
        'dim':     Style.DIM,
        'reset':   Style.RESET_ALL,
    }
    return colors.get(color, '') + str(text) + Style.RESET_ALL


def bold(text: str) -> str:
    return c(text, 'bright')


# ═══════════════════════════════════════════════════════════════════════════════
# Transaction Generators
# ═══════════════════════════════════════════════════════════════════════════════

def make_clean_transaction() -> Tuple[dict, str]:
    """Generate a legitimate, low-risk transaction (RiskScore < 0.20)."""
    account = random.choice(ACCOUNT_IDS)
    balance = round(random.uniform(5000, 80000), 2)
    amount  = round(random.uniform(10, min(500, balance * 0.05)), 2)

    txn = {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     random.choice(TXN_TYPES_CLEAN),
        'MerchantID':          random.choice(MERCHANT_IDS),
        'MerchantCategory':    random.choice(CATEGORIES),
        'DeviceID':            random.choice(DEVICE_IDS),
        'Location':            random.choice(CLEAN_LOCATIONS),
        'LoginAttempts':       random.randint(1, 2),
        'TransactionDuration': round(random.uniform(60, 300), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }
    return txn, 'CLEAN'


def make_fraud_wire_transfer() -> Tuple[dict, str]:
    """Fraud Archetype 1: High-Value Rapid Wire Transfer ($9,500–$48,000)."""
    account = random.choice(ACCOUNT_IDS)
    balance = round(random.uniform(12000, 60000), 2)
    amount  = round(random.uniform(9500, 48000), 2)

    txn = {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     'Wire Transfer',
        'MerchantID':          f"M{random.randint(900, 999)}",
        'MerchantCategory':    'Financial Services',
        'DeviceID':            random.choice(DEVICE_IDS),
        'Location':            random.choice(CLEAN_LOCATIONS),
        'LoginAttempts':       random.randint(4, 9),
        'TransactionDuration': round(random.uniform(5, 25), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }
    return txn, 'WIRE FRAUD'


def make_fraud_location_hop() -> Tuple[dict, str]:
    """Fraud Archetype 2: Suspicious Location Hop to high-risk geo."""
    account = random.choice(ACCOUNT_IDS)
    balance = round(random.uniform(8000, 50000), 2)
    amount  = round(random.uniform(500, 15000), 2)

    txn = {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     random.choice(['International Wire', 'Wire Transfer']),
        'MerchantID':          f"M{random.randint(800, 899)}",
        'MerchantCategory':    'International Transfer',
        'DeviceID':            random.choice(DEVICE_IDS),
        'Location':            random.choice(RISKY_LOCATIONS),
        'LoginAttempts':       random.randint(2, 6),
        'TransactionDuration': round(random.uniform(10, 60), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }
    return txn, 'GEO HOP'


def make_fraud_balance_drain() -> Tuple[dict, str]:
    """Fraud Archetype 3: Account Takeover — Balance Drain (90%+ in one shot)."""
    account = random.choice(ACCOUNT_IDS)
    balance = round(random.uniform(10000, 90000), 2)
    amount  = round(balance * random.uniform(0.90, 0.99), 2)  # 90-99% drain

    txn = {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     'Online Debit',
        'MerchantID':          f"M{random.randint(700, 799)}",
        'MerchantCategory':    'Wire Transfer',
        'DeviceID':            random.choice(DEVICE_IDS),
        'Location':            random.choice(CLEAN_LOCATIONS + RISKY_LOCATIONS[:3]),
        'LoginAttempts':       random.randint(5, 10),
        'TransactionDuration': round(random.uniform(3, 15), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }
    return txn, 'BALANCE DRAIN'


def make_fraud_card_probing() -> Tuple[dict, str]:
    """Fraud Archetype 4: Automated Velocity Card Probing ($0.50–$2.99)."""
    account = random.choice(ACCOUNT_IDS)
    balance = round(random.uniform(500, 5000), 2)
    amount  = round(random.uniform(0.50, 2.99), 2)

    txn = {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     'Card Not Present',
        'MerchantID':          f"M{random.randint(500, 599)}",
        'MerchantCategory':    'Micro-Transaction',
        'DeviceID':            random.choice(DEVICE_IDS),
        'Location':            random.choice(CLEAN_LOCATIONS),
        'LoginAttempts':       random.randint(6, 10),
        'TransactionDuration': round(random.uniform(1, 8), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }
    return txn, 'CARD PROBE'


def make_fraud_device_collusion() -> Tuple[dict, str]:
    """Fraud Archetype 5: Shared Device Collusion Ring (DVC_SUSP_01)."""
    account = random.choice(ACCOUNT_IDS)
    balance = round(random.uniform(3000, 30000), 2)
    amount  = round(random.uniform(1000, 20000), 2)

    txn = {
        'AccountID':           account,
        'CustomerID':          f"CUS{account[2:]}",
        'TransactionAmount':   amount,
        'AccountBalance':      balance,
        'TransactionType':     random.choice(['Wire Transfer', 'Online Debit', 'ACH Transfer']),
        'MerchantID':          f"M{random.randint(600, 699)}",
        'MerchantCategory':    'Financial Services',
        'DeviceID':            SUSPECT_DEVICE,
        'Location':            random.choice(CLEAN_LOCATIONS + RISKY_LOCATIONS[:2]),
        'LoginAttempts':       random.randint(3, 8),
        'TransactionDuration': round(random.uniform(5, 30), 1),
        'Timestamp':           datetime.now(timezone.utc).isoformat(),
    }
    return txn, 'DEVICE COLLUSION'


FRAUD_ARCHETYPES = [
    make_fraud_wire_transfer,
    make_fraud_location_hop,
    make_fraud_balance_drain,
    make_fraud_card_probing,
    make_fraud_device_collusion,
]


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_batch() -> list:
    """
    Build one calibrated batch of 12 transactions:
    - 9 clean (legitimate)
    - 3 risky (one per fraud archetype, randomly sampled)
    """
    batch = []

    # 9 clean transactions
    for _ in range(CLEAN_PER_BATCH):
        txn, label = make_clean_transaction()
        batch.append((txn, label))

    # 3 risky transactions — pick 3 distinct archetypes
    chosen_archetypes = random.sample(FRAUD_ARCHETYPES, RISKY_PER_BATCH)
    for archetype_fn in chosen_archetypes:
        txn, label = archetype_fn()
        batch.append((txn, label))

    # Shuffle within batch to avoid ordering bias
    random.shuffle(batch)
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# API Sender
# ═══════════════════════════════════════════════════════════════════════════════

def send_transaction(txn: dict) -> dict:
    """POST transaction to /api/analyze and return response dict."""
    try:
        resp = requests.post(API_URL, json=txn, timeout=10)
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {'error': 'CONNECTION_REFUSED', 'status': 'ERROR'}
    except requests.exceptions.Timeout:
        return {'error': 'TIMEOUT', 'status': 'ERROR'}
    except Exception as e:
        return {'error': str(e), 'status': 'ERROR'}


# ═══════════════════════════════════════════════════════════════════════════════
# Display Formatters
# ═══════════════════════════════════════════════════════════════════════════════

def format_status_badge(result: dict, archetype_label: str) -> str:
    """Format colored status badge from API result."""
    if 'error' in result:
        return c(f"[ERROR: {result['error']}]", 'red')

    status    = result.get('status', 'Unknown')
    risk      = float(result.get('risk_score', 0))
    is_blocked = result.get('is_blocked', False)
    is_fraud  = result.get('is_fraud', False)

    if is_blocked:
        return c('🔒 BLOCKED', 'red') + c(f" ({risk:.3f})", 'yellow')
    elif is_fraud or risk >= 0.7:
        color = 'red' if risk >= 0.7 else 'yellow'
        return c(f'🚨 FLAGGED [{status}]', color) + c(f" ({risk:.3f})", 'yellow')
    elif risk >= 0.5:
        return c(f'⚠️  HIGH RISK', 'yellow') + c(f" ({risk:.3f})", 'yellow')
    else:
        return c(f'✅ Cleared', 'green') + c(f" ({risk:.3f})", 'dim')


def print_transaction(slot: int, total_in_batch: int, txn: dict, archetype_label: str, result: dict):
    """Print a formatted transaction row to terminal."""
    slot_tag   = c(f"({slot}/{total_in_batch})", 'dim')
    txn_id     = result.get('transaction_id', txn.get('TransactionID', '?'))
    account_id = txn.get('AccountID', '?')
    amount     = float(txn.get('TransactionAmount', 0))
    location   = txn.get('Location', '?')
    txn_type   = txn.get('TransactionType', '?')[:16]
    risk       = float(result.get('risk_score', 0))
    status_str = format_status_badge(result, archetype_label)

    # Archetype tag
    if archetype_label == 'CLEAN':
        arch_tag = c('[LEGIT]', 'dim')
    else:
        arch_tag = c(f'[{archetype_label}]', 'magenta')

    amount_str = c(f"${amount:>10,.2f}", 'cyan' if amount > 1000 else 'white')

    print(
        f"  {slot_tag} {c(txn_id, 'blue')} │ "
        f"{c(account_id, 'white')} │ "
        f"{amount_str} │ "
        f"{c(location[:14],'yellow'):16} │ "
        f"{txn_type[:16]:18} │ "
        f"{status_str} {arch_tag}"
    )


def print_batch_header(batch_num: int):
    line = "─" * 115
    print(f"\n  {c(line, 'dim')}")
    print(
        f"  {bold('BATCH')} {c(f'#{batch_num:04d}', 'cyan')}  │  "
        f"{c(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'), 'dim')}  │  "
        f"Window: {BATCH_SIZE} txns  │  Target fraud rate: "
        f"{c(f'{(RISKY_PER_BATCH/BATCH_SIZE)*100:.1f}%', 'yellow')}"
    )
    print(f"  {c(line, 'dim')}")
    print(
        f"  {'Slot':6} {'TxnID':12} │ {'Account':8} │ {'Amount':>12} │ "
        f"{'Location':16} │ {'Type':18} │ Status"
    )
    print(f"  {c(line, 'dim')}")


def print_batch_summary(batch_results: list):
    """Print running stats after each batch."""
    flagged = sum(1 for _, r in batch_results if r.get('is_fraud'))
    cleared = sum(1 for _, r in batch_results if r.get('status') == 'Cleared')
    avg_risk = sum(float(r.get('risk_score', 0)) for _, r in batch_results) / max(len(batch_results), 1)
    actual_rate = (flagged / max(len(batch_results), 1)) * 100.0

    # Update global stats
    stats['total']   += len(batch_results)
    stats['flagged'] += flagged
    stats['cleared'] += cleared
    stats['batches'] += 1
    stats['blocked'] += sum(1 for _, r in batch_results if r.get('is_blocked'))

    elapsed = time.time() - stats['start_time']
    elapsed_str = f"{int(elapsed//60)}m{int(elapsed%60)}s"

    print(f"\n  {'─'*115}")
    summary = (
        f"  📊 Batch Summary │ "
        f"Flagged: {c(str(flagged), 'red')}/{BATCH_SIZE} │ "
        f"Rate: {c(f'{actual_rate:.1f}%', 'yellow')} │ "
        f"Avg Risk: {c(f'{avg_risk:.3f}', 'cyan')} │ "
        f"Total Scanned: {c(str(stats['total']), 'white')} │ "
        f"Uptime: {c(elapsed_str, 'dim')}"
    )
    print(summary)
    print(
        f"  🌐 Cumulative    │ "
        f"Total Flagged: {c(str(stats['flagged']), 'red')} │ "
        f"Cleared: {c(str(stats['cleared']), 'green')} │ "
        f"Blocked: {c(str(stats['blocked']), 'magenta')} │ "
        f"Batches Run: {c(str(stats['batches']), 'cyan')}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Wait for Server
# ═══════════════════════════════════════════════════════════════════════════════

def wait_for_server(max_retries: int = 30, retry_interval: float = 2.0) -> bool:
    """Poll /api/health until server is ready."""
    health_url = "http://localhost:5000/api/health"
    print(f"\n  {c('⏳ Waiting for FraudGuard AI server...', 'yellow')}")
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  {c('✅ Server ready!', 'green')} (models_loaded: {data.get('models_loaded', '?')})")
                return True
        except Exception:
            pass
        print(f"  {c(f'  Attempt {attempt}/{max_retries}...', 'dim')}", end='\r')
        time.sleep(retry_interval)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Startup Banner ────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(bold("  🛡️  FraudGuard AI — Live Transaction Simulator"))
    print(f"  Calibration: {BATCH_SIZE} txns/batch │ {RISKY_PER_BATCH} risky │ {CLEAN_PER_BATCH} clean │ {DELAY_SECONDS}s delay")
    print(f"  5 Fraud archetypes: Wire Fraud, Geo Hop, Balance Drain, Card Probe, Device Collusion")
    print("="*80)

    # ── Wait for server ────────────────────────────────────────────────────────
    if not wait_for_server():
        print(c("\n  ❌ Could not connect to FraudGuard AI server.", 'red'))
        print(c("  → Start the server first: python app.py", 'yellow'))
        return

    print(f"\n  {c('▶ Streaming transactions...', 'green')} Press Ctrl+C to stop.\n")
    batch_num = 0

    try:
        while True:
            batch_num += 1
            batch = build_batch()
            print_batch_header(batch_num)

            batch_results = []
            for slot, (txn, archetype_label) in enumerate(batch, start=1):
                result = send_transaction(txn)

                if 'error' in result and result['error'] == 'CONNECTION_REFUSED':
                    print(c(f"\n  ❌ Server connection lost. Retrying...", 'red'))
                    if not wait_for_server(max_retries=15, retry_interval=2.0):
                        print(c("  Server not responding. Exiting.", 'red'))
                        return

                print_transaction(slot, BATCH_SIZE, txn, archetype_label, result)
                batch_results.append((txn, result))
                time.sleep(DELAY_SECONDS)

            print_batch_summary(batch_results)
            time.sleep(0.5)   # Brief pause between batches

    except KeyboardInterrupt:
        elapsed = time.time() - stats['start_time']
        print(f"\n\n  {c('⏹  Simulation stopped by user.', 'yellow')}")
        print(f"  Total transactions sent: {c(str(stats['total']), 'cyan')}")
        print(f"  Total flagged:           {c(str(stats['flagged']), 'red')}")
        print(f"  Total cleared:           {c(str(stats['cleared']), 'green')}")
        print(f"  Total runtime:           {c(f'{int(elapsed//60)}m {int(elapsed%60)}s', 'dim')}")
        fraud_rate = (stats['flagged'] / max(stats['total'], 1)) * 100
        print(f"  Overall fraud rate:      {c(f'{fraud_rate:.1f}%', 'yellow')}")
        print()


if __name__ == '__main__':
    main()
