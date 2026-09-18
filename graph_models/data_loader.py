"""
graph_models/data_loader.py
FraudGuard AI - Transaction Graph Builder
Constructs Account-Merchant-Device heterogeneous bipartite graph for GNN inference
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import hashlib
import threading

try:
    import torch
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


def _node_id(prefix: str, value: str) -> str:
    """Create a stable, namespaced node identifier."""
    return f"{prefix}:{value}"


def _hash_to_float(value: str) -> float:
    """Convert string to a stable float in [0, 1] for node features."""
    h = int(hashlib.md5(value.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0


class TransactionGraphBuilder:
    """
    Builds a heterogeneous bipartite transaction graph for GNN inference.
    
    Node types:
      - Account nodes  (blue)  : identified by AccountID
      - Merchant nodes (green) : identified by MerchantID
      - Device nodes   (amber) : identified by DeviceID
    
    Edge types:
      - Account → Merchant  (transaction)
      - Account → Device    (device usage)
      - Merchant → Device   (merchant-device link)
    
    Node features (8-dim per node):
      [normalized_amount, normalized_balance, login_attempts, duration,
       is_high_risk, node_type_account, node_type_merchant, node_type_device]
    """

    def __init__(self):
        self.node_index: Dict[str, int] = {}
        self.node_features: List[List[float]] = []
        self.edges: List[Tuple[int, int]] = []
        self._node_counter = 0
        self._lock = threading.Lock()

    def _get_or_create_node(self, node_id: str, features: List[float]) -> int:
        """Idempotent node creation — returns existing index or creates new."""
        with self._lock:
            if node_id not in self.node_index:
                self.node_index[node_id] = self._node_counter
                self.node_features.append(features)
                self._node_counter += 1
            return self.node_index[node_id]

    def add_transaction(self, txn: dict) -> None:
        """
        Add a single transaction to the graph.
        
        Expected keys: AccountID, MerchantID, DeviceID, TransactionAmount,
                       AccountBalance, LoginAttempts, TransactionDuration, IsHighRisk
        """
        # Normalize numeric features
        amount_norm = min(float(txn.get('TransactionAmount', 0)) / 50000.0, 1.0)
        balance_norm = min(float(txn.get('AccountBalance', 0)) / 100000.0, 1.0)
        login_norm = min(float(txn.get('LoginAttempts', 0)) / 10.0, 1.0)
        duration_norm = min(float(txn.get('TransactionDuration', 60)) / 300.0, 1.0)
        is_hr = 1.0 if txn.get('IsHighRisk', False) else 0.0

        account_id = str(txn.get('AccountID', 'UNKNOWN'))
        merchant_id = str(txn.get('MerchantID', 'MERCH_00'))
        device_id = str(txn.get('DeviceID', 'DEV_00'))

        # Account node: [amount, balance, login, duration, is_hr, 1, 0, 0]
        acc_node = _node_id('ACC', account_id)
        acc_features = [amount_norm, balance_norm, login_norm, duration_norm, is_hr, 1.0, 0.0, 0.0]
        acc_idx = self._get_or_create_node(acc_node, acc_features)

        # Merchant node: [hash_amount, 0, 0, 0, is_hr, 0, 1, 0]
        merch_node = _node_id('MERCH', merchant_id)
        merch_features = [_hash_to_float(merchant_id), 0.0, 0.0, 0.0, is_hr, 0.0, 1.0, 0.0]
        merch_idx = self._get_or_create_node(merch_node, merch_features)

        # Device node: [hash_device, 0, login, 0, is_hr, 0, 0, 1]
        dev_node = _node_id('DEV', device_id)
        dev_features = [_hash_to_float(device_id), 0.0, login_norm, 0.0, is_hr, 0.0, 0.0, 1.0]
        dev_idx = self._get_or_create_node(dev_node, dev_features)

        # Add edges (bidirectional)
        with self._lock:
            self.edges.extend([
                (acc_idx, merch_idx), (merch_idx, acc_idx),  # Account ↔ Merchant
                (acc_idx, dev_idx),   (dev_idx, acc_idx),    # Account ↔ Device
                (merch_idx, dev_idx), (dev_idx, merch_idx),  # Merchant ↔ Device
            ])

    def build(self) -> Optional[object]:
        """
        Finalize and return a PyG Data object (or dict fallback).
        Returns None if no transactions were added.
        """
        with self._lock:
            if not self.node_features:
                return None
            
            node_features_copy = list(self.node_features)
            edges_copy = list(self.edges)
            num_nodes = len(node_features_copy)
            
        if TORCH_GEOMETRIC_AVAILABLE:
            x = torch.tensor(node_features_copy, dtype=torch.float)

            if edges_copy:
                edge_index = torch.tensor(edges_copy, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        else:
            # Fallback: return as dict
            return {
                'node_features': node_features_copy,
                'edges': edges_copy,
                'num_nodes': num_nodes
            }

    def reset(self) -> None:
        """Clear graph state for next batch."""
        with self._lock:
            self.node_index.clear()
            self.node_features.clear()
            self.edges.clear()
            self._node_counter = 0

    def get_vis_network_data(self) -> dict:
        """
        Export graph as Vis-Network compatible nodes/edges JSON for the dashboard.
        Returns: {'nodes': [...], 'edges': [...]}
        """
        vis_nodes = []
        vis_edges = []

        type_colors = {
            'ACC':   {'background': '#3b82f6', 'border': '#1d4ed8', 'label': 'Account'},
            'MERCH': {'background': '#22c55e', 'border': '#15803d', 'label': 'Merchant'},
            'DEV':   {'background': '#f59e0b', 'border': '#d97706', 'label': 'Device'},
        }

        with self._lock:
            node_index_copy = dict(self.node_index)
            edges_copy = list(self.edges)

        for node_id_str, idx in node_index_copy.items():
            prefix = node_id_str.split(':')[0]
            label_value = node_id_str.split(':', 1)[1]
            color_info = type_colors.get(prefix, {'background': '#94a3b8', 'border': '#64748b', 'label': 'Unknown'})

            vis_nodes.append({
                'id': idx,
                'label': label_value[:12],
                'title': f"{color_info['label']}: {label_value}",
                'color': {'background': color_info['background'], 'border': color_info['border']},
                'font': {'color': '#ffffff', 'size': 11},
                'shape': 'dot',
                'size': 16 if prefix == 'ACC' else (12 if prefix == 'MERCH' else 10),
                'group': prefix,
            })

        seen_edges = set()
        for src, dst in edges_copy:
            key = tuple(sorted([src, dst]))
            if key not in seen_edges:
                seen_edges.add(key)
                vis_edges.append({
                    'from': src,
                    'to': dst,
                    'color': {'color': '#cbd5e1', 'opacity': 0.7},
                    'width': 1,
                    'smooth': {'type': 'dynamic'},
                })

        return {'nodes': vis_nodes, 'edges': vis_edges}


def build_graph_from_transactions(transactions: list) -> Tuple[Optional[object], dict]:
    """
    Convenience function: build graph from list of transaction dicts.
    Returns: (pyg_data_or_dict, vis_network_dict)
    """
    builder = TransactionGraphBuilder()
    for txn in transactions:
        builder.add_transaction(txn)
    return builder.build(), builder.get_vis_network_data()
