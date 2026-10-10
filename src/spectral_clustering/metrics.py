"""Evaluation metrics for comparing predicted clusters to ground truth
(when available) or judging cluster quality on their own (modularity).
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def accuracy_vs_ground_truth(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Fraction of nodes correctly labeled, accounting for the fact
    that cluster IDs are arbitrary (predicted cluster '0' might
    correspond to true label '1'). Tries both possible 2-cluster
    label mappings and returns the better one.
    """
    pred = np.asarray(pred_labels)
    true = np.asarray(true_labels)
    direct = np.mean(pred == true)
    flipped = np.mean((1 - pred) == true)
    return float(max(direct, flipped))


def modularity(G: nx.Graph, labels: np.ndarray) -> float:
    """Newman's modularity Q of a partition: measures how much denser
    the connections WITHIN predicted communities are compared to what
    you'd expect by random chance, given the same degree sequence.
    Ranges roughly [-0.5, 1]; values above ~0.3 typically indicate
    genuine community structure.
    """
    nodes = list(G.nodes())
    communities: dict[int, set] = {}
    for node, lab in zip(nodes, labels):
        communities.setdefault(int(lab), set()).add(node)
    return nx.algorithms.community.quality.modularity(G, communities.values())


def ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Rand Index: pair-counting agreement between two
    partitions, corrected for chance. 1.0 = identical partitions
    (up to relabeling), ~0.0 = agreement no better than random.
    """
    return float(adjusted_rand_score(true_labels, pred_labels))


def nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Normalized Mutual Information: information-theoretic agreement
    between two partitions. 1.0 = identical partitions, 0.0 = no
    shared information.
    """
    return float(normalized_mutual_info_score(true_labels, pred_labels))
