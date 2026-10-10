"""
Benchmark: our Fiedler-vector spectral clustering vs. established
community-detection methods, on Zachary's Karate Club (ground truth
available), scored by Accuracy, ARI, NMI, and Modularity.

Methods compared:
  1. Our implementation, unnormalized Laplacian (L = D - A)
  2. Our implementation, symmetric normalized Laplacian (L_sym)
  3. networkx built-in: spectral_modularity_bipartition
     (Newman's leading-eigenvector method)
  4. networkx built-in: Louvain method (modularity maximization,
     restricted/mapped to 2 groups for a fair accuracy comparison)

Run:  python scripts/benchmark_methods.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import networkx as nx
from spectral_clustering import graphs, laplacian, spectral, metrics


def labels_from_two_groups(G, group_a, group_b):
    labels = {}
    for n in group_a:
        labels[n] = 0
    for n in group_b:
        labels[n] = 1
    return np.array([labels[n] for n in G.nodes()])


def run():
    G = graphs.load_karate_club()
    truth = graphs.ground_truth_labels(G)
    true_labels = np.array([truth[n] for n in G.nodes()])

    results = []

    # ---- 1. Our method: unnormalized Laplacian ----
    A = laplacian.adjacency_matrix(G)
    L = laplacian.unnormalized_laplacian(A)
    eigvals, eigvecs = spectral.eigen_decompose(L)
    fiedler = spectral.fiedler_vector(eigvecs)
    pred1 = spectral.two_way_split(fiedler)
    results.append(("Ours: unnormalized Laplacian", pred1))

    # ---- 2. Our method: symmetric normalized Laplacian ----
    L_sym = laplacian.normalized_laplacian_sym(A)
    eigvals_s, eigvecs_s = spectral.eigen_decompose(L_sym)
    fiedler_s = spectral.fiedler_vector(eigvecs_s)
    pred2 = spectral.two_way_split(fiedler_s)
    results.append(("Ours: normalized Laplacian", pred2))

    # ---- 3. networkx: spectral modularity bipartition (Newman) ----
    group_a, group_b = nx.community.spectral_modularity_bipartition(G)
    pred3 = labels_from_two_groups(G, group_a, group_b)
    results.append(("networkx: Newman leading-eigenvector", pred3))

    # ---- 4. networkx: Louvain (mapped to 2 groups via modularity matrix sign) ----
    louvain_communities = nx.community.louvain_communities(G, seed=42)
    # Louvain may find >2 communities; merge into 2 by modularity-greedy pairing
    # for a fair like-for-like accuracy comparison against the known 2-way split.
    sizes = sorted(louvain_communities, key=len, reverse=True)
    group_a4 = set(sizes[0])
    group_b4 = set().union(*sizes[1:]) if len(sizes) > 1 else set()
    pred4 = labels_from_two_groups(G, group_a4, group_b4)
    results.append((f"networkx: Louvain ({len(louvain_communities)} raw communities, merged to 2)", pred4))

    # ---- score everything ----
    print(f"{'Method':45s} {'Accuracy':>9s} {'ARI':>7s} {'NMI':>7s} {'Modularity':>11s}")
    print("-" * 82)
    for name, pred in results:
        acc = metrics.accuracy_vs_ground_truth(pred, true_labels)
        a = metrics.ari(true_labels, pred)
        n = metrics.nmi(true_labels, pred)
        q = metrics.modularity(G, pred)
        print(f"{name:45s} {acc*100:8.1f}% {a:7.3f} {n:7.3f} {q:11.3f}")

    print()
    print("Reference (Newman 2006, 'Finding community structure using eigenvectors'):")
    print("  best modularity reported on this dataset: Q ~ 0.42 (normalized Laplacian)")


if __name__ == "__main__":
    run()
