"""
Full pipeline demo on Zachary's Karate Club:
Adjacency -> Laplacian -> Eigen-decomposition -> Fiedler vector ->
2-way split -> compare to real historical outcome -> save figures.

Run:  python scripts/run_karate_club.py
Outputs written to results/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from spectral_clustering import graphs, laplacian, spectral, metrics, visualize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    G = graphs.load_karate_club()
    truth = graphs.ground_truth_labels(G)
    true_labels = np.array([truth[n] for n in G.nodes()])

    print(f"Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}")

    A = laplacian.adjacency_matrix(G)
    L = laplacian.unnormalized_laplacian(A)

    rank, nullity = laplacian.rank_nullity(L)
    print(f"rank(L) = {rank}   nullity(L) = {nullity}  (1 = fully connected)")

    ortho_err, recon_err = laplacian.verify_diagonalization(L)
    print(f"Diagonalization check: orthogonality err = {ortho_err:.2e}, "
          f"reconstruction err = {recon_err:.2e}")

    eigvals, eigvecs = spectral.eigen_decompose(L)
    print(f"Smallest 5 eigenvalues: {np.round(eigvals[:5], 4)}")
    print(f"Fiedler value: {eigvals[1]:.4f}")

    fiedler = spectral.fiedler_vector(eigvecs)
    pred_labels = spectral.two_way_split(fiedler)

    acc = metrics.accuracy_vs_ground_truth(pred_labels, true_labels)
    a = metrics.ari(true_labels, pred_labels)
    n = metrics.nmi(true_labels, pred_labels)
    q = metrics.modularity(G, pred_labels)
    print(f"\nAccuracy vs. real historical split: {acc*100:.1f}%")
    print(f"ARI = {a:.3f}   NMI = {n:.3f}   Modularity = {q:.3f}")

    k = spectral.eigengap_heuristic(eigvals, max_k=6)
    print(f"\nEigengap heuristic suggests k = {k} clusters")

    # ---- figures ----
    pos = visualize.plot_clusters(
        G, pred_labels, "Predicted by Fiedler Vector",
        os.path.join(RESULTS_DIR, "karate_predicted.png"))
    visualize.plot_clusters(
        G, true_labels, "Actual Historical Split", os.path.join(RESULTS_DIR, "karate_actual.png"), pos=pos)
    visualize.plot_eigenvalue_spectrum(
        eigvals, os.path.join(RESULTS_DIR, "karate_eigenvalue_spectrum.png"))
    visualize.plot_fiedler_bar(
        fiedler, os.path.join(RESULTS_DIR, "karate_fiedler_vector.png"))

    print(f"\nFigures saved to {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    run()
