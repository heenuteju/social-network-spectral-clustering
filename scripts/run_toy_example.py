"""
Full pipeline demo on the 9-node toy graph (for understanding every
step before trusting the method on real data).

Run:  python scripts/run_toy_example.py
Outputs written to results/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from spectral_clustering import graphs, laplacian, spectral, visualize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    G = graphs.load_toy_graph()
    A = laplacian.adjacency_matrix(G)
    print("Adjacency matrix:\n", A.astype(int))

    L = laplacian.unnormalized_laplacian(A)
    print("\nLaplacian:\n", L.astype(int))

    rank, nullity = laplacian.rank_nullity(L)
    print(f"\nrank(L) = {rank}   nullity(L) = {nullity}")

    ortho_err, recon_err = laplacian.verify_diagonalization(L)
    print(f"Diagonalization check: orthogonality err = {ortho_err:.2e}, "
          f"reconstruction err = {recon_err:.2e}")

    eigvals, eigvecs = spectral.eigen_decompose(L)
    print(f"\nEigenvalues (sorted): {np.round(eigvals, 4)}")

    fiedler = spectral.fiedler_vector(eigvecs)
    print(f"Fiedler vector: {np.round(fiedler, 3)}")

    labels = spectral.two_way_split(fiedler)
    group_a = [i + 1 for i in range(9) if labels[i] == 0]
    group_b = [i + 1 for i in range(9) if labels[i] == 1]
    print(f"\nGroup A (people): {group_a}")
    print(f"Group B (people): {group_b}")

    visualize.plot_clusters(G, labels, "Toy Graph: Spectral Clustering Result",
                             os.path.join(RESULTS_DIR, "toy_clusters.png"))
    visualize.plot_fiedler_bar(fiedler, os.path.join(RESULTS_DIR, "toy_fiedler_vector.png"))
    print(f"\nFigures saved to {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    run()
