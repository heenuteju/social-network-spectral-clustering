"""Eigen-decomposition, the Fiedler vector, and spectral clustering
(2-way via sign, k-way via k-means on the spectral embedding), plus
the eigengap heuristic for choosing k automatically.
"""

from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans


def eigen_decompose(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a symmetric Laplacian L, sorted ascending.

    Uses eigh() (not eig()) because L is symmetric: eigh() guarantees
    real eigenvalues and an orthonormal eigenvector matrix even when
    eigenvalues repeat, which plain eig() does not guarantee.

    Returns
    -------
    eigvals : (n,) array, sorted smallest -> largest
    eigvecs : (n, n) array, eigvecs[:, i] is the eigenvector for eigvals[i]
    """
    eigvals, eigvecs = np.linalg.eigh(L)
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


def fiedler_vector(eigvecs: np.ndarray, sign_fix: bool = True) -> np.ndarray:
    """Extract the Fiedler vector (eigenvector of the 2nd-smallest
    eigenvalue) from a sorted eigenvector matrix.

    Eigenvectors are only defined up to a sign (Lv = lambda*v implies
    L(-v) = lambda*(-v) too), so `sign_fix` optionally flips the vector
    so its first entry is positive, for a reproducible, comparable
    output across runs.
    """
    v2 = eigvecs[:, 1].copy()
    if sign_fix and v2[0] < 0:
        v2 = -v2
    return v2


def two_way_split(fiedler: np.ndarray) -> np.ndarray:
    """Split nodes into two clusters by the sign of the Fiedler vector.
    Returns a (n,) array of labels in {0, 1}: label 0 for entries > 0,
    label 1 for entries <= 0.
    """
    return (fiedler <= 0).astype(int)


def eigengap_heuristic(eigvals: np.ndarray, max_k: int = 10) -> int:
    """Suggest a number of clusters k by finding the largest gap
    between consecutive sorted eigenvalues (the 'eigengap heuristic').
    A large gap after the k-th eigenvalue suggests k clusters are a
    natural fit for the graph's structure.
    """
    max_k = min(max_k, len(eigvals) - 1)
    gaps = np.diff(eigvals[: max_k + 1])
    # +1 because gaps[i] is the gap AFTER the (i+1)-th eigenvalue (1-indexed)
    best_k = int(np.argmax(gaps)) + 1
    return max(best_k, 2)


def k_way_spectral_clustering(L: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    """General k-way spectral clustering: embed each node using the
    k eigenvectors of smallest eigenvalue, then run k-means on those
    embedded points. For k=2 this is equivalent in spirit to the sign
    based Fiedler split, but generalizes to any number of clusters.
    """
    eigvals, eigvecs = eigen_decompose(L)
    embedding = eigvecs[:, :k]                      # n x k spectral embedding
    # row-normalize (standard practice, improves k-means stability)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding_normalized = embedding / norms
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    return km.fit_predict(embedding_normalized)
                     