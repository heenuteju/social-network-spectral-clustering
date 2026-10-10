"""Build adjacency / degree / Laplacian matrices and verify their
structural properties (rank-nullity, symmetry, diagonalizability).
"""

from __future__ import annotations
import numpy as np
import networkx as nx


def adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """n x n adjacency matrix, A[i,j] = 1 if i and j are connected."""
    return nx.to_numpy_array(G)


def degree_matrix(A: np.ndarray) -> np.ndarray:
    """Diagonal matrix of node degrees (row sums of A)."""
    return np.diag(np.sum(A, axis=1))


def unnormalized_laplacian(A: np.ndarray) -> np.ndarray:
    """L = D - A. Symmetric, positive semi-definite. Every row sums
    to zero by construction, which guarantees the all-ones vector is
    always an eigenvector with eigenvalue 0.
    """
    D = degree_matrix(A)
    return D - A


def normalized_laplacian_sym(A: np.ndarray) -> np.ndarray:
    """Symmetric normalized Laplacian: L_sym = I - D^-1/2 A D^-1/2.
    Still symmetric (so eigh() applies), but rescales for degree
    heterogeneity -- often gives a cleaner split on real (non-toy)
    networks with very uneven degree distributions.
    """
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    n = A.shape[0]
    return np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt


def normalized_laplacian_rw(A: np.ndarray) -> np.ndarray:
    """Random-walk normalized Laplacian: L_rw = I - D^-1 A.
    NOT symmetric in general -- provided for completeness /
    comparison, but eigh() should not be used on it (use eig()).
    """
    deg = np.sum(A, axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = np.diag(deg_inv)
    n = A.shape[0]
    return np.eye(n) - D_inv @ A


def rank_nullity(L: np.ndarray) -> tuple[int, int]:
    """Return (rank, nullity) of L. nullity(L) equals the number of
    connected components of the underlying graph -- a graph is fully
    connected iff nullity == 1, which is the precondition for
    Fiedler's two-way-partition theorem to apply meaningfully.
    """
    rank = np.linalg.matrix_rank(L)
    nullity = L.shape[0] - rank
    return rank, nullity


def verify_diagonalization(L: np.ndarray) -> tuple[float, float]:
    """Numerically verify that a symmetric L is diagonalizable with an
    orthonormal eigenbasis: L = V D V^T with V^T V = I.

    Returns (orthogonality_error, reconstruction_error), both of which
    should be at machine precision (~1e-14) for a valid symmetric L.
    Uses numpy.linalg.eigh, which is the correct solver for symmetric
    matrices and guarantees orthonormal eigenvectors even when
    eigenvalues repeat (unlike numpy.linalg.eig).
    """
    D, V = np.linalg.eigh(L)
    ortho_err = float(np.max(np.abs(V.T @ V - np.eye(L.shape[0]))))
    recon_err = float(np.max(np.abs(V @ np.diag(D) @ V.T - L)))
    return ortho_err, recon_err
