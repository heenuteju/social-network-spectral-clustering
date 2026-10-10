"""Unit tests validating the mathematical properties the whole project
depends on: symmetry, positive semi-definiteness, rank-nullity vs.
connected components, exact diagonalizability, and end-to-end accuracy
on the one dataset with a known correct answer (Zachary's Karate Club).

Run with:  pytest tests/ -v
"""

import sys
import os
import numpy as np
import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectral_clustering import graphs, laplacian, spectral, metrics


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def toy_graph():
    return graphs.load_toy_graph()


@pytest.fixture
def karate_graph():
    return graphs.load_karate_club()


# ---------------------------------------------------------------------
# Laplacian structural properties
# ---------------------------------------------------------------------
def test_laplacian_is_symmetric(toy_graph):
    A = laplacian.adjacency_matrix(toy_graph)
    L = laplacian.unnormalized_laplacian(A)
    assert np.allclose(L, L.T)


def test_laplacian_rows_sum_to_zero(toy_graph):
    A = laplacian.adjacency_matrix(toy_graph)
    L = laplacian.unnormalized_laplacian(A)
    assert np.allclose(L @ np.ones(L.shape[0]), 0)


def test_laplacian_is_positive_semidefinite(karate_graph):
    A = laplacian.adjacency_matrix(karate_graph)
    L = laplacian.unnormalized_laplacian(A)
    eigvals = np.linalg.eigvalsh(L)
    assert np.all(eigvals > -1e-9)  # allow tiny float error, never truly negative


@pytest.mark.parametrize("graph_fixture,expected_components", [
    ("toy_graph", 1),
    ("karate_graph", 1),
])
def test_rank_nullity_matches_connected_components(graph_fixture, expected_components, request):
    G = request.getfixturevalue(graph_fixture)
    A = laplacian.adjacency_matrix(G)
    L = laplacian.unnormalized_laplacian(A)
    rank, nullity = laplacian.rank_nullity(L)
    assert nullity == expected_components
    assert nullity == nx.number_connected_components(G)
    assert rank + nullity == L.shape[0]


def test_nullity_counts_disconnected_components():
    """Sanity check on a graph we KNOW is disconnected: two separate
    triangles should give nullity = 2, not 1.
    """
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])       # triangle 1
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])        # triangle 2 (disconnected)
    A = laplacian.adjacency_matrix(G)
    L = laplacian.unnormalized_laplacian(A)
    _, nullity = laplacian.rank_nullity(L)
    assert nullity == 2


# ---------------------------------------------------------------------
# Diagonalization
# ---------------------------------------------------------------------
def test_diagonalization_is_exact(karate_graph):
    A = laplacian.adjacency_matrix(karate_graph)
    L = laplacian.unnormalized_laplacian(A)
    ortho_err, recon_err = laplacian.verify_diagonalization(L)
    assert ortho_err < 1e-8
    assert recon_err < 1e-8


# ---------------------------------------------------------------------
# Eigen-decomposition / Fiedler vector
# ---------------------------------------------------------------------
def test_smallest_eigenvalue_is_zero(toy_graph):
    A = laplacian.adjacency_matrix(toy_graph)
    L = laplacian.unnormalized_laplacian(A)
    eigvals, _ = spectral.eigen_decompose(L)
    assert abs(eigvals[0]) < 1e-9


def test_fiedler_value_is_positive_for_connected_graph(karate_graph):
    A = laplacian.adjacency_matrix(karate_graph)
    L = laplacian.unnormalized_laplacian(A)
    eigvals, _ = spectral.eigen_decompose(L)
    assert eigvals[1] > 0   # Fiedler value (algebraic connectivity) > 0 <=> connected


def test_toy_graph_splits_as_expected(toy_graph):
    A = laplacian.adjacency_matrix(toy_graph)
    L = laplacian.unnormalized_laplacian(A)
    eigvals, eigvecs = spectral.eigen_decompose(L)
    fiedler = spectral.fiedler_vector(eigvecs)
    labels = spectral.two_way_split(fiedler)
    # nodes 0-3 (people 1-4) should all share one label,
    # nodes 4-8 (people 5-9) should all share the other label
    assert len(set(labels[0:4])) == 1
    assert len(set(labels[4:9])) == 1
    assert labels[0] != labels[4]


# ---------------------------------------------------------------------
# End-to-end accuracy on real data (the one dataset with ground truth)
# ---------------------------------------------------------------------
def test_karate_club_prediction_accuracy(karate_graph):
    truth = graphs.ground_truth_labels(karate_graph)
    true_labels = np.array([truth[n] for n in karate_graph.nodes()])

    A = laplacian.adjacency_matrix(karate_graph)
    L = laplacian.unnormalized_laplacian(A)
    eigvals, eigvecs = spectral.eigen_decompose(L)
    fiedler = spectral.fiedler_vector(eigvecs)
    pred_labels = spectral.two_way_split(fiedler)

    acc = metrics.accuracy_vs_ground_truth(pred_labels, true_labels)
    # historically, spectral bisection on this dataset gets 33/34 (97.1%);
    # assert a safe lower bound so the test isn't brittle to float noise
    assert acc >= 0.90


def test_eigengap_heuristic_suggests_two_for_karate(karate_graph):
    A = laplacian.adjacency_matrix(karate_graph)
    L = laplacian.unnormalized_laplacian(A)
    eigvals, _ = spectral.eigen_decompose(L)
    k = spectral.eigengap_heuristic(eigvals, max_k=6)
    assert k >= 2  # method should never suggest fewer than 2 clusters


def test_k_way_clustering_runs_and_returns_correct_shape(karate_graph):
    A = laplacian.adjacency_matrix(karate_graph)
    L = laplacian.unnormalized_laplacian(A)
    labels = spectral.k_way_spectral_clustering(L, k=4)
    assert labels.shape[0] == karate_graph.number_of_nodes()
    assert len(set(labels)) <= 4


# ---------------------------------------------------------------------
# Metrics sanity checks
# ---------------------------------------------------------------------
def test_metrics_perfect_agreement():
    true = np.array([0, 0, 1, 1])
    pred = np.array([1, 1, 0, 0])  # same partition, flipped labels
    assert metrics.accuracy_vs_ground_truth(pred, true) == 1.0
    assert metrics.ari(true, pred) == pytest.approx(1.0)
    assert metrics.nmi(true, pred) == pytest.approx(1.0)


def test_modularity_positive_for_real_community_structure(karate_graph):
    A = laplacian.adjacency_matrix(karate_graph)
    L = laplacian.unnormalized_laplacian(A)
    eigvals, eigvecs = spectral.eigen_decompose(L)
    fiedler = spectral.fiedler_vector(eigvecs)
    labels = spectral.two_way_split(fiedler)
    Q = metrics.modularity(karate_graph, labels)
    assert Q > 0.3  # genuine community structure, not a random split
