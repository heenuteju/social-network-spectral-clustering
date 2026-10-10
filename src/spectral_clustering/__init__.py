"""
spectral_clustering
====================
A small, tested library for community detection in social networks using
the graph Laplacian, eigen-decomposition, and the Fiedler vector.

Modules
-------
graphs      : load toy / benchmark graphs
laplacian   : build adjacency, degree, and Laplacian matrices (unnormalized
              and normalized), plus structural checks (rank/nullity,
              diagonalization verification)
spectral    : eigen-decomposition, Fiedler vector, 2-way and k-way
              spectral clustering, eigengap heuristic for choosing k
metrics     : accuracy vs. ground truth, modularity, ARI, NMI
visualize   : plotting helpers (graph layout, eigenvalue spectrum, Fiedler bar)
"""

__version__ = "1.0.0"
