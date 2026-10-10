"""Plotting helpers. Kept separate from the analysis code so the core
library has no hard requirement on a display backend.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def plot_clusters(G: nx.Graph, labels: np.ndarray, title: str, save_path: str,
                   pos: dict | None = None) -> dict:
    """Draw the graph with nodes colored by cluster label. Returns the
    layout used (pass it back in for a second plot to keep node
    positions consistent, e.g. predicted vs. ground truth).
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    colors = [PALETTE[int(lab) % len(PALETTE)] for lab in labels]
    plt.figure(figsize=(6, 5))
    nx.draw(G, pos, node_color=colors, with_labels=True, font_color="white",
            font_size=8, node_size=380, edge_color="#bbbbbb")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close()
    return pos


def plot_eigenvalue_spectrum(eigvals: np.ndarray, save_path: str, max_k: int = 10) -> None:
    """Scree plot of the smallest eigenvalues -- used to visually
    justify the eigengap heuristic's choice of k.
    """
    k = min(max_k, len(eigvals))
    plt.figure(figsize=(6, 4))
    plt.plot(range(k), eigvals[:k], "o-", color="#4C72B0")
    plt.xlabel("Eigenvalue index (sorted ascending)")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue spectrum (eigengap heuristic)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close()


def plot_fiedler_bar(fiedler: np.ndarray, save_path: str) -> None:
    """Bar chart of Fiedler vector values, colored by sign."""
    colors = [PALETTE[0] if v > 0 else PALETTE[1] for v in fiedler]
    plt.figure(figsize=(7, 3.5))
    plt.bar(range(len(fiedler)), fiedler, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Node index")
    plt.ylabel("Fiedler vector value")
    plt.title("Fiedler vector: sign determines cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close()
