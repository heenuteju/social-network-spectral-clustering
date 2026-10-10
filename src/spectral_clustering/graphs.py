"""Load the graphs used in this project."""

import networkx as nx


def load_toy_graph() -> nx.Graph:
    """A 9-node toy social network with two obvious communities
    ({1..4} and {5..9}) bridged by a single edge (3-5). Used to
    demonstrate every step of the pipeline on a small, visualizable
    example before applying it to real data.

    Nodes are 0-indexed internally (0..8) to match standard graph
    library conventions; add 1 when displaying "person numbers".
    """
    edges = [(0, 1), (0, 3), (1, 2), (2, 3), (2, 4),
             (4, 5), (4, 6), (4, 7), (4, 8),
             (5, 6), (6, 7), (6, 8), (7, 8)]
    G = nx.Graph()
    G.add_nodes_from(range(9))
    G.add_edges_from(edges)
    return G


def load_karate_club() -> nx.Graph:
    """Zachary's Karate Club (1970-1972): 34 members, 78 friendships.
    Includes the real historical split as a node attribute 'club'
    ('Mr. Hi' or 'Officer'), used as ground truth for evaluation.
    """
    return nx.karate_club_graph()


def ground_truth_labels(G: nx.Graph) -> dict:
    """Return {node: 0/1} ground-truth labels for a graph that has a
    'club' node attribute (currently only load_karate_club()).
    0 = 'Mr. Hi' faction, 1 = 'Officer' faction.
    """
    club = nx.get_node_attributes(G, "club")
    return {n: (0 if c == "Mr. Hi" else 1) for n, c in club.items()}
