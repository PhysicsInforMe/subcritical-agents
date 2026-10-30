"""Graph generators returning networkx graphs, plus a neighbour-list view."""
from __future__ import annotations

import networkx as nx
import numpy as np


def lattice_2d(L: int) -> nx.Graph:
    """L x L periodic square lattice (4 neighbours)."""
    return nx.grid_2d_graph(L, L, periodic=True)


def complete_graph(N: int) -> nx.Graph:
    return nx.complete_graph(N)


def star(N: int) -> nx.Graph:
    """Hub 0 connected to N-1 leaves (supervisor topology)."""
    return nx.star_graph(N - 1)


def chain(N: int) -> nx.Graph:
    return nx.path_graph(N)


def small_world(N: int, k: int, p: float, seed: int) -> nx.Graph:
    return nx.watts_strogatz_graph(N, k, p, seed=seed)


def adjacency_lists(G: nx.Graph) -> list[np.ndarray]:
    """Neighbour arrays indexed 0..N-1 in the order of ``G.nodes``."""
    nodes = list(G.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    return [np.array([index[m] for m in G.neighbors(n)], dtype=np.int64) for n in nodes]
