import random
import networkx as nx


# This function generates a random Erdős-Rényi graph.
def generate_random_graph(n: int, p: float, seed: int | None = None) -> nx.Graph:
    return nx.erdos_renyi_graph(n, p, seed=seed)


# This function generates a crown graph of size n.
def generate_crown_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    for i in range(n):
        G.add_node(f"a{i}")
        G.add_node(f"b{i}")

    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(f"a{i}", f"b{j}")

    return G


# This function generates a Mycielski graph of order k.
def generate_mycielski_graph(k: int) -> nx.Graph:
    return nx.mycielski_graph(k)


# This function generates an adversarial graph for greedy coloring algorithms.
def generate_adversarial_for_greedy(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    half = n // 2

    # dense connections in the first half
    for i in range(half):
        for j in range(i + 1, half):
            if random.random() < 0.6:
                G.add_edge(i, j)

    # dense-ish connections between halves
    for i in range(half):
        for j in range(half, n):
            if random.random() < 0.7:
                G.add_edge(i, j)

    # sparse connections in the second half
    for i in range(half, n):
        for j in range(i + 1, n):
            if random.random() < 0.2:
                G.add_edge(i, j)

    return G


# This function generates a bipartite-like graph with some noise.
def generate_bipartite_like_graph(n_left: int, n_right: int, p: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    left = [f"L{i}" for i in range(n_left)]
    right = [f"R{i}" for i in range(n_right)]
    G.add_nodes_from(left + right)

    # bipartite edges
    for u in left:
        for v in right:
            if random.random() < p:
                G.add_edge(u, v)

    # small noise inside each side
    for i in range(n_left):
        for j in range(i + 1, n_left):
            if random.random() < 0.1:
                G.add_edge(left[i], left[j])

    for i in range(n_right):
        for j in range(i + 1, n_right):
            if random.random() < 0.1:
                G.add_edge(right[i], right[j])

    return G
