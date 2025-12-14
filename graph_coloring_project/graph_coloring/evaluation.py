from typing import Any, Dict

import networkx as nx

from .algorithms import (
    greedy_coloring,
    dsatur_coloring,
    largest_degree_first,
    welsh_powell_coloring,
    random_greedy_coloring,
    genetic_algorithm_coloring,
)

# Verify if the coloring is valid (no two adjacent nodes have the same color)
def verify_coloring(graph: nx.Graph, colors: Dict[Any, int]) -> bool:
    for u, v in graph.edges():
        if colors.get(u) == colors.get(v):
            return False
    return True

# Run all algorithms on the given graph and return their results
def run_all_algorithms(
    graph: nx.Graph,
    use_genetic: bool = True,
    genetic_params: dict | None = None,
) -> Dict[str, dict]:

    results: Dict[str, dict] = {}

    algorithms = [
        ("Greedy", greedy_coloring),
        ("DSatur", dsatur_coloring),
        ("LDF", largest_degree_first),
        ("Welsh-Powell", welsh_powell_coloring),
        ("Random Greedy", lambda g: random_greedy_coloring(g, iterations=20)),
    ]

    for name, func in algorithms:
        colors, n_colors, t = func(graph)
        results[name] = {
            "colors": colors,
            "num_colors": n_colors,
            "time": t,
            "valid": verify_coloring(graph, colors),
        }

    if use_genetic:
        params = genetic_params or {"population_size": 50, "generations": 100}
        colors, n_colors, t = genetic_algorithm_coloring(graph, **params)
        results["Genetic"] = {
            "colors": colors,
            "num_colors": n_colors,
            "time": t,
            "valid": verify_coloring(graph, colors),
        }

    return results
