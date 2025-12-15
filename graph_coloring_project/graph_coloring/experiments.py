from typing import List

import numpy as np
import matplotlib.pyplot as plt

from .graph_generators import (
    generate_random_graph,
    generate_mycielski_graph,
    generate_crown_graph,
    generate_adversarial_for_greedy,
    generate_bipartite_like_graph,
)
from .evaluation import run_all_algorithms
from .visualization import (
    visualize_colored_graph,
    plot_comparison_bar,
    plot_time_comparison,
    save_and_show,
)


# The first experiment will compare all algorithms on a random graph
def experiment_random_graph_comparison() -> None:
    print("=== Experiment 1: Random Graph Comparison ===")
    G = generate_random_graph(80, 0.35, seed=42)
    results = run_all_algorithms(
        G,
        use_genetic=True,
        genetic_params={"population_size": 60, "generations": 80},
    )

    for name, r in results.items():
        print(
            f"{name:<14} colors={r['num_colors']:<3}  "
            f"time={r['time'] * 1000:>7.2f} ms  valid={r['valid']}"
        )

    alg_order = list(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    # top row: graph visualizations (limit to first 4 algorithms)
    for i, alg in enumerate(alg_order[:4]):
        visualize_colored_graph(
            G,
            results[alg]["colors"],
            title=f"{alg}",
            ax=axes[i],
        )

    # bottom row: bar charts
    plot_comparison_bar(results, ax=axes[4], title="Colors on Random Graph (n=80,p=0.35)")
    plot_time_comparison(results, ax=axes[5], title="Time on Random Graph (n=80,p=0.35)")

    fig.suptitle("Experiment 1: Random Graph Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp1_random_comparison.png")


# The second experiment will focus on challenging graph structures
def experiment_challenging_graph_structures() -> None:
    print("\n=== Experiment 2: Challenging Graph Structures ===")

    graphs = {
        "Mycielski-5": (generate_mycielski_graph(5), 5),
        "Crown-8": (generate_crown_graph(8), 2),
        "Adversarial-30": (generate_adversarial_for_greedy(30), None),
        "Bipartite-like": (generate_bipartite_like_graph(15, 15, 0.4), None),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    for idx, (name, (G, optimal)) in enumerate(graphs.items()):
        print(f"\n>>> {name}")
        print(f"Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}, Optimal={optimal}")
        results = run_all_algorithms(
            G,
            use_genetic=True,
            genetic_params={"population_size": 40, "generations": 60},
        )

        for alg, r in results.items():
            print(
                f"{alg:<14} colors={r['num_colors']:<3} "
                f"time={r['time'] * 1000:>7.2f} ms  valid={r['valid']}"
            )

        # top row: graph drawing
        visualize_colored_graph(G, results["DSatur"]["colors"], f"{name} (DSatur)", axes[idx])

        # bottom row: color count bar
        ax_bar = axes[idx + 4]
        plot_comparison_bar(results, ax=ax_bar, title=f"{name} - colors")

    fig.suptitle("Experiment 2: Challenging Graph Structures", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp2_challenging_graphs.png")


# The third experiment will test scalability by varying n
def experiment_scalability() -> None:
    print("\n=== Experiment 3: Scalability Test (vary n) ===")
    sizes = [30, 50, 80, 100, 150, 200]
    alg_names = ["Greedy", "DSatur", "LDF", "Welsh-Powell", "Random Greedy", "Genetic"]

    color_stats = {alg: [] for alg in alg_names}
    time_stats = {alg: [] for alg in alg_names}

    for n in sizes:
        G = generate_random_graph(n, 0.3, seed=42 + n)
        gen_params = {"population_size": max(20, 60 - n // 5), "generations": max(30, 80 - n // 3)}
        results = run_all_algorithms(G, use_genetic=True, genetic_params=gen_params)

        print(f"\n-- n = {n} --")
        for alg in alg_names:
            r = results[alg]
            color_stats[alg].append(r["num_colors"])
            time_stats[alg].append(r["time"] * 1000)
            print(
                f"{alg:<14} colors={r['num_colors']:<3} "
                f"time={r['time'] * 1000:>7.2f} ms  valid={r['valid']}"
            )

    # plot colors vs n, time vs n
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for alg, series in color_stats.items():
        ax1.plot(sizes, series, marker="o", label=alg)
    ax1.set_xlabel("Number of Vertices (n)")
    ax1.set_ylabel("Number of Colors")
    ax1.set_title("Colors vs Graph Size (p=0.3)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    for alg, series in time_stats.items():
        ax2.plot(sizes, series, marker="o", label=alg)
    ax2.set_xlabel("Number of Vertices (n)")
    ax2.set_ylabel("Time (ms, log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Runtime vs Graph Size (p=0.3)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.suptitle("Experiment 3: Scalability", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp3_scalability.png")


# The fourth experiment will test the effect of density by varying p
def experiment_density() -> None:
    print("\n=== Experiment 4: Density Analysis (vary p) ===")
    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n_nodes = 60
    alg_names = ["Greedy", "DSatur", "LDF", "Welsh-Powell", "Random Greedy", "Genetic"]

    color_stats = {alg: [] for alg in alg_names}
    time_stats = {alg: [] for alg in alg_names}

    for p in densities:
        G = generate_random_graph(n_nodes, p, seed=int(p * 1000))
        results = run_all_algorithms(
            G,
            use_genetic=True,
            genetic_params={"population_size": 40, "generations": 50},
        )

        print(f"\n-- p = {p:.2f} --")
        for alg in alg_names:
            r = results[alg]
            color_stats[alg].append(r["num_colors"])
            time_stats[alg].append(r["time"] * 1000)
            print(
                f"{alg:<14} colors={r['num_colors']:<3} "
                f"time={r['time'] * 1000:>7.2f} ms  valid={r['valid']}"
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for alg, series in color_stats.items():
        ax1.plot(densities, series, marker="o", label=alg)
    ax1.set_xlabel("Edge Probability p")
    ax1.set_ylabel("Number of Colors")
    ax1.set_title(f"Colors vs Density (n={n_nodes})")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    for alg, series in time_stats.items():
        ax2.plot(densities, series, marker="o", label=alg)
    ax2.set_xlabel("Edge Probability p")
    ax2.set_ylabel("Time (ms, log scale)")
    ax2.set_yscale("log")
    ax2.set_title(f"Runtime vs Density (n={n_nodes})")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.suptitle("Experiment 4: Density Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp4_density.png")


# The fifth experiment will analyze crown graphs
def experiment_crown_analysis() -> None:
    # Crown-n scaling
    ns = [4, 6, 8, 10]
    greedy_colors, dsatur_colors, ga_colors = [], [], []
    opt_colors = [2] * len(ns)

    for n in ns:
        G = generate_crown_graph(n)
        results = run_all_algorithms(G)
        greedy_colors.append(results["Greedy"]["num_colors"])
        dsatur_colors.append(results["DSatur"]["num_colors"])
        ga_colors.append(results["Genetic"]["num_colors"])

    # Representative instance (Crown-8) for visualization + single-instance comparison
    n0 = 8
    G0 = generate_crown_graph(n0)
    results0 = run_all_algorithms(G0)

    # ---- Figure layout ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: graph visualizations (Crown-8)
    visualize_colored_graph(
        G0,
        results0["Greedy"]["colors"],
        title=f"Greedy: {results0['Greedy']['num_colors']} colors",
        ax=axes[0, 0],
    )
    visualize_colored_graph(
        G0,
        results0["DSatur"]["colors"],
        title=f"DSatur: {results0['DSatur']['num_colors']} colors",
        ax=axes[0, 1],
    )
    visualize_colored_graph(
        G0,
        results0["Genetic"]["colors"],
        title=f"Genetic: {results0['Genetic']['num_colors']} colors",
        ax=axes[0, 2],
    )

    # Bottom-left: scaling line chart
    ax = axes[1, 0]
    ax.plot(ns, greedy_colors, marker="o", label="Greedy")
    ax.plot(ns, dsatur_colors, marker="s", label="DSatur")
    ax.plot(ns, ga_colors, marker="^", label="Genetic")
    ax.plot(ns, opt_colors, linestyle="--", label="Optimal=2")
    ax.set_title("Crown Graph: Color Comparison")
    ax.set_xlabel("Crown Graph Parameter n")
    ax.set_ylabel("Number of Colors")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # Bottom-middle: single-instance bar comparison (Crown-8)
    plot_comparison_bar(results0, ax=axes[1, 1], title="Crown-8: Algorithm Comparison")
    axes[1, 1].tick_params(axis="x", labelrotation=35)

    # Bottom-right: improvement over greedy (%)
    ax = axes[1, 2]
    improvements_dsatur = [(g - d) / g * 100 if g > 0 else 0.0 for g, d in zip(greedy_colors, dsatur_colors)]
    improvements_ga = [(g - a) / g * 100 if g > 0 else 0.0 for g, a in zip(greedy_colors, ga_colors)]

    x = np.arange(len(ns))
    width = 0.36
    ax.bar(x - width / 2, improvements_dsatur, width, label="DSatur vs Greedy")
    ax.bar(x + width / 2, improvements_ga, width, label="Genetic vs Greedy")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_title("Heuristic Improvement over Greedy")
    ax.set_xlabel("Crown Graph Parameter n")
    ax.set_ylabel("Improvement (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for i, v in enumerate(improvements_dsatur):
        ax.text(i - width / 2, v + 1.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(improvements_ga):
        ax.text(i + width / 2, v + 1.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Crown Graph: Adversarial Case Analysis", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # IMPORTANT: match your report's expected filename
    save_and_show(fig, "exp5_adversarial.png")



# The last experiment will perform statistical analysis over multiple trials
def experiment_statistics(num_trials: int = 30) -> None:
    print("\n=== Experiment 6: Statistical Analysis ===")
    n_nodes, p_edges = 60, 0.35
    alg_names = ["Greedy", "DSatur", "LDF", "Welsh-Powell", "Random Greedy", "Genetic"]

    stats = {alg: {"colors": [], "times": []} for alg in alg_names}

    for trial in range(num_trials):
        G = generate_random_graph(n_nodes, p_edges, seed=2000 + trial)
        results = run_all_algorithms(
            G,
            use_genetic=True,
            genetic_params={"population_size": 30, "generations": 40},
        )
        for alg in alg_names:
            r = results[alg]
            stats[alg]["colors"].append(r["num_colors"])
            stats[alg]["times"].append(r["time"] * 1000)

    print(f"{'Algorithm':<15} {'Mean':<7} {'Min':<5} {'Max':<5} {'Std':<7} {'Time(ms)':<10}")
    for alg in alg_names:
        colors = np.array(stats[alg]["colors"])
        times = np.array(stats[alg]["times"])
        print(
            f"{alg:<15} {colors.mean():<7.2f} {colors.min():<5d} {colors.max():<5d} "
            f"{colors.std():<7.2f} {times.mean():<10.2f}"
        )

    # boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.boxplot([stats[alg]["colors"] for alg in alg_names], labels=alg_names)
    ax1.set_ylabel("Number of Colors")
    ax1.set_title("Distribution of Color Counts")

    ax2.boxplot([stats[alg]["times"] for alg in alg_names], labels=alg_names)
    ax2.set_yscale("log")
    ax2.set_ylabel("Time (ms, log scale)")
    ax2.set_title("Distribution of Runtimes")

    fig.suptitle("Experiment 6: Statistical Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp6_statistics.png")


def run_all_experiments() -> None:
    experiment_random_graph_comparison()
    experiment_challenging_graph_structures()
    experiment_scalability()
    experiment_density()
    experiment_crown_analysis()
    experiment_statistics()
    print("\nAll experiments completed. Figures saved to 'img/' folder.")
