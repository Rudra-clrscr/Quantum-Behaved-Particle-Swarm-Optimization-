#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp
from app.core.benchmark_trials import run_trials, summarize_trials

def main():
    print("Generating time-dependent robustness experiment...")
    net = generate_synthetic_city_graph(n_nodes=40, seed=42)
    problem = generate_synthetic_vrp(net, n_customers=20, depot=0, vehicle_capacity=80, time_dependent=True, seed=1)

    algorithms = ["qpso", "standard_pso", "ga", "sa", "greedy"]
    seeds = [1, 2, 3, 4, 5]
    max_iter = 150

    per_algo = run_trials(problem, algorithms, seeds, max_iter=max_iter)
    summary = summarize_trials(per_algo)

    md = [
        "# Time-Dependent Baseline Comparison",
        "",
        "This experiment measures whether QPSO+LS remains the strongest algorithm when ",
        "routing over a time-dependent congestion profile. Evaluated over 5 seeds on a ",
        "20-customer synthetic instance.",
        "",
        "| Algorithm | Mean Fitness | Best | Feasible Rate | Median Runtime (ms) |",
        "|---|---|---|---|---|",
    ]

    import statistics

    def fmt_fitness(s):
        if not s:
            return "N/A"
        return f"{statistics.mean(s):.1f} ± {statistics.stdev(s):.1f}" if len(s) > 1 else f"{s[0]:.1f}"

    def fmt_best(s):
        return f"{min(s):.1f}" if s else "N/A"

    def fmt_runtime(s):
        return f"{statistics.median(s):.0f}" if s else "N/A"

    for algo in algorithms:
        if algo not in summary:
            continue
        stats = summary[algo]
        fit = stats["fitness_samples"]
        rt = stats["runtime_samples_ms"]
        md.append(
            f"| {stats['name']} | {fmt_fitness(fit)} | {fmt_best(fit)} | "
            f"{stats['feasible_rate']*100:.0f}% | {fmt_runtime(rt)} |"
        )

    md.append("")

    os.makedirs("data", exist_ok=True)
    out_path = "data/time_dependent_results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Report written to {out_path}")

if __name__ == "__main__":
    main()
