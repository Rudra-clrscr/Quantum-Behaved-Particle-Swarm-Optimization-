"""
scripts/plot_benchmark_charts.py
---------------------------------
Benchmark figures drawn from measurements, never from assumed curves.

Two kinds of figure:

  1. Repeated-trial charts. Every algorithm is run once per seed on the same
     instance (app/core/benchmark_trials.py), and every trial is plotted -- a
     metaheuristic's result moves with its seed, so one run per algorithm only
     shows who won that run.

        data/trials_quality.png           fitness of every trial, per algorithm
        data/trials_runtime.png           wall-clock time of every trial
        data/trials_rates.png             feasible rate and success rate
        data/trials_time_vs_distance.png  fleet time against distance, per trial

  2. Scalability charts, read from the JSON that
     scripts/generate_stress_test_cache.py writes. A run that hit its time
     budget has no runtime to plot, so it is a gap in the line and is named
     under the chart, rather than drawn as a point claiming it finished at
     exactly the cutoff.

        data/scalability_<name>.png       runtime and fitness against customers

Every figure has a table twin in data/benchmark_charts.md.

    python scripts/plot_benchmark_charts.py                      # everything
    python scripts/plot_benchmark_charts.py --trials 10 --max-iter 150
    python scripts/plot_benchmark_charts.py --skip-trials         # scalability only

Ported, as static figures, from the dashboard charts added upstream in
MargdarshaQ (repeated benchmark runs, and charts drawn from measurements).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.core.benchmark_trials import ALL_ALGORITHMS, run_trials, summarize_trials
from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp

# ---------------------------------------------------------------------------
# Palette: the reference light theme. The first three categorical slots are the
# ones validated all-pairs (worst CVD dE 9.2, normal-vision 24.0); anything past
# three is muted gray and told apart by marker shape, never by a fourth hue.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]

ORDER = ["qpso", "standard_pso", "ga", "sa", "greedy"]
LABEL = {"qpso": "QPSO + LS", "standard_pso": "Standard PSO", "ga": "GA",
         "sa": "SA", "greedy": "Greedy NN"}
SCATTER_STYLE = {                     # (colour, marker)
    "qpso": (SLOT[0], "o"),
    "standard_pso": (SLOT[1], "o"),
    "ga": (SLOT[2], "o"),
    "sa": (MUTED, "s"),
    "greedy": (MUTED, "^"),
}

SCALE_STYLE = {
    "qpso_local_search": ("QPSO + local search", SLOT[0]),
    "standard_pso": ("Standard PSO", SLOT[1]),
}


def _style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
    })


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Repeated-trial charts
# ---------------------------------------------------------------------------
def _spread_plot(summary, key, scale, ylabel, title, path, log=False):
    """Every trial as a dot over a thin box. QPSO in slot 1, the rest muted."""
    algos = [a for a in ORDER if a in summary]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    rng = np.random.default_rng(0)       # jitter only; fixed so figures are stable

    for x, algo in enumerate(algos):
        vals = np.asarray(summary[algo][key], dtype=float) * scale
        colour = SLOT[0] if algo == "qpso" else MUTED
        ax.boxplot(vals, positions=[x], widths=0.42, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor=colour, linewidth=1),
                   medianprops=dict(color=colour, linewidth=2),
                   whiskerprops=dict(color=colour, linewidth=1),
                   capprops=dict(color=colour, linewidth=1))
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(x + jitter, vals, s=22, color=colour, edgecolors=SURFACE,
                   linewidths=1, zorder=3)

    ax.set_xticks(range(len(algos)), [LABEL[a] for a in algos])
    ax.set_xlim(-0.6, len(algos) - 0.4)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    _save(fig, path)


def _rates_plot(summary, threshold_pct, path):
    algos = [a for a in ORDER if a in summary]
    y = np.arange(len(algos))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    feas = [100 * summary[a]["feasible_rate"] for a in algos]
    succ = [100 * summary[a]["success_rate"] for a in algos]
    ax.barh(y - h / 2 - 0.02, feas, height=h, color=SLOT[0], label="Feasible")
    ax.barh(y + h / 2 + 0.02, succ, height=h, color=SLOT[1],
            label=f"Within {threshold_pct:g}% of best fitness found")
    ax.set_yticks(y, [LABEL[a] for a in algos])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of trials (%)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    _save(fig, path)


def _time_vs_distance_plot(summary, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for algo in ORDER:
        if algo not in summary:
            continue
        colour, marker = SCATTER_STYLE[algo]
        ax.scatter(summary[algo]["distance_samples"], summary[algo]["time_samples"],
                   s=30, color=colour, marker=marker, edgecolors=SURFACE,
                   linewidths=1, label=LABEL[algo], zorder=3)
    ax.grid(axis="x")
    ax.set_xlabel("Total distance (km)")
    ax.set_ylabel("Total fleet time (min)")
    ax.set_title("Fleet time against distance, one point per trial", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, path)


def run_trial_charts(args) -> List[str]:
    net = generate_synthetic_city_graph(n_nodes=40, seed=7)
    problem = generate_synthetic_vrp(net, n_customers=args.customers, depot=0,
                                     vehicle_capacity=80, seed=3)
    seeds = [args.seed + i for i in range(args.trials)]
    print(f"Running {len(ALL_ALGORITHMS)} algorithms x {len(seeds)} trials "
          f"({args.customers} customers, max_iter={args.max_iter})...")
    per_algo = run_trials(problem, ALL_ALGORITHMS, seeds, args.max_iter)
    summary = summarize_trials(per_algo, success_threshold_pct=args.success_pct)

    out = args.out_dir
    _spread_plot(summary, "fitness_samples", 1.0, "Fitness (lower is better)",
                 f"Solution quality over {args.trials} seeds",
                 os.path.join(out, "trials_quality.png"))
    _spread_plot(summary, "runtime_samples_ms", 1e-3, "Runtime per trial (s, log scale)",
                 f"Computation time over {args.trials} seeds",
                 os.path.join(out, "trials_runtime.png"), log=True)
    _rates_plot(summary, args.success_pct, os.path.join(out, "trials_rates.png"))
    _time_vs_distance_plot(summary, os.path.join(out, "trials_time_vs_distance.png"))

    def fmt(v, p=1):
        return f"{v:.{p}f}"

    lines = [
        "## Repeated trials",
        "",
        f"Instance: synthetic 40-node city graph (seed 7), {args.customers} customers "
        f"(seed 3), vehicle capacity 80. Seeds {seeds[0]}–{seeds[-1]}, "
        f"`max_iter={args.max_iter}`, 50 particles / population. Greedy NN is "
        "deterministic, so its trials are identical by construction.",
        "",
        f"\"Success\" means fitness within {args.success_pct:g}% of the best fitness any "
        "algorithm found in this run. The true optimum is unknown, so this measures "
        "agreement within the run, not distance from a proven optimum. QPSO runs with its "
        "memetic local search on. Trials run in parallel in a process pool, so runtimes "
        "include contention between them; compare runtimes within this table only.",
        "",
        "| Algorithm | Trials | Fitness median | Fitness min | Fitness max | Fitness mean ± std "
        "| Runtime median (s) | Feasible | Success |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for algo in ORDER:
        if algo not in summary:
            continue
        s = summary[algo]
        f = s["fitness_samples"]
        std = statistics.pstdev(f) if len(f) > 1 else 0.0
        lines.append(
            f"| {LABEL[algo]} | {s['trials']} | {fmt(statistics.median(f))} | {fmt(min(f))} "
            f"| {fmt(max(f))} | {fmt(statistics.mean(f))} ± {fmt(std)} "
            f"| {fmt(statistics.median(s['runtime_samples_ms']) / 1000, 2)} "
            f"| {100 * s['feasible_rate']:.0f}% | {100 * s['success_rate']:.0f}% |"
        )
    lines += ["", "Figures: `trials_quality.png`, `trials_runtime.png`, `trials_rates.png`, "
              "`trials_time_vs_distance.png`.", ""]
    return lines


# ---------------------------------------------------------------------------
# Scalability charts
# ---------------------------------------------------------------------------
def run_scalability_chart(json_path: str, name: str, out_dir: str) -> Optional[List[str]]:
    if not os.path.exists(json_path):
        print(f"  skipped {json_path}: not found")
        return None
    with open(json_path) as f:
        doc = json.load(f)

    results = doc["results"]
    sizes = sorted(int(k) for k in results)
    algos = [a for a in SCALE_STYLE if any(a in results[str(n)] for n in sizes)]
    budget = (doc.get("measurement") or {}).get("time_budget_seconds")

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    timeouts, infeasible = [], []
    for algo in algos:
        label, colour = SCALE_STYLE[algo]
        rt, fit = [], []
        for n in sizes:
            r = results[str(n)].get(algo) or {}
            done = r.get("fitness") is not None
            rt.append(r["runtime_ms"] / 1000 if done else np.nan)
            fit.append(r["fitness"] if done else np.nan)
            if not done:
                timeouts.append(f"{label} at {n}")
            elif not r.get("feasible"):
                infeasible.append((algo, n, r["fitness"]))
        ax_t.plot(sizes, rt, color=colour, linewidth=2, marker="o", markersize=5,
                  markeredgecolor=SURFACE, label=label)
        ax_f.plot(sizes, fit, color=colour, linewidth=2, marker="o", markersize=5,
                  markeredgecolor=SURFACE, label=label)

    # Infeasible results are hollow: they finished, but broke a constraint.
    for algo, n, fv in infeasible:
        ax_f.scatter([n], [fv], s=40, facecolor=SURFACE,
                     edgecolor=SCALE_STYLE[algo][1], linewidths=1.5, zorder=4)

    for ax in (ax_t, ax_f):
        ax.set_xticks(sizes)
        ax.set_xlabel("Customers")
        if len(sizes) <= 2:
            pad = max(10, (sizes[-1] - sizes[0]) * 0.25)
            ax.set_xlim(sizes[0] - pad, sizes[-1] + pad)
    ax_t.set_ylabel("Runtime of best seed (s)")
    ax_f.set_ylabel("Fitness of best seed")
    ax_t.set_title("Runtime", loc="left")
    ax_f.set_title("Solution quality (hollow = infeasible)", loc="left")
    ax_t.set_ylim(0, np.nanmax([l.get_ydata() for l in ax_t.lines]) * 1.15)
    ax_f.set_ylim(0, np.nanmax([l.get_ydata() for l in ax_f.lines]) * 1.1)
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 1.0),
               ncol=len(labels))

    notes = []
    if timeouts:
        cutoff = f" the {budget:g}s budget" if budget else " its time budget"
        notes.append("Did not finish within" + cutoff + ": " + "; ".join(timeouts) + ".")
    if notes:
        fig.text(0.01, -0.04, " ".join(notes), color=INK_2, fontsize=8, ha="left")
    png = os.path.join(out_dir, f"scalability_{name}.png")
    _save(fig, png)

    m = doc.get("measurement") or {}
    source = m.get("network_source") or doc.get("network_id", "unknown")
    lines = [
        f"## Scalability: `{os.path.basename(json_path)}`",
        "",
        f"Network: `{doc.get('network_id', source)}`. Generated {doc.get('generated_at', 'n/a')}. "
        + (f"Best of {m['seeds_per_point']} seeds per point, {m['time_budget_seconds']:g}s budget per run."
           if m else "The file does not record its settings; the generator that produced "
                     "it ran the best of 3 seeds per point with a 300s budget per run."),
        "",
        "| Customers | Algorithm | Runtime (s) | Fitness | Feasible |",
        "|---|---|---|---|---|",
    ]
    for n in sizes:
        for algo in algos:
            r = results[str(n)].get(algo) or {}
            if r.get("fitness") is None:
                lines.append(f"| {n} | {SCALE_STYLE[algo][0]} | timeout | — | — |")
            else:
                lines.append(f"| {n} | {SCALE_STYLE[algo][0]} | {r['runtime_ms'] / 1000:.1f} "
                             f"| {r['fitness']:.1f} | {'yes' if r.get('feasible') else 'no'} |")
    lines += ["", f"Figure: `{os.path.basename(png)}`.", ""]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=10, help="Seeds per algorithm (default: 10)")
    parser.add_argument("--seed", type=int, default=1, help="First seed (default: 1)")
    parser.add_argument("--max-iter", type=int, default=150, help="Iterations per run (default: 150)")
    parser.add_argument("--customers", type=int, default=18, help="Customers in the instance (default: 18)")
    parser.add_argument("--success-pct", type=float, default=5.0,
                        help="Success threshold, %% above the best fitness found (default: 5)")
    parser.add_argument("--skip-trials", action="store_true", help="Only redraw the scalability charts")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    _style()
    os.makedirs(args.out_dir, exist_ok=True)

    doc = ["# Benchmark charts — table view", "",
           "Generated by `scripts/plot_benchmark_charts.py`. Every value here is measured; "
           "each figure in `data/` has its numbers in one of the tables below.", ""]

    if not args.skip_trials:
        doc += run_trial_charts(args)

    for path, name in [("data/stress_test_synthetic.json", "synthetic"),
                       ("data/stress_test_delhi.json", "delhi")]:
        section = run_scalability_chart(path, name, args.out_dir)
        if section:
            doc += section

    md_path = os.path.join(args.out_dir, "benchmark_charts.md")
    if args.skip_trials and os.path.exists(md_path):
        # Keep the trial tables from the last full run; replace only scalability.
        with open(md_path, encoding="utf-8") as f:
            old = f.read()
        head = old.split("## Scalability:")[0].rstrip() + "\n\n"
        doc = [head] + [l for l in doc[4:]]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print(f"  wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
