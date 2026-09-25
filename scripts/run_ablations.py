"""
scripts/run_ablations.py
-------------------------
The experiments behind the paper's three algorithmic claims, each over several
seeds so a claim never rests on one lucky run.

  A. Optimality on small instances. QPSO (+ local search) against the exact
     optimum from app/core/exact_vrp.py, on instances small enough to solve
     exactly. Reports the optimality gap and how often QPSO reaches the optimum.

  B. Jump-cap ablation. QPSO with the dimension-dependent cap on ln(1/u)
     against the same QPSO with the cap removed. Local search is OFF in both,
     so the difference is the cap alone.

  C. Local-search ablation. QPSO with and without the memetic local search,
     against standard PSO and GA, at the same particle / population budget.

Writes data/ablation_results.md (tables), data/ablation_results.json (every
raw run) and two figures:

    data/ablation_jump_cap.png
    data/ablation_local_search.png

    python scripts/run_ablations.py
    python scripts/run_ablations.py --seeds 10 --sizes 20 40 60 80
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.core.classical_baselines_vrp import run_ga_vrp, run_standard_pso_vrp
from app.core.exact_vrp import optimality_gap, solve_vrp_exact
from app.core.graph_model import generate_synthetic_city_graph
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.vrp_problem import generate_synthetic_vrp

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]

N_PARTICLES = 50


# ---------------------------------------------------------------------------
# Workers (module level so a process pool can pickle them)
# ---------------------------------------------------------------------------
def _instance(n_customers: int, instance_seed: int, graph_nodes: int):
    net = generate_synthetic_city_graph(n_nodes=graph_nodes, seed=instance_seed)
    return generate_synthetic_vrp(net, n_customers=n_customers, depot=0,
                                  vehicle_capacity=80, seed=instance_seed)


def _run(task):
    """One run. task = (config, n_customers, instance_seed, graph_nodes, seed, max_iter)."""
    config, n, inst, nodes, seed, max_iter = task
    problem = _instance(n, inst, nodes)
    t0 = time.perf_counter()
    if config == "exact":
        r = solve_vrp_exact(problem)
        sol = r.best_solution
    elif config in ("qpso_ls", "qpso", "qpso_nocap"):
        sol = QPSOVRPOptimizer(
            problem, n_particles=N_PARTICLES, max_iter=max_iter, seed=seed,
            use_local_search=(config == "qpso_ls"),
            max_jump_factor=(math.inf if config == "qpso_nocap" else None),
        ).optimize().best_solution
    elif config == "standard_pso":
        sol = run_standard_pso_vrp(problem, n_particles=N_PARTICLES, max_iter=max_iter,
                                   seed=seed).best_solution
    elif config == "ga":
        sol = run_ga_vrp(problem, pop_size=N_PARTICLES, max_iter=max_iter, seed=seed).best_solution
    else:
        raise ValueError(config)
    return {"config": config, "n": n, "instance": inst, "seed": seed,
            "fitness": sol.fitness, "feasible": bool(sol.feasible),
            "runtime_s": time.perf_counter() - t0}


def _pool_map(tasks, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, tasks))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    })


def _gap_plot(runs, configs, sizes, title, path):
    """
    Fitness as % above the best any configuration found at that size, so every
    size shares one axis. Each seed is a dot; the line joins the medians.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    offsets = np.linspace(-0.18, 0.18, len(configs)) if len(configs) > 1 else [0.0]
    for (config, label, colour, marker), off in zip(configs, offsets):
        medians = []
        for x, n in enumerate(sizes):
            best = min(r["fitness"] for r in runs if r["n"] == n)
            vals = [100 * (r["fitness"] - best) / best
                    for r in runs if r["n"] == n and r["config"] == config]
            medians.append(statistics.median(vals))
            ax.scatter([x + off] * len(vals), vals, s=18, color=colour, marker=marker,
                       edgecolors=SURFACE, linewidths=1, alpha=0.9, zorder=3)
        ax.plot(np.arange(len(sizes)) + off, medians, color=colour, linewidth=2,
                marker=marker, markersize=6, markeredgecolor=SURFACE, label=label, zorder=4)
    ax.set_xticks(range(len(sizes)), [str(n) for n in sizes])
    ax.set_xlabel("Customers")
    ax.set_ylabel("% above best fitness found at this size")
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 1.0),
               ncol=len(labels))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _summary_rows(runs, configs, sizes):
    rows = []
    for n in sizes:
        for config, label, *_ in configs:
            rs = [r for r in runs if r["n"] == n and r["config"] == config]
            f = [r["fitness"] for r in rs]
            rows.append(
                f"| {n} | {label} | {statistics.mean(f):.1f} ± {statistics.pstdev(f):.1f} "
                f"| {statistics.median(f):.1f} | {min(f):.1f} "
                f"| {sum(r['feasible'] for r in rs)}/{len(rs)} "
                f"| {statistics.median(r['runtime_s'] for r in rs):.1f} |"
            )
    return rows


def _paired_wins(runs, a, b, sizes):
    """Seeds on which config a beat config b (same instance, same seed)."""
    out = []
    for n in sizes:
        fa = {r["seed"]: r["fitness"] for r in runs if r["n"] == n and r["config"] == a}
        fb = {r["seed"]: r["fitness"] for r in runs if r["n"] == n and r["config"] == b}
        seeds = sorted(set(fa) & set(fb))
        wins = sum(1 for s in seeds if fa[s] < fb[s] - 1e-9)
        ties = sum(1 for s in seeds if abs(fa[s] - fb[s]) <= 1e-9)
        out.append((n, wins, ties, len(seeds)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per configuration (default: 5)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60],
                        help="Customer counts for ablations B and C (default: 20 40 60)")
    parser.add_argument("--exact-sizes", type=int, nargs="+", default=[6, 7, 8, 9],
                        help="Customer counts for the exact comparison (default: 6 7 8 9)")
    parser.add_argument("--exact-instances", type=int, default=3,
                        help="Instances per exact size (default: 3)")
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    _style()
    seeds = list(range(1, args.seeds + 1))
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- A. exact optimality -------------------------------------------------
    print("A. QPSO against the exact optimum...")
    a_tasks = []
    for n in args.exact_sizes:
        for inst in range(1, args.exact_instances + 1):
            a_tasks.append(("exact", n, inst, 40, 0, args.max_iter))
            a_tasks += [("qpso_ls", n, inst, 40, s, args.max_iter) for s in seeds]
    a_runs = _pool_map(a_tasks, args.workers)

    # ---- B + C. ablations ----------------------------------------------------
    print("B/C. Jump-cap and local-search ablations...")
    bc_configs = ["qpso_ls", "qpso", "qpso_nocap", "standard_pso", "ga"]
    bc_tasks = [(c, n, 1, 150, s, args.max_iter)
                for n in args.sizes for c in bc_configs for s in seeds]
    bc_runs = _pool_map(bc_tasks, args.workers)

    with open("data/ablation_results.json", "w") as f:
        json.dump({"generated_at": started, "settings": vars(args),
                   "exact": a_runs, "ablations": bc_runs}, f, indent=2)
    print("  wrote data/ablation_results.json")

    # ---- figures -------------------------------------------------------------
    jump_cfg = [("qpso", "QPSO with jump cap", SLOT[0], "o"),
                ("qpso_nocap", "QPSO without jump cap", SLOT[1], "s")]
    ls_cfg = [("qpso_ls", "QPSO + local search", SLOT[0], "o"),
              ("qpso", "QPSO alone", SLOT[1], "s"),
              ("standard_pso", "Standard PSO", SLOT[2], "^"),
              ("ga", "GA", MUTED, "D")]
    _gap_plot([r for r in bc_runs if r["config"] in ("qpso", "qpso_nocap")], jump_cfg,
              args.sizes, f"Jump-cap ablation, local search off ({args.seeds} seeds)",
              "data/ablation_jump_cap.png")
    _gap_plot([r for r in bc_runs if r["config"] in ("qpso_ls", "qpso", "standard_pso", "ga")],
              ls_cfg, args.sizes, f"Local-search ablation ({args.seeds} seeds)",
              "data/ablation_local_search.png")

    # ---- tables --------------------------------------------------------------
    md = [
        "# Ablation results",
        "",
        f"_Generated by `scripts/run_ablations.py` on {started[:10]} with `--seeds {args.seeds} "
        f"--max-iter {args.max_iter}`. Every raw run is in `data/ablation_results.json`. "
        "Re-run the script to reproduce every figure below._",
        "",
        f"All metaheuristics use {N_PARTICLES} particles (or a population of {N_PARTICLES}) and "
        f"{args.max_iter} iterations. Instances are synthetic: a city graph and customer set "
        "generated from a fixed seed, vehicle capacity 80, objective weights "
        "$w_T = 0.6$, $w_D = 0.4$ (docs/FORMULATION.md §4). Fitness is lower-is-better.",
        "",
        "## A. QPSO against the exact optimum",
        "",
        f"{args.exact_instances} instances per size (40-node graph, instance seeds "
        f"1–{args.exact_instances}), QPSO + local search run with {args.seeds} seeds on each. "
        "Gap = (QPSO − optimum) / optimum.",
        "",
        "| Customers | QPSO runs | Reached optimum | Median gap | Worst gap | Optimum feasible |",
        "|---|---|---|---|---|---|",
    ]
    for n in args.exact_sizes:
        gaps, hit, total, feas = [], 0, 0, 0
        for inst in range(1, args.exact_instances + 1):
            opt = next(r for r in a_runs if r["config"] == "exact" and r["n"] == n and r["instance"] == inst)
            feas += opt["feasible"]
            for r in a_runs:
                if r["config"] == "qpso_ls" and r["n"] == n and r["instance"] == inst:
                    g = optimality_gap(r["fitness"], opt["fitness"])
                    gaps.append(g)
                    hit += g < 1e-6
                    total += 1
        md.append(f"| {n} | {total} | {hit}/{total} | {statistics.median(gaps):.2f}% "
                  f"| {max(gaps):.2f}% | {feas}/{args.exact_instances} |")

    header = ["| Customers | Configuration | Fitness mean ± std | Median | Best | Feasible | Median runtime (s) |",
              "|---|---|---|---|---|---|---|"]
    md += ["", "## B. Jump-cap ablation (local search off in both arms)", "",
           f"One instance per size (150-node graph, instance seed 1); the same {args.seeds} seeds "
           "in both arms, so each seed is a paired comparison.", ""]
    md += header + _summary_rows(bc_runs, jump_cfg, args.sizes)
    md += ["", "Paired comparison, per seed:", "",
           "| Customers | Cap better | Tie | Cap worse |", "|---|---|---|---|"]
    for n, w, t, tot in _paired_wins(bc_runs, "qpso", "qpso_nocap", args.sizes):
        md.append(f"| {n} | {w}/{tot} | {t}/{tot} | {tot - w - t}/{tot} |")
    md += ["", "Figure: `ablation_jump_cap.png`.", "",
           "## C. Local-search ablation", "",
           "Same instances and seeds as B.", ""]
    md += header + _summary_rows(bc_runs, ls_cfg, args.sizes)
    md += ["", "Paired comparison, QPSO + local search against standard PSO, per seed:", "",
           "| Customers | QPSO + LS better | Tie | QPSO + LS worse |", "|---|---|---|---|"]
    for n, w, t, tot in _paired_wins(bc_runs, "qpso_ls", "standard_pso", args.sizes):
        md.append(f"| {n} | {w}/{tot} | {t}/{tot} | {tot - w - t}/{tot} |")
    md += ["", "Figure: `ablation_local_search.png`. Runtimes were measured with runs executing "
           "in parallel on a shared machine, so compare them within a table, not with the "
           "single-process timings in `stress_test_*.json`.", ""]

    with open("data/ablation_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("  wrote data/ablation_results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
