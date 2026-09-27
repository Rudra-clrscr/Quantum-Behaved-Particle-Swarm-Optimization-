"""
scripts/run_budget_ablation.py
-------------------------------
How much of the 150-iteration budget does a memetic run need?

data/swarm_contribution.md shows the swarm almost never improves the global
best after the first local-search call (iteration 15). If so, most of the
budget buys nothing. This runs QPSO + LS and PSO + LS on the same instances and
seeds as scripts/run_ablations.py with

    fixed budgets    max_iter = 30, 45, 60, 150  (1, 2, 3 and 9 local-search
                     calls before the final polish)
    early stopping   max_iter = 150, stop after 2 consecutive local-search
                     windows without a new global best

and reports each against the 150-iteration run on the same seed: fitness,
feasibility, iterations run, every objective evaluation (the swarm's and local
search's), and wall-clock time.

Writes data/budget_results.md, data/budget_results.json and
data/budget_quality.png.

    python scripts/run_budget_ablation.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_ablations import N_PARTICLES, SLOT, MUTED, SURFACE, _instance, _style  # noqa: E402

FULL = 150
PATIENCE = 2
# (label, max_iter, stagnation_patience)
BUDGETS = [("30 it.", 30, None), ("45 it.", 45, None), ("60 it.", 60, None),
           ("150 it.", FULL, None), (f"early stop ({PATIENCE})", FULL, PATIENCE)]
SWARMS = {"qpso_ls": "QPSO + LS", "pso_ls": "PSO + LS"}


def _run(task):
    swarm, budget, n, seed = task
    _, max_iter, patience = next(b for b in BUDGETS if b[0] == budget)

    import app.core.classical_baselines_vrp as baselines
    import app.core.local_search as ls
    import app.core.qpso_vrp as qpso_mod
    import app.core.vrp_problem as vp

    # Count every objective evaluation, the swarm's and local search's alike.
    # Both modules hold their own reference to evaluate_solution.
    counter = {"evals": 0, "ls_calls": 0}
    real_eval, real_refine = vp.evaluate_solution, ls.refine_best

    def counted_eval(*a, **k):
        counter["evals"] += 1
        return real_eval(*a, **k)

    def counted_refine(*a, **k):
        counter["ls_calls"] += 1
        return real_refine(*a, **k)

    vp.evaluate_solution = counted_eval
    ls.evaluate_solution = counted_eval
    qpso_mod.refine_best = counted_refine
    baselines.refine_best = counted_refine

    problem = _instance(n, 1, 150)
    t0 = time.perf_counter()
    if swarm == "qpso_ls":
        r = qpso_mod.QPSOVRPOptimizer(problem, n_particles=N_PARTICLES, max_iter=max_iter, seed=seed,
                                      use_local_search=True, stagnation_patience=patience).optimize()
    else:
        r = baselines.run_standard_pso_vrp(problem, n_particles=N_PARTICLES, max_iter=max_iter,
                                           seed=seed, use_local_search=True,
                                           stagnation_patience=patience)
    runtime = time.perf_counter() - t0
    return {"swarm": swarm, "budget": budget, "n": n, "seed": seed,
            "fitness": r.best_solution.fitness, "feasible": bool(r.best_solution.feasible),
            "iterations": len(r.convergence_curve), "evaluations": counter["evals"],
            "ls_calls": counter["ls_calls"], "runtime_s": runtime}


def _by(runs, swarm, budget, n):
    return {r["seed"]: r for r in runs if r["swarm"] == swarm and r["budget"] == budget and r["n"] == n}


def _pct(a, b):
    return 100.0 * (a - b) / b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Parallel runs (default: half the logical CPUs, to limit contention)")
    args = parser.parse_args()

    _style()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tasks = [(s, b[0], n, seed) for n in args.sizes for s in SWARMS for b in BUDGETS
             for seed in range(1, args.seeds + 1)]
    # Longest first, so the pool does not end on a tail of slow runs.
    tasks.sort(key=lambda t: (-t[2], t[1] != "150 it."))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(_run, tasks))

    with open("data/budget_results.json", "w") as f:
        json.dump({"generated_at": started, "settings": vars(args), "runs": runs}, f, indent=2)
    print("  wrote data/budget_results.json")

    labels = [b[0] for b in BUDGETS]
    md = [
        "# Iteration budget for the memetic optimisers",
        "",
        f"_Generated by `scripts/run_budget_ablation.py` on {started[:10]} with `--seeds {args.seeds} "
        f"--sizes {' '.join(map(str, args.sizes))} --workers {args.workers}`. "
        "Every run is in `data/budget_results.json`._",
        "",
        "Same instances and seeds as `data/ablation_results.md`; the 150-iteration arm repeats "
        "its QPSO + LS and PSO + LS runs. Local search is called every 15 iterations, so "
        "budgets of 30, 45, 60 and 150 iterations make 1, 2, 3 and 9 calls before the final "
        f"polish. Early stopping runs up to 150 iterations and stops after {PATIENCE} "
        "consecutive local-search windows without a new global best.",
        "",
        "Δ is each run's fitness against the 150-iteration run on the same seed, in percent; "
        "positive is worse. Evaluations count every call of the objective, by the swarm and by "
        f"local search. Runs share {args.workers} worker processes, so runtimes compare within "
        "this file only.",
        "",
    ]
    for swarm, name in SWARMS.items():
        md += [f"## {name}", "",
               "| Customers | Budget | Feasible | Median Δ | Worst Δ | Median iterations "
               "| Median evaluations | Median runtime (s) | Runtime vs 150 it. |",
               "|---|---|---|---|---|---|---|---|---|"]
        for n in args.sizes:
            full = _by(runs, swarm, "150 it.", n)
            full_rt = statistics.median(r["runtime_s"] for r in full.values())
            for label in labels:
                rs = _by(runs, swarm, label, n)
                d = [_pct(rs[s]["fitness"], full[s]["fitness"]) for s in rs]
                rt = statistics.median(r["runtime_s"] for r in rs.values())
                md.append(
                    f"| {n} | {label} | {sum(r['feasible'] for r in rs.values())}/{len(rs)} "
                    f"| {statistics.median(d):+.2f}% | {max(d):+.2f}% "
                    f"| {statistics.median(r['iterations'] for r in rs.values()):.0f} "
                    f"| {statistics.median(r['evaluations'] for r in rs.values()):,.0f} "
                    f"| {rt:.1f} | {100 * rt / full_rt:.0f}% |")
        md.append("")

    md += ["## Summary over all sizes", "",
           "Pooled over every size and seed: the median and worst Δ, how many runs are within "
           "1% of the 150-iteration result, and total runtime as a share of the 150-iteration "
           "arm's.", "",
           "| Algorithm | Budget | Median Δ | Worst Δ | Within 1% | Feasible | Total runtime vs 150 it. |",
           "|---|---|---|---|---|---|---|"]
    for swarm, name in SWARMS.items():
        full_total = sum(r["runtime_s"] for r in runs if r["swarm"] == swarm and r["budget"] == "150 it.")
        for label in labels:
            d, feas, tot, rt = [], 0, 0, 0.0
            for n in args.sizes:
                full, rs = _by(runs, swarm, "150 it.", n), _by(runs, swarm, label, n)
                d += [_pct(rs[s]["fitness"], full[s]["fitness"]) for s in rs]
                feas += sum(r["feasible"] for r in rs.values())
                tot += len(rs)
                rt += sum(r["runtime_s"] for r in rs.values())
            md.append(f"| {name} | {label} | {statistics.median(d):+.2f}% | {max(d):+.2f}% "
                      f"| {sum(1 for x in d if x <= 1.0)}/{len(d)} | {feas}/{tot} "
                      f"| {100 * rt / full_total:.0f}% |")
    md += ["", "Figure: `budget_quality.png`.", ""]

    with open("data/budget_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("  wrote data/budget_results.md")

    # Figure: median Δ against the 150-iteration run, by size, one panel per swarm.
    colours = {"30 it.": SLOT[1], "45 it.": SLOT[2], "60 it.": SLOT[0],
               f"early stop ({PATIENCE})": MUTED}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, (swarm, name) in zip(axes, SWARMS.items()):
        for label, colour in colours.items():
            med = []
            for n in args.sizes:
                full, rs = _by(runs, swarm, "150 it.", n), _by(runs, swarm, label, n)
                med.append(statistics.median(_pct(rs[s]["fitness"], full[s]["fitness"]) for s in rs))
            ax.plot(args.sizes, med, color=colour, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, label=label)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(name, loc="left")
        ax.set_xticks(args.sizes)
        ax.set_xlabel("Customers")
    axes[0].set_ylabel("Median % above the 150-iteration run")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower left", bbox_to_anchor=(0.01, 1.0), ncol=len(lbls))
    fig.tight_layout()
    fig.savefig("data/budget_quality.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote data/budget_quality.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
