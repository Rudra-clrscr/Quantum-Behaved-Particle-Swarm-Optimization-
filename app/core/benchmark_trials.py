"""
benchmark_trials.py
--------------------
Repeated benchmark runs: every (algorithm, seed) pair, as raw per-trial samples.

A metaheuristic's answer moves with its seed, so one run per algorithm shows
which algorithm won that run -- a weaker claim than which algorithm is better.
Repeating the run is what turns a single figure into a spread, and a spread is
what a box plot and a success rate are drawn from
(see scripts/plot_benchmark_charts.py).

Trials are independent, so they go to a process pool. Threads would not do: the
work is a pure-Python route walk (evaluate_solution accounts for ~85% of a solve
by profile) and so is held by the GIL throughout. Measured upstream on a 12-core machine,
12 trials went from 22.67s to 4.24s, a 5.3x speedup, with results bit-identical
to the sequential run.

A GPU was considered and rejected on measurement rather than taste. The route
clock is sequential, so a batched version must launch one kernel per stop with
very little work in each; at 40 particles CUDA came out 22x SLOWER than numpy,
and did not overtake it until roughly 20,000 particles.

Ported from the trial runner behind /api/benchmark/run in the upstream
MargdarshaQ application, without the web layer.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.core.classical_baselines_vrp import (
    run_ga_vrp, run_greedy_nn_vrp, run_sa_vrp, run_standard_pso_vrp,
)
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.vrp_problem import VRPProblem

logger = logging.getLogger(__name__)

ALL_ALGORITHMS = ["greedy", "qpso", "ga", "sa", "standard_pso"]


@dataclass
class Trial:
    """One (algorithm, seed) run. Carries the seed so samples stay in seed
    order no matter how the pool scheduled the work."""
    algorithm: str
    name: str
    seed: int
    solution: Any
    curve: List[float]
    n_evaluations: int
    runtime_ms: float
    fitness: float
    feasible: bool


def solve_one(problem: VRPProblem, algorithm: str, seed: int, max_iter: int,
              n_particles: int = 50, use_local_search: bool = True):
    """Run one algorithm once. Returns (solution, curve, n_evaluations, name, runtime_ms)."""
    t0 = time.perf_counter()

    if algorithm == "qpso":
        res = QPSOVRPOptimizer(problem, n_particles=n_particles, max_iter=max_iter,
                               seed=seed, use_local_search=use_local_search).optimize()
        sol, curve, n_eval, name = (res.best_solution, res.convergence_curve,
                                    res.n_evaluations,
                                    "QPSO + local search" if use_local_search else "QPSO")
    elif algorithm == "ga":
        r = run_ga_vrp(problem, pop_size=n_particles, max_iter=max_iter, seed=seed)
        sol, curve, n_eval, name = r.best_solution, r.convergence_curve, r.n_evaluations, r.algorithm
    elif algorithm == "sa":
        r = run_sa_vrp(problem, max_iter=max_iter * 20, seed=seed)
        sol, curve, n_eval, name = r.best_solution, r.convergence_curve, r.n_evaluations, r.algorithm
    elif algorithm == "standard_pso":
        r = run_standard_pso_vrp(problem, n_particles=n_particles, max_iter=max_iter, seed=seed)
        sol, curve, n_eval, name = r.best_solution, r.convergence_curve, r.n_evaluations, r.algorithm
    elif algorithm == "greedy":
        r = run_greedy_nn_vrp(problem)
        sol, curve, n_eval, name = r.best_solution, r.convergence_curve, r.n_evaluations, r.algorithm
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'")

    return sol, curve, n_eval, name, (time.perf_counter() - t0) * 1000


def run_single_trial(problem: VRPProblem, algo: str, seed: int, max_iter: int) -> Optional[Trial]:
    sol, curve, n_eval, name, runtime_ms = solve_one(problem, algo, seed, max_iter)
    if sol is None:
        return None
    return Trial(
        algorithm=algo, name=name, seed=seed, solution=sol, curve=curve,
        n_evaluations=n_eval, runtime_ms=runtime_ms,
        fitness=sol.fitness, feasible=sol.feasible,
    )


_TRIAL_PROBLEM = None


def _init_trial_worker(problem):
    """
    Hand each worker the problem once, at process start, instead of once per
    task. The problem carries its precomputed distance, time and path matrices,
    which is the large part of the pickle -- sending it per trial would spend
    more on serialisation than the trial itself costs.
    """
    global _TRIAL_PROBLEM
    _TRIAL_PROBLEM = problem


def _run_trial_task(task):
    algo, seed, max_iter = task
    return run_single_trial(_TRIAL_PROBLEM, algo, seed, max_iter)


def _run_sequential(problem, tasks, algorithms):
    out = {algo: [] for algo in algorithms}
    for algo, seed, mi in tasks:
        t = run_single_trial(problem, algo, seed, mi)
        if t is not None:
            out[algo].append(t)
    return out


def run_trials(problem: VRPProblem, algorithms: Sequence[str], seeds: Sequence[int],
               max_iter: int) -> Dict[str, List[Trial]]:
    """
    Every (algorithm, seed) pair, grouped by algorithm, in seed order.

    With a single seed this is a plain sequential loop -- no pool is created, so
    the single-run path cannot be destabilised by anything in the parallel
    branch. The pool is only reached when more than one trial was asked for.
    """
    tasks = [(algo, seed, max_iter) for algo in algorithms for seed in seeds]

    if len(seeds) == 1:
        return _run_sequential(problem, tasks, algorithms)

    out = {algo: [] for algo in algorithms}
    workers = max(1, min(os.cpu_count() or 1, len(tasks)))
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_trial_worker,
                                 initargs=(problem,)) as pool:
            for t in pool.map(_run_trial_task, tasks):
                if t is not None:
                    out[t.algorithm].append(t)
    except Exception as exc:
        # A pool can fail for reasons unrelated to the benchmark: a sandbox that
        # forbids subprocesses, a container with no /dev/shm. None of those are a
        # reason to fail a run the machine can compute, just more slowly.
        logger.warning("Parallel trials unavailable (%s); falling back to sequential. "
                       "Results are identical, the run is just slower.", exc)
        out = _run_sequential(problem, tasks, algorithms)

    order = {seed: i for i, seed in enumerate(seeds)}
    for algo in out:
        out[algo].sort(key=lambda t: order[t.seed])
    return out


def summarize_trials(per_algo: Dict[str, List[Trial]],
                     success_threshold_pct: float = 5.0) -> Dict[str, Dict[str, Any]]:
    """
    Per-algorithm samples and rates.

    "Success" needs something to be successful against. The true optimum is
    unknown for these instances -- that is why a metaheuristic is being used --
    so the best fitness any algorithm achieved in any trial of this run stands
    in for it. The success rate is therefore a statement about agreement within
    the run, not about distance from a proven optimum.
    """
    all_fit = [t.fitness for trials in per_algo.values() for t in trials]
    best_overall = min(all_fit) if all_fit else 0.0
    cutoff = best_overall * (1.0 + success_threshold_pct / 100.0) if best_overall > 0 else 0.0

    summary: Dict[str, Dict[str, Any]] = {}
    for algo, trials in per_algo.items():
        if not trials:
            continue
        fits = [t.fitness for t in trials]
        summary[algo] = {
            "name": trials[0].name,
            "trials": len(trials),
            "fitness_samples": fits,
            "runtime_samples_ms": [t.runtime_ms for t in trials],
            "time_samples": [t.solution.total_time for t in trials],
            "distance_samples": [t.solution.total_distance for t in trials],
            "feasible_rate": sum(1 for t in trials if t.feasible) / len(trials),
            "success_rate": (sum(1 for f in fits if f <= cutoff) / len(fits)) if cutoff > 0 else 1.0,
        }
    return summary
