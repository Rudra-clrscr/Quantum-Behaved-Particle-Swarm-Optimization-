"""
test_pso_local_search.py
------------------------
"PSO + local search" exists to answer one question: does QPSO benefit from
local search more than standard PSO does? That comparison is only valid if the
two arms run the *same* memetic step and differ only in the base swarm. These
tests hold that: same functions, same call schedule, same pass counts, and
standard PSO unchanged when local search is off.
"""

from __future__ import annotations

import pytest

import app.core.classical_baselines_vrp as baselines
import app.core.local_search as ls
import app.core.qpso_vrp as qpso_mod
from app.core.classical_baselines_vrp import run_standard_pso_vrp
from app.core.graph_model import generate_synthetic_city_graph
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.vrp_problem import generate_synthetic_vrp


@pytest.fixture(scope="module")
def problem():
    net = generate_synthetic_city_graph(n_nodes=60, seed=3)
    return generate_synthetic_vrp(net, n_customers=12, depot=0, vehicle_capacity=80, seed=3)


def _record_refinements(monkeypatch):
    """Record the max_passes of every local-search call made by the memetic step."""
    calls = []
    real = ls.local_search_refine

    def spy(problem, routes, max_passes=2):
        calls.append(max_passes)
        return real(problem, routes, max_passes=max_passes)

    monkeypatch.setattr(ls, "local_search_refine", spy)
    return calls


def test_both_optimisers_use_the_same_memetic_functions():
    for name in ("is_refinement_iteration", "refine_best", "reinject_into_worst"):
        assert getattr(qpso_mod, name) is getattr(ls, name)
        assert getattr(baselines, name) is getattr(ls, name)


def test_same_refinement_schedule_and_passes(problem, monkeypatch):
    calls = _record_refinements(monkeypatch)
    QPSOVRPOptimizer(problem, n_particles=10, max_iter=46, seed=1, use_local_search=True).optimize()
    qpso_calls = list(calls)
    calls.clear()
    run_standard_pso_vrp(problem, n_particles=10, max_iter=46, seed=1, use_local_search=True)

    # iterations 15, 30, 45 at 2 passes, then the final polish at 3
    assert qpso_calls == [2, 2, 2, 3]
    assert calls == qpso_calls


def test_defaults_match_between_arms():
    import inspect
    q = inspect.signature(QPSOVRPOptimizer.__init__).parameters
    p = inspect.signature(run_standard_pso_vrp).parameters
    for name in ("local_search_interval", "local_search_passes"):
        assert q[name].default == p[name].default


def test_pso_without_local_search_never_calls_it(problem, monkeypatch):
    calls = _record_refinements(monkeypatch)
    r = run_standard_pso_vrp(problem, n_particles=10, max_iter=40, seed=2)
    assert calls == []
    assert r.algorithm == "Standard PSO"


def test_pso_with_local_search_returns_a_complete_solution(problem):
    r = run_standard_pso_vrp(problem, n_particles=10, max_iter=40, seed=2, use_local_search=True)
    served = sorted(n for route in r.best_solution.routes for n in route)
    assert served == sorted(c.node_id for c in problem.customers)
    assert r.best_fitness == r.best_solution.fitness
    assert r.convergence_curve[-1] == r.best_fitness
    assert r.algorithm == "Standard PSO + local search"


def test_refinement_is_never_accepted_unless_strictly_better(problem):
    base = run_standard_pso_vrp(problem, n_particles=10, max_iter=5, seed=4).best_solution
    assert ls.refine_best(problem, base, base.fitness - 1e9, passes=2) is None
