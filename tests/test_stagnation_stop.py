"""
test_stagnation_stop.py
-----------------------
Early stopping for the memetic optimisers: stop once `patience` consecutive
local-search windows bring no new global best. It must be inert when off, stop
only at a refinement iteration, and behave the same way in QPSO and PSO.
"""

from __future__ import annotations

import pytest

from app.core.classical_baselines_vrp import run_standard_pso_vrp
from app.core.graph_model import generate_synthetic_city_graph
from app.core.local_search import StagnationStop
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.vrp_problem import generate_synthetic_vrp


@pytest.fixture(scope="module")
def problem():
    net = generate_synthetic_city_graph(n_nodes=60, seed=5)
    return generate_synthetic_vrp(net, n_customers=15, depot=0, vehicle_capacity=80, seed=5)


def _qpso(problem, **kw):
    r = QPSOVRPOptimizer(problem, n_particles=10, max_iter=150, seed=3, **kw).optimize()
    return r.best_fitness, r.convergence_curve


def _pso(problem, **kw):
    r = run_standard_pso_vrp(problem, n_particles=10, max_iter=150, seed=3,
                             use_local_search=True, **kw)
    return r.best_fitness, r.convergence_curve


def test_counts_consecutive_windows_without_a_new_best():
    s = StagnationStop(2)
    assert [s.should_stop(f) for f in (10.0, 9.0, 9.0, 8.0, 8.0, 8.0)] == \
        [False, False, False, False, False, True]


def test_never_stops_when_off():
    s = StagnationStop(None)
    assert not any(s.should_stop(5.0) for _ in range(100))


@pytest.mark.parametrize("run", [_qpso, _pso])
def test_an_unreachable_patience_changes_nothing(problem, run):
    assert run(problem, stagnation_patience=10 ** 6) == run(problem)


@pytest.mark.parametrize("run", [_qpso, _pso])
def test_stops_early_and_only_after_a_refinement(problem, run):
    fitness, curve = run(problem, stagnation_patience=1)
    iterations = len(curve)
    assert iterations < 150
    last = iterations - 1
    assert last > 0 and last % 15 == 0
    assert curve[-1] == fitness


def test_stopping_only_truncates_the_pso_run(problem):
    """
    Up to the stop, a PSO run with early stopping is the full run: same
    trajectory, same local-search calls. Only the final entry differs, because
    the final polish is applied at the stop instead of at max_iter. (QPSO
    anneals alpha over max_iter, so an early-stopped QPSO run is not a prefix
    of any fixed-budget run and is not checked here.)
    """
    _, early = _pso(problem, stagnation_patience=1)
    _, full = _pso(problem)
    assert early[:-1] == full[:len(early) - 1]
