"""
test_route_costs.py
-------------------
Local search ranks candidate moves with RouteCosts, which re-walks only the
routes a move changes instead of calling evaluate_solution on the whole
solution. That is only acceptable if it changes nothing but speed, so:

  1. its fitness matches evaluate_solution on every objective feature --
     time windows, capacity, time-dependent travel, a mixed fleet, forced use
     of every vehicle -- to far inside the 1e-9 acceptance margin; and
  2. the 2-opt and or-opt passes make exactly the moves the previous
     implementation made. The previous implementation is kept below, verbatim,
     as the oracle.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from app.core.graph_model import generate_synthetic_city_graph
from app.core.local_search import (
    RouteCosts, local_search_refine, or_opt_pass, two_opt_pass,
)
from app.core.vrp_problem import decode_chromosome, evaluate_solution, generate_synthetic_vrp


# ---------------------------------------------------------------------------
# The previous passes, verbatim: every candidate scored by evaluate_solution
# ---------------------------------------------------------------------------
def _reference_two_opt_pass(problem, routes):
    routes = copy.deepcopy(routes)
    base_fit = evaluate_solution(problem, routes).fitness
    for r_idx, route in enumerate(routes):
        n = len(route)
        if n < 3:
            continue
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    trial_routes = routes[:r_idx] + [candidate] + routes[r_idx + 1:]
                    fit = evaluate_solution(problem, trial_routes).fitness
                    if fit < base_fit - 1e-9:
                        route = candidate
                        base_fit = fit
                        improved = True
            routes[r_idx] = route
    return routes


def _reference_or_opt_pass(problem, routes):
    routes = copy.deepcopy(routes)
    base_fit = evaluate_solution(problem, routes).fitness
    for customer in [c.node_id for c in problem.customers]:
        src_idx, pos = None, None
        for r_idx, route in enumerate(routes):
            if customer in route:
                src_idx, pos = r_idx, route.index(customer)
                break
        if src_idx is None:
            continue
        src_route = routes[src_idx]
        without = src_route[:pos] + src_route[pos + 1:]
        best_fit, best_routes = base_fit, None
        for dst_idx, dst_route in enumerate(routes):
            candidate_dst = dst_route if dst_idx != src_idx else without
            for insert_at in range(len(candidate_dst) + 1):
                new_dst = candidate_dst[:insert_at] + [customer] + candidate_dst[insert_at:]
                trial = [r[:] for r in routes]
                trial[src_idx] = without
                trial[dst_idx] = new_dst
                if src_idx == dst_idx:
                    trial[src_idx] = new_dst
                fit = evaluate_solution(problem, trial).fitness
                if fit < best_fit - 1e-9:
                    best_fit, best_routes = fit, trial
        if best_routes is not None:
            routes, base_fit = best_routes, best_fit
    return routes


# ---------------------------------------------------------------------------
# Instances covering every term of the objective
# ---------------------------------------------------------------------------
def _problem(kind: str, n: int = 14, seed: int = 2):
    net = generate_synthetic_city_graph(n_nodes=60, seed=seed)
    p = generate_synthetic_vrp(
        net, n_customers=n, depot=0, vehicle_capacity=60, seed=seed,
        time_dependent=(kind in ("time_dependent", "everything")),
        require_all_vehicles=(kind in ("all_vehicles", "everything")),
    )
    if kind in ("mixed_fleet", "everything"):
        k = p.n_vehicles
        p.vehicle_capacities = [40.0 + 15.0 * (v % 3) for v in range(k)]
        p.vehicle_speed_factors = [0.8 + 0.2 * (v % 3) for v in range(k)]
        p.vehicle_cost_per_km = [1.0 + 0.5 * (v % 2) for v in range(k)]
    return p


KINDS = ["plain", "time_dependent", "mixed_fleet", "all_vehicles", "everything"]


def _random_routes(problem, rng):
    """A random, usually poor, starting solution -- plenty of moves to make."""
    return decode_chromosome(problem, rng.uniform(0, problem.n_vehicles, len(problem.customers)))


@pytest.mark.parametrize("kind", KINDS)
def test_fitness_matches_evaluate_solution(kind):
    problem = _problem(kind)
    costs = RouteCosts(problem)
    rng = np.random.default_rng(0)
    for _ in range(200):
        routes = _random_routes(problem, rng)
        if rng.uniform() < 0.3 and len(routes) > 1:        # force some empty routes
            routes[int(rng.integers(len(routes)))] = []
        parts = [costs.route(v, r) for v, r in enumerate(routes)]
        fast = costs.fitness(parts, sum(1 for r in routes if not r))
        ref = evaluate_solution(problem, routes).fitness
        assert abs(fast - ref) <= 1e-10 * max(1.0, abs(ref))


@pytest.mark.parametrize("kind", KINDS)
def test_passes_make_exactly_the_reference_moves(kind):
    problem = _problem(kind)
    rng = np.random.default_rng(1)
    for _ in range(6):
        start = _random_routes(problem, rng)
        assert two_opt_pass(problem, start) == _reference_two_opt_pass(problem, start)
        assert or_opt_pass(problem, start) == _reference_or_opt_pass(problem, start)


def test_a_customer_moved_out_of_a_single_stop_route_empties_it():
    """The emptiness bookkeeping matters when every vehicle must be used."""
    problem = _problem("all_vehicles", n=8)
    rng = np.random.default_rng(3)
    for _ in range(10):
        start = _random_routes(problem, rng)
        if any(len(r) == 1 for r in start):
            assert or_opt_pass(problem, start) == _reference_or_opt_pass(problem, start)


def test_refine_keeps_every_customer_exactly_once():
    problem = _problem("everything", n=20)
    routes = local_search_refine(problem, _random_routes(problem, np.random.default_rng(4)), max_passes=3)
    served = sorted(n for r in routes for n in r)
    assert served == sorted(c.node_id for c in problem.customers)
