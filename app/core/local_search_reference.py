"""
local_search_reference.py
-------------------------
The 2-opt and or-opt passes as they were before RouteCosts: every candidate
move scored by a full evaluate_solution call. Kept verbatim for two uses only,
never on the optimisation path:

  - the oracle in tests/test_route_costs.py, which requires the current passes
    to make exactly the same moves; and
  - the before/after timing in the scalability measurements
    (algorithm "qpso_local_search_reference" in app/core/benchmark_vrp.py).
"""

from __future__ import annotations

import copy

from app.core.vrp_problem import evaluate_solution


def reference_two_opt_pass(problem, routes):
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


def reference_or_opt_pass(problem, routes):
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
