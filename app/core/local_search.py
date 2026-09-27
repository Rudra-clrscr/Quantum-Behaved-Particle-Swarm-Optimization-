"""
local_search.py
-----------------
2-opt and or-opt local search operators for CVRPTW routes. Used to build a
"memetic" (hybrid metaheuristic + local search) version of QPSO: the global
search finds promising regions of the solution space, and local search polishes
solutions within those regions -- standard practice for closing the gap between
population-based metaheuristics and true optimality on larger VRP instances.

    2-opt (intra-route):  reverse a segment of a single vehicle's route to
                           remove distance-increasing "crossings" in the path.
    or-opt (inter-route):  relocate a single customer to a different position
                           (possibly a different vehicle's route) if it reduces
                           total cost -- lets the search fix bad vehicle
                           assignments that 2-opt alone can't touch.
"""

from __future__ import annotations
import copy
import inspect
from typing import List, Optional, Tuple

import numpy as np

from app.core.vrp_problem import VRPProblem, evaluate_solution, VRPSolution


_EVAL_DEFAULTS = inspect.signature(evaluate_solution).parameters


class RouteCosts:
    """
    evaluate_solution, split by route, for ranking candidate moves.

    The objective is a sum over routes plus an idle-vehicle term that only counts
    empty routes. A 2-opt move changes one route and a relocation at most two, so
    only those need re-walking; the rest keep their cached parts. Each changed
    route is still walked in full and in order, with travel priced at the
    departure clock, so the downstream effects of a change on time-dependent
    travel, waiting and time windows are exactly those evaluate_solution sees.
    There is no edge-level shortcut.

    Summing per-route parts rounds differently from evaluate_solution's single
    running total, by about 1e-12, far below the 1e-9 acceptance margin. Only
    move ranking uses these values: every solution local search hands back is
    scored by evaluate_solution itself. tests/test_route_costs.py checks both
    the values and that the passes make the same moves as before.
    """

    def __init__(self, problem: VRPProblem):
        self.problem = problem
        self.lookup = {c.node_id: c for c in problem.customers}
        self.w_time = getattr(problem, "objective_w_time", 0.6)
        self.w_distance = getattr(problem, "objective_w_distance", 0.4)
        self.cap_w = _EVAL_DEFAULTS["capacity_penalty_weight"].default
        self.tw_w = _EVAL_DEFAULTS["time_window_penalty_weight"].default
        self.idle_w = _EVAL_DEFAULTS["idle_vehicle_penalty_weight"].default
        self.require_all = getattr(problem, "require_all_vehicles", False)

    def route(self, v_idx: int, route: List[int]) -> Tuple[float, float, float, float]:
        """(distance, time, capacity violation, time-window violation) of one route,
        computed step for step as in evaluate_solution."""
        if not route:
            return (0.0, 0.0, 0.0, 0.0)
        p, lookup = self.problem, self.lookup
        distance = time = capacity_violation = tw_violation = 0.0

        capacity = p.capacity_for(v_idx)
        demand = sum(lookup[n].demand for n in route)
        if demand > capacity:
            capacity_violation = demand - capacity
        speed, cost_per_km = p.speed_factor_for(v_idx), p.cost_per_km_for(v_idx)

        node, clock = p.depot, 0.0
        for node_id in route:
            travel_t = p.travel_time(node, node_id, depart_at=clock) * speed
            travel_d = p.travel_distance(node, node_id)
            if not np.isfinite(travel_t):
                tw_violation += 1000.0
                node = node_id
                continue
            arrival = clock + travel_t
            cust = lookup[node_id]
            if arrival < cust.ready_time:
                wait, start = cust.ready_time - arrival, cust.ready_time
            else:
                wait, start = 0.0, arrival
            if start > cust.due_time:
                tw_violation += start - cust.due_time
            distance += travel_d * cost_per_km
            time += travel_t + wait
            clock = start + cust.service_time
            node = node_id

        back_t = p.travel_time(node, p.depot, depart_at=clock) * speed
        back_d = p.travel_distance(node, p.depot)
        if np.isfinite(back_t):
            distance += back_d * cost_per_km
            time += back_t
        return (distance, time, capacity_violation, tw_violation)

    def fitness(self, parts: List[Tuple[float, float, float, float]], n_empty: int) -> float:
        distance = time = capacity_violation = tw_violation = 0.0
        for d, t, c, w in parts:
            distance += d
            time += t
            capacity_violation += c
            tw_violation += w
        idle = n_empty * self.idle_w if self.require_all else 0.0
        return (self.w_time * time + self.w_distance * distance
                + self.cap_w * capacity_violation + self.tw_w * tw_violation + idle)


def two_opt_pass(problem: VRPProblem, routes: List[List[int]]) -> List[List[int]]:
    """One first-improvement 2-opt pass over every route independently."""
    routes = copy.deepcopy(routes)
    costs = RouteCosts(problem)
    parts = [costs.route(v, r) for v, r in enumerate(routes)]
    n_empty = sum(1 for r in routes if not r)       # 2-opt never empties or fills a route
    base_fit = costs.fitness(parts, n_empty)

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
                    candidate_parts = costs.route(r_idx, candidate)
                    fit = costs.fitness(parts[:r_idx] + [candidate_parts] + parts[r_idx + 1:], n_empty)
                    if fit < base_fit - 1e-9:
                        route = candidate
                        parts[r_idx] = candidate_parts
                        base_fit = fit
                        improved = True
            routes[r_idx] = route

    return routes


def or_opt_pass(problem: VRPProblem, routes: List[List[int]]) -> List[List[int]]:
    """One first-improvement or-opt pass: try relocating each customer to every
    other position (same or different route) and keep the move if it helps.

    Important: we look up each customer's CURRENT position fresh on every
    iteration (rather than iterating a stale (route_index, position) snapshot
    taken before any moves) -- routes mutate as we go, so stale indices would
    silently duplicate or drop customers once an earlier move shifts list
    positions out from under a later one.
    """
    routes = copy.deepcopy(routes)
    costs = RouteCosts(problem)
    parts = [costs.route(v, r) for v, r in enumerate(routes)]
    base_fit = costs.fitness(parts, sum(1 for r in routes if not r))
    all_customer_ids = [c.node_id for c in problem.customers]

    for customer in all_customer_ids:
        src_idx, pos = None, None
        for r_idx, route in enumerate(routes):
            if customer in route:
                src_idx = r_idx
                pos = route.index(customer)
                break
        if src_idx is None:
            continue  # shouldn't happen, but guard against inconsistent state

        src_route = routes[src_idx]
        without = src_route[:pos] + src_route[pos + 1:]
        without_parts = costs.route(src_idx, without)
        n_empty = sum(1 for r in routes if not r)

        best_fit = base_fit
        best = None

        for dst_idx, dst_route in enumerate(routes):
            candidate_dst = dst_route if dst_idx != src_idx else without
            # Emptiness after the move: the destination is never empty; the
            # source is empty only if it was this customer alone.
            if dst_idx == src_idx:
                trial_empty = n_empty
            else:
                trial_empty = n_empty + (not without) - (not dst_route)
            for insert_at in range(len(candidate_dst) + 1):
                new_dst = candidate_dst[:insert_at] + [customer] + candidate_dst[insert_at:]

                trial_parts = list(parts)
                trial_parts[src_idx] = without_parts
                trial_parts[dst_idx] = costs.route(dst_idx, new_dst)

                fit = costs.fitness(trial_parts, trial_empty)
                if fit < best_fit - 1e-9:
                    best_fit = fit
                    best = (dst_idx, new_dst, trial_parts)

        if best is not None:
            dst_idx, new_dst, parts = best
            routes = [r[:] for r in routes]
            routes[src_idx] = without
            routes[dst_idx] = new_dst
            base_fit = best_fit

    return routes


def local_search_refine(problem: VRPProblem, routes: List[List[int]],
                         max_passes: int = 2) -> List[List[int]]:
    """Alternate 2-opt and or-opt passes until no improvement or max_passes reached."""
    current = routes
    for _ in range(max_passes):
        before_fit = evaluate_solution(problem, current).fitness
        current = two_opt_pass(problem, current)
        current = or_opt_pass(problem, current)
        after_fit = evaluate_solution(problem, current).fitness
        if after_fit >= before_fit - 1e-9:
            break
    return current


def routes_to_chromosome(problem: VRPProblem, routes: List[List[int]]):
    """Re-encode a routes structure back into a random-key chromosome, so a
    locally-refined solution can be reinjected into a continuous-encoding
    metaheuristic's population (Lamarckian learning)."""
    node_to_customer_idx = {c.node_id: i for i, c in enumerate(problem.customers)}
    chromosome = np.zeros(len(problem.customers))

    for v_idx, route in enumerate(routes):
        n = len(route)
        for k, node_id in enumerate(route):
            c_idx = node_to_customer_idx[node_id]
            # spread priorities evenly within [v_idx, v_idx+1)
            frac = (k + 0.5) / max(n, 1)
            chromosome[c_idx] = v_idx + frac

    return chromosome


# ---------------------------------------------------------------------------
# The memetic step, shared by every swarm optimiser
# ---------------------------------------------------------------------------
# QPSO and standard PSO both call these, so "QPSO + LS" and "PSO + LS" differ
# only in the base metaheuristic: same interval, same pass count, same
# strict-improvement acceptance, same Lamarckian reinjection, same final polish.
# A copy of this logic in each optimiser could drift, and the ablation that
# compares them would then be measuring the drift.

LOCAL_SEARCH_INTERVAL = 15
LOCAL_SEARCH_PASSES = 2


class StagnationStop:
    """
    Early stop for a memetic run: stop once `patience` consecutive local-search
    windows (the iterations since the previous call, plus the call itself) end
    without a new global best. Checked only at refinement iterations. With
    patience None it never stops, and the run goes to max_iter as before.
    """

    def __init__(self, patience: Optional[int]):
        self.patience = patience
        self.best = float("inf")
        self.idle = 0

    def should_stop(self, best_fit: float) -> bool:
        if best_fit < self.best:
            self.best = best_fit
            self.idle = 0
        else:
            self.idle += 1
        return self.patience is not None and self.idle >= self.patience


def is_refinement_iteration(it: int, interval: int) -> bool:
    """Refine on every `interval`-th iteration, never on the first."""
    return it > 0 and it % interval == 0


def refine_best(problem: VRPProblem, best_sol: VRPSolution, best_fit: float,
                passes: int) -> Optional[Tuple[VRPSolution, "np.ndarray"]]:
    """
    Run local search from the swarm's best solution.

    Returns (refined solution, its chromosome) only if it is strictly better than
    `best_fit`; otherwise None and the swarm is left untouched.
    """
    refined_routes = local_search_refine(problem, best_sol.routes, max_passes=passes)
    refined = evaluate_solution(problem, refined_routes)
    if refined.fitness < best_fit:
        chromosome = np.clip(routes_to_chromosome(problem, refined_routes),
                             0.0, problem.n_vehicles - 1e-9)
        return refined, chromosome
    return None


def reinject_into_worst(positions, pbest, pbest_fit, chromosome, fitness) -> int:
    """
    Lamarckian reinjection: overwrite the particle with the worst personal best
    with the refined solution, so the swarm builds on it. Returns its index.
    """
    worst = int(np.argmax(pbest_fit))
    positions[worst] = chromosome.copy()
    pbest[worst] = chromosome.copy()
    pbest_fit[worst] = fitness
    return worst


if __name__ == "__main__":
    from app.core.graph_model import generate_synthetic_city_graph
    from app.core.vrp_problem import generate_synthetic_vrp, evaluate_solution
    from app.core.classical_baselines_vrp import run_greedy_nn_vrp

    net = generate_synthetic_city_graph(n_nodes=40, seed=7)
    vrp = generate_synthetic_vrp(net, n_customers=18, depot=0, vehicle_capacity=80, seed=3)

    greedy = run_greedy_nn_vrp(vrp)
    print("Before local search:", greedy.best_fitness)

    refined_routes = local_search_refine(vrp, greedy.best_solution.routes, max_passes=3)
    refined_sol = evaluate_solution(vrp, refined_routes)
    print("After local search: ", refined_sol.fitness, "feasible:", refined_sol.feasible)
