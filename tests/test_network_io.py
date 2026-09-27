"""
test_network_io.py
------------------
A frozen network must rebuild the network it was saved from exactly: the same
nodes and edges in the same order (customer sampling and shortest-path ties
depend on it), and therefore the same VRP instance and the same solver result.
"""

from __future__ import annotations

import os

from app.core.classical_baselines_vrp import run_standard_pso_vrp
from app.core.graph_model import generate_synthetic_city_graph, load_network, save_network
from app.core.vrp_problem import generate_synthetic_vrp

DELHI = os.path.join(os.path.dirname(__file__), "..", "data", "networks", "delhi_osm.json")


def test_round_trip_rebuilds_the_same_network_and_instance(tmp_path):
    net = generate_synthetic_city_graph(n_nodes=60, seed=4)
    path = tmp_path / "net.json"
    save_network(net, str(path), {"note": "test"})
    back = load_network(str(path))

    assert list(back.graph.nodes(data=True)) == list(net.graph.nodes(data=True))
    assert list(back.graph.edges(data=True)) == list(net.graph.edges(data=True))

    a = generate_synthetic_vrp(net, n_customers=12, depot=0, vehicle_capacity=80, seed=2)
    b = generate_synthetic_vrp(back, n_customers=12, depot=0, vehicle_capacity=80, seed=2)
    assert [c.node_id for c in a.customers] == [c.node_id for c in b.customers]
    assert a.time_matrix == b.time_matrix and a.dist_matrix == b.dist_matrix
    assert (run_standard_pso_vrp(a, n_particles=10, max_iter=20, seed=1).best_fitness
            == run_standard_pso_vrp(b, n_particles=10, max_iter=20, seed=1).best_fitness)


def test_frozen_delhi_extract_loads():
    net = load_network(DELHI)
    assert net.graph.number_of_nodes() == 373
    assert net.graph.number_of_edges() == 884
    # Real one-way streets survive: some roads exist in one direction only.
    assert any(not net.graph.has_edge(v, u) for u, v in net.graph.edges())
