"""
scripts/freeze_osm_network.py
------------------------------
Download a real road network from OpenStreetMap and freeze it to JSON, so
measurements made on it can be repeated without OSM, osmnx or a network
connection -- and without OSM's data changing underneath them.

The loader below is upstream MargdarshaQ's `load_osm_network`
(app/core/osm_network.py), copied here with its behaviour unchanged, because
the New Delhi scalability measurement (data/stress_test_delhi.json) was made
with it: `load_osm_network(place="New Delhi, India", max_nodes=500)`. That is
this script's default.

Needs `osmnx` (1.x) and an internet connection; nothing else in the
repository does. Everything downstream reads the frozen file through
app.core.graph_model.load_network.

    python scripts/freeze_osm_network.py
    python scripts/freeze_osm_network.py --place "Connaught Place, New Delhi, India" --out data/networks/cp.json
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.graph_model import TrafficNetwork, load_network, save_network  # noqa: E402


def load_osm_network(place: str, network_type: str = "drive", simplify: bool = True,
                     max_nodes: int = 500, congestion_seed: int = 1,
                     congestion_low: float = 1.0, congestion_high: float = 2.5):
    """
    Upstream MargdarshaQ's loader, unchanged in behaviour (place-name path only).
    Returns (network, speed imputation error or None). When osmnx cannot impute
    speed limits, every road falls back to 40 km/h, as upstream's did.
    """
    import networkx as nx
    import osmnx as ox

    G = ox.graph_from_place(place, network_type=network_type, simplify=simplify)
    speed_error = None
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception as e:
        speed_error = str(e)
        print(f"Warning: ox.add_edge_speeds failed ({e}), falling back to default speeds.")

    # Keep the largest strongly connected component, so every customer is reachable.
    G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()

    # Crop to max_nodes by BFS from the first node, then re-take the largest SCC.
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
        start_node = list(G.nodes())[0]
        keep_nodes = {start_node}
        for u, v in nx.bfs_edges(G, start_node):
            keep_nodes.add(v)
            if len(keep_nodes) >= max_nodes:
                break
        G = G.subgraph(list(keep_nodes)).copy()
        G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()

    net = TrafficNetwork()
    for node_id, data in G.nodes(data=True):
        net.add_node(node_id, x=data["x"], y=data["y"])      # x = longitude, y = latitude
    for u, v, data in G.edges(data=True):
        speed_kph = data.get("speed_kph", 40.0)
        if isinstance(speed_kph, list):
            speed_kph = speed_kph[0]
        # OSM is directed: a two-way street appears as both (u, v) and (v, u).
        net.add_edge(u, v, distance=data.get("length", 100.0) / 1000.0,
                     base_speed_kmph=speed_kph, bidirectional=False)
    net.randomize_congestion(seed=congestion_seed, low=congestion_low, high=congestion_high)
    return net, speed_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--place", default="New Delhi, India")
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--out", default=os.path.join("data", "networks", "delhi_osm.json"))
    args = parser.parse_args()

    import networkx as nx
    import osmnx as ox

    print(f"Downloading '{args.place}' from OpenStreetMap (osmnx {ox.__version__})...")
    net, speed_error = load_osm_network(args.place, max_nodes=args.max_nodes)
    metadata = {
        "place": args.place,
        "speed_limits": ("imputed by osmnx from OSM maxspeed tags" if speed_error is None else
                         f"not imputed (osmnx add_edge_speeds failed: {speed_error}); "
                         "every road uses the 40 km/h default"),
        "network_type": "drive",
        "max_nodes": args.max_nodes,
        "congestion": {"seed": 1, "low": 1.0, "high": 2.5},
        "downloaded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "osmnx_version": ox.__version__,
        "networkx_version": nx.__version__,
        "nodes": net.graph.number_of_nodes(),
        "edges": net.graph.number_of_edges(),
        "source": "OpenStreetMap contributors, ODbL (https://www.openstreetmap.org/copyright)",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_network(net, args.out, metadata)

    # The frozen copy must rebuild the same network: same nodes and edges, in order.
    back = load_network(args.out)
    assert list(back.graph.nodes(data=True)) == list(net.graph.nodes(data=True))
    assert list(back.graph.edges(data=True)) == list(net.graph.edges(data=True))
    print(f"Froze {metadata['nodes']} nodes, {metadata['edges']} edges to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
