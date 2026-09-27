"""
scripts/generate_stress_test_cache.py
--------------------------------------
Measure how QPSO (+ local search) and standard PSO actually scale with customer
count, and record the result as JSON for scripts/plot_benchmark_charts.py.

Every point is timed, not extrapolated. Defaults are chosen so the result is
reproducible by whoever asks:

  - a synthetic 300-node city graph, so the sweep needs no internet and gives
    the same graph on every machine;
  - sizes small enough to finish in minutes;
  - a fixed seed range, so re-running reproduces the figures rather than
    producing new ones that quietly disagree with the committed file.

The settings used are written into the output next to the measurements, so a
chart drawn from it can state exactly what was run.

    python scripts/generate_stress_test_cache.py
    python scripts/generate_stress_test_cache.py --sizes 20 40 60 --seeds 5

The New Delhi measurement runs on the OpenStreetMap extract frozen in
data/networks/delhi_osm.json (scripts/freeze_osm_network.py), with the same
sizes, seeds and budget as the original upstream measurement, which is kept as
data/stress_test_delhi_2026-09-16.json:

    python scripts/generate_stress_test_cache.py --network data/networks/delhi_osm.json \
        --sizes 100 200 --budget 300 --out data/stress_test_delhi.json \
        --algorithms qpso_local_search qpso_local_search_reference standard_pso

"qpso_local_search_reference" is QPSO + LS with the previous local-search
implementation (app/core/local_search_reference.py): the same moves, so the
same result, with the old runtime.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.benchmark_vrp import run_stress_test_at_scale

OUT_PATH = os.path.join("data", "stress_test_synthetic.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 80, 100, 150, 200],
                        help="Customer counts to measure (default: 20 40 60 80 100 150 200)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Runs per (size, algorithm); the best is kept (default: 3)")
    parser.add_argument("--budget", type=float, default=90.0,
                        help="Seconds before a single run is called a timeout (default: 90)")
    parser.add_argument("--out", default=OUT_PATH, help=f"Output path (default: {OUT_PATH})")
    parser.add_argument("--network", default="synthetic",
                        help="'synthetic', or the path of a frozen network JSON (default: synthetic)")
    parser.add_argument("--algorithms", nargs="+", default=["qpso_local_search", "standard_pso"],
                        help="Any of qpso_local_search, qpso_local_search_reference, standard_pso")
    args = parser.parse_args()

    algorithms = args.algorithms

    print(f"Measuring scalability: sizes={args.sizes}, seeds={args.seeds}, "
          f"network={args.network}, budget={args.budget}s")
    print(f"{len(args.sizes) * len(algorithms) * args.seeds} runs in total.\n")

    results = run_stress_test_at_scale(
        n_customers_list=args.sizes,
        algorithms=algorithms,
        network_source=args.network,
        time_budget_seconds=args.budget,
        n_seeds=args.seeds,
    )

    results["measurement"] = {
        "sizes": args.sizes,
        "algorithms": algorithms,
        "seeds_per_point": args.seeds,
        "time_budget_seconds": args.budget,
        "network_source": args.network,
        "note": "Runtime and fitness are measured, not extrapolated. Each point "
                "is the best of `seeds_per_point` runs at that size.",
    }
    if args.network != "synthetic":
        with open(args.network) as f:
            results["measurement"]["network_metadata"] = json.load(f).get("metadata", {})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWritten to {args.out}")
    for size in args.sizes:
        row = results["results"].get(str(size), {})
        cells = "  ".join(
            f"{algo.split('_')[0]}={(row.get(algo) or {}).get('runtime_ms') or float('nan'):.0f}ms"
            for algo in algorithms
        )
        print(f"  {size:>4} customers   {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
