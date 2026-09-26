# Quantum-Behaved Particle Swarm Optimization for Congestion-Aware Vehicle Routing

A quantum-behaved particle swarm optimization (QPSO) algorithm for the
Capacitated Vehicle Routing Problem with Time Windows (CVRPTW), with two
modifications over vanilla QPSO, plus an optional time-dependent congestion
extension.

This is a slim, algorithm-focused research artifact — code, benchmarks, and
results meant to be run and cited alongside a paper, not a live demo app. It
is split off from a larger hackathon project (see [Origin & attribution](#origin--attribution)).

**Paper:** "Quantum-Behaved Particle Swarm Optimization with Local Search
Refinement for Congestion-Aware Vehicle Routing", submitted to
[ANTIC 2026](https://antic-conf.org/) (6th International Conference on
Advanced Network Technologies and Intelligent Computing), IIIT Lucknow.

## What's here

Two modifications on top of standard QPSO for CVRPTW:

1. **Jump-cap fix** — QPSO's stochastic position update can produce
   dimension-dependent jump instability as problem size grows. A cap on the
   jump that tightens with dimension mitigates it
   ([`app/core/qpso_vrp.py`](app/core/qpso_vrp.py)). Without local search the
   effect is large from 60 customers up and absent at 20; with local search on,
   it has no measurable effect.
2. **Memetic hybridization** — QPSO's global search is paired with 2-opt and
   single-customer relocation ([`app/core/local_search.py`](app/core/local_search.py)).
   On its own QPSO trails standard PSO from 40 customers up; with local search
   it leads, at a runtime cost. The same local search added to standard PSO
   does as well, so the gain comes from local search rather than from QPSO.
   Once local search starts, the swarm almost never improves the best solution
   again ([`docs/RESULTS.md`](docs/RESULTS.md) §3–4).

Plus an optional **time-dependent, congestion-aware** extension
([`app/core/traffic_profile.py`](app/core/traffic_profile.py)) that prices
routes against a time-of-day congestion curve instead of static edge weights.
See [Time-dependent congestion — scope and limitations](#time-dependent-congestion--scope-and-limitations)
below before citing this part of the work — it is at an earlier stage of
validation than everything else here.

## Repository layout

```
app/core/
  qpso_vrp.py               # QPSO with jump-cap fix + memetic hybridization
  local_search.py           # 2-opt / single-customer relocation operators
  vrp_problem.py            # CVRPTW formulation, synthetic instance generator
  graph_model.py            # Traffic network model
  exact_vrp.py               # Exact solver for small instances (subset enumeration + DP)
  classical_baselines_vrp.py # GA, SA, standard PSO, greedy nearest-neighbor baselines
  benchmark_vrp.py           # Convergence / scalability / robustness benchmark suite
  benchmark_trials.py        # Repeated (algorithm, seed) runs in a process pool
  benchmark_solomon.py       # Solomon-instance benchmark runner
  solomon.py                 # Solomon (1987) instance parser + best-known lookups
  traffic_profile.py         # Time-of-day congestion model
  route_export.py            # Route serialization (GPX/GeoJSON export)
  impact.py                  # Distance/time savings -> fuel/CO2 conversions

scripts/
  run_solomon_benchmark.py   # Reproduces data/solomon_results.md
  impact_report.py           # Reproduces data/impact_report.md
  measure_exact_vrp.py       # Exact-solver runtime by instance size
  run_ablations.py           # Reproduces data/ablation_results.md (exact gap, jump-cap, local search, QPSO vs PSO + LS)
  measure_swarm_contribution.py  # Reproduces data/swarm_contribution.md (swarm vs local-search share)
  generate_stress_test_cache.py # Measures scalability -> data/stress_test_synthetic.json
  plot_benchmark_charts.py   # Repeated-trial and scalability figures + data/benchmark_charts.md

data/
  solomon_results.md         # Solomon CVRPTW benchmark results (this repo's headline numbers)
  impact_report.md           # Distance/time savings converted to fuel and CO2
  ablation_results.md/.json  # Exact-optimum gap, jump-cap and local-search ablations
  swarm_contribution.md/.json # How much of each memetic run's improvement the swarm vs local search made
  benchmark_charts.md        # Table view of every figure below (repeated trials, scalability)
  stress_test_synthetic.json # Measured scalability, synthetic graph (reproducible offline)
  stress_test_delhi.json     # Measured scalability, New Delhi OSM extract (upstream app)
  trials_*.png, scalability_*.png, ablation_*.png
  benchmarks/solomon/        # Solomon (1987) instance files (C101, C201, R101, R201, RC101, RC201)
  vrp_convergence.png, vrp_scalability_*.png

docs/
  FORMULATION.md             # Mathematical formulation (CVRPTW, objective, QPSO update rule)
  RESULTS.md                 # Results section, written up from the generated data

tests/                        # pytest suite covering the algorithm, local search,
                               # exact solver, Solomon parsing, time-dependent pricing,
                               # repeated trials, and FORMULATION.md-matches-code
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Only `numpy`, `networkx`, `matplotlib`, and `pytest`
are needed — there is no web server, database, or external map service in
this repository.

## Running the tests

```bash
pytest tests/
```

## Reproducing the results

```bash
python scripts/run_solomon_benchmark.py --max-iter 200 --seed 1   # -> data/solomon_results.md
python scripts/impact_report.py                                    # -> data/impact_report.md
python scripts/measure_exact_vrp.py                                 # exact-solver runtime by size
python scripts/run_ablations.py                                     # -> data/ablation_results.md
python scripts/measure_swarm_contribution.py                        # -> data/swarm_contribution.md
python scripts/generate_stress_test_cache.py                        # -> data/stress_test_synthetic.json
python scripts/plot_benchmark_charts.py                             # -> data/benchmark_charts.md + figures
python -m app.core.benchmark_vrp                                    # convergence/scalability plots
```

## The mathematical model

The problem is a CVRPTW over a road graph, optionally time-dependent. The
formal model is in [docs/FORMULATION.md](docs/FORMULATION.md). It covers the
decision variables, the objective, the constraints, and the QPSO update with
the parameter values used.

The objective being minimised is:

$$\mathcal{F} \;=\; w_T \cdot T_{\text{total}} \;+\; w_D \cdot D_{\text{total}} \;+\; \lambda_{\text{cap}} \cdot \mathcal{P}_{\text{cap}} \;+\; \lambda_{\text{time}} \cdot \mathcal{P}_{\text{time}} \;+\; \lambda_{\text{idle}} \cdot \mathcal{P}_{\text{idle}}$$

It is a weighted sum of fleet time and distance ($w_T = 0.6$, $w_D = 0.4$ on the
synthetic instances; Solomon instances use distance alone). Capacity and time
windows are soft penalties. Every solver, including the exact one, calls the
same `evaluate_solution`, and
[`tests/test_formulation_matches_code.py`](tests/test_formulation_matches_code.py)
checks that the equations in the document reproduce its output exactly.

## Results

The full write-up, with the setup, every table and the limitations, is in
[`docs/RESULTS.md`](docs/RESULTS.md). The raw tables are in
[`data/ablation_results.md`](data/ablation_results.md),
[`data/swarm_contribution.md`](data/swarm_contribution.md),
[`data/benchmark_charts.md`](data/benchmark_charts.md),
[`data/solomon_results.md`](data/solomon_results.md) and
[`data/impact_report.md`](data/impact_report.md). Headline claims, each graded
by the strength of its evidence:

| Claim | Evidence | Strength |
|---|---|---|
| QPSO + LS reaches the exact optimum on small instances | 47/60 runs at 6–9 customers; median gap ≤ 0.56% | Strong up to 8 customers; weaker at 9 |
| Jump cap improves QPSO alone as dimension grows | 24–49% lower mean fitness at 60–100 customers, no effect at 20 | Moderate: 5 seeds, one instance per size |
| Jump cap improves QPSO + LS | 8 better / 11 worse / 2 tied on seeds where both are feasible | Not supported: no measurable effect with local search on |
| QPSO + LS beats standard PSO and GA at 20–100 customers | Lowest mean at every size; 24/25 paired wins over PSO | Strong at equal particles/iterations; unequal in time |
| Local search improves swarm metaheuristics on CVRPTW | Improves QPSO on 24/25 and standard PSO on 25/25 paired seeds at 20–100 customers | Strong |
| The gain comes from QPSO specifically | With identical local search, PSO + LS matches QPSO + LS (9 better / 10 worse / 1 tied on seeds where both are feasible) | Not supported |
| The swarm keeps contributing once local search starts | Improved the best in 4 of 450 windows between local-search calls | Not supported: local search does the work |
| Solomon benchmark | Feasible on 4/6; 12.5–34.7% above best-known distance | Moderate: one seed; objective is distance, not Solomon's hierarchy |
| Scales to large instances | Times out at 100 customers (90 s) and 200 (300 s) | Not supported: runtime is the main limitation |
| Time-dependent congestion-aware routing | 5.8% on one instance | Preliminary; see below |

Note on the Solomon table: Solomon's objective is hierarchical (fewest
vehicles first, then shortest distance), and a run that violates a time
window has no meaningful distance to report — scoring it anyway would reward
solutions for skipping constraints. `data/solomon_results.md` reports
feasibility explicitly rather than papering over it.

## Time-dependent congestion — scope and limitations

The congestion-aware extension prices routes against a time-of-day demand
curve (`app/core/traffic_profile.py`) instead of static edge weights. On a
12-customer illustrative instance, the same route leg costs 35 minutes
off-peak vs. 63 minutes at morning peak, and a route plan built while ignoring
time-dependence comes out 27 minutes (5.8%) worse once time-varying cost is
applied.

This result is included as a demonstration of the formulation's value, **not**
as a fully benchmarked contribution on the same footing as the results above.
Specifically, at this stage:

- **Single instance, single measurement** — no 5-seed robustness protocol has
  been applied here yet, unlike the rest of the results.
- **No baseline comparison under time-dependent conditions** — the current
  result shows "time-aware planning beats time-naive planning," which holds
  for any solver. It does not yet show QPSO specifically handles
  time-dependence better than GA/SA/standard PSO/greedy.
- **Possible FIFO consistency risk** — the travel-time matrix is a 30-minute
  bucket step function. Per Ichoua, Gendreau & Potvin (2003), a naively
  bucketed travel-time function can let a later departure arrive earlier
  across a bucket boundary, which would violate the FIFO property
  time-dependent routing normally assumes. This has not yet been formally
  verified against `traffic_profile.py`.
- **Default horizon (06:00–14:00)** misses the evening peak.
- **The congestion curve is a generically calibrated commuter pattern**, not
  measured local traffic data. The amplitudes are stated in
  `traffic_profile.py` and are meant to be challenged or replaced with real
  observed counts, not treated as a measurement.

If you build on this part of the repository, treat it as a formulation and a
worked example, not as a validated empirical result.

## Origin & attribution

This repository is split off from
[`MargdarshaQ`](https://github.com/Dev-saxena11/MargdarshaQ) (originally
built for Smart India Hackathon problem statement SIH26137, "Quantum-Inspired
Intelligent Traffic Route Optimization in Transportation Systems Using
Metaheuristic Techniques"), maintained by Dev Saxena and team. This repository
keeps only the algorithm, benchmarks, and evidence relevant to the paper above
— the web application (FastAPI backend, dashboard frontend, live map/OSM
integration, auth, and AI assistant) has been removed as out of scope for a
reproducibility artifact.

See [`LICENSE`](LICENSE) for licensing status and its dependency on the
original repository, which does not yet carry a license of its own.

## References

Key references for this work (full list in the paper):

- Kennedy, J., Eberhart, R.C. (1995). Particle Swarm Optimization. *Proc. IEEE ICNN*, 1942–1948.
- Sun, J., Feng, B., Xu, W. (2004). Particle Swarm Optimization with Particles Having Quantum Behavior. *Proc. CEC 2004*, 325–331.
- Li, Y., Li, D., Wang, D. (2012). QPSO Based on Border Mutation and Chaos for VRP. *LNCS* vol. 7331, 63–73.
- Croes, G.A. (1958). A method for solving traveling-salesman problems. *Operations Research* 6(6), 791–812.
- Or, I. (1976). *Traveling Salesman-Type Combinatorial Problems and Their Relation to the Logistics of Regional Blood Banking*. Ph.D. thesis, Northwestern University.
- Ichoua, S., Gendreau, M., Potvin, J.-Y. (2003). Vehicle dispatching with time-dependent travel times. *EJOR* 144(2), 379–396.
- Solomon, M.M. (1987). Algorithms for the VRP and Scheduling Problems with Time Window Constraints. *Operations Research* 35(2), 254–265.
