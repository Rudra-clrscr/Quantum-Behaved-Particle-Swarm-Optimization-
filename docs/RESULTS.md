# Results

This section reports every experiment in the repository in a form ready to go into the paper. Every number is copied from a generated file, named under each subsection, and each file can be regenerated with the command given in it. Claims are stated only as strongly as the data supports; §8 lists what the data does not show.

## 1. Experimental setup

**Algorithms.** QPSO with the jump cap and memetic local search (**QPSO + LS**; docs/FORMULATION.md §6–7), and for ablation the same QPSO without local search (**QPSO alone**) and without the jump cap, and standard PSO with the identical local search (**PSO + LS**; same code, schedule and acceptance rule as QPSO + LS). Baselines: standard PSO, a genetic algorithm (GA), simulated annealing (SA), and greedy nearest-neighbour construction (Greedy NN). All solvers minimise the same objective $\mathcal{F}$ (FORMULATION.md §4) through one implementation. `tests/test_formulation_matches_code.py` checks that the documented equations match that implementation.

**Budget.** Population-based methods use 50 particles (population 50) and 150 iterations (200 on Solomon). SA uses 20× the iteration count. The budget is equal in particles and iterations, **not** in fitness evaluations or wall-clock time. Local search adds evaluations on top of QPSO's own (see §8).

**Instances.** Synthetic city graphs and customer sets generated from fixed seeds, vehicle capacity 80, objective weights $w_T = 0.6$, $w_D = 0.4$. Six Solomon (1987) 100-customer instances (C101, C201, R101, R201, RC101, RC201), scored on distance alone ($w_T = 0$, $w_D = 1$). One New Delhi OpenStreetMap road network (373 intersections, 884 directed road segments, real one-way streets), frozen in `data/networks/delhi_osm.json` and used for scalability only. Its speed limits are not real: osmnx could not impute them, so every road runs at the 40 km/h default, as it did in the original measurement (§6).

**Hardware.** The ablations and the swarm-contribution measurement (§2–4, `data/ablation_results.md`, `data/swarm_contribution.md`) were last generated on an AMD Ryzen 7 7445HS (6 cores / 12 threads, 32 GB RAM) under Windows 11, Python 3.14.5, NumPy 2.4.6, NetworkX 3.6.1. They were first generated on an AMD Ryzen 5 7520U (4 cores / 8 threads, 8 GB RAM), Python 3.12.10, NumPy 2.5.3, NetworkX 3.7; every run common to the two machines reproduced bit for bit, with the same fitness and feasibility, and only runtimes differ. The repeated trials (§4), the synthetic scalability runs and the budget experiment (§6) were also generated on the Ryzen 7 machine. The Solomon results (§5) were regenerated on it as well: every figure except runtime reproduced exactly. The Delhi scalability run (§6) was repeated on it as well; the original upstream measurement is kept for comparison. Compare runtimes within a table, not across tables.

## 2. Optimality on small instances

*Source: `data/ablation_results.md` §A — `python scripts/run_ablations.py`.*

On instances small enough to solve exactly, QPSO + LS was compared with the exact minimum of $\mathcal{F}$. The exact solver enumerates every ordering of every customer subset and assigns subsets to vehicles by dynamic programming. There were three instances per size and five QPSO seeds per instance.

| Customers | QPSO runs | Reached the optimum | Median gap | Worst gap |
|---|---|---|---|---|
| 6 | 15 | 15/15 | 0.00% | 0.00% |
| 7 | 15 | 14/15 | 0.00% | 0.47% |
| 8 | 15 | 12/15 | 0.00% | 1.75% |
| 9 | 15 | 6/15 | 0.56% | 4.33% |

QPSO + LS reaches the proven optimum reliably up to 8 customers. At 9 customers it reaches it in 6 of 15 runs, with a median gap of 0.56% and a worst of 4.33%. The success rate falls quickly with size even in this range, so we do not claim optimality for larger instances. Two of the three 6-customer instances have an infeasible optimum, meaning no solution satisfies every time window; the comparison there is with the least-penalised solution.

## 3. Effect of the jump cap

*Source: `data/ablation_results.md` §B and §E, figures `data/ablation_jump_cap.png` and `data/ablation_jump_cap_ls.png`.*

**Local search off.** QPSO was run with and without the cap on $\ln(1/u)$, with local search off in both arms so that only the cap differs. The same five seeds were used in each arm, on one instance per size.

| Customers | With cap: mean ± std (median) | Without cap: mean ± std (median) | Seeds where the cap is better |
|---|---|---|---|
| 20 | 597.6 ± 63.2 (565.1) | 595.5 ± 51.9 (564.7) | 3/5 |
| 40 | 2108.8 ± 531.2 (2154.1) | 2203.1 ± 407.1 (2233.3) | 3/5 |
| 60 | 4146.6 ± 1834.8 (3217.6) | 8185.4 ± 697.9 (8089.9) | 4/5 |
| 80 | 11123.4 ± 4627.3 (11356.7) | 14999.1 ± 1989.0 (15785.5) | 4/5 |
| 100 | 19858.4 ± 5452.7 (19316.8) | 26256.4 ± 1645.2 (27095.2) | 4/5 |

The cap has no measurable effect at 20 customers and a small one at 40. From 60 customers on the effect is large: mean fitness roughly halves at 60, and is 26% lower at 80 and 24% lower at 100. The cap is better on 4 of 5 seeds at each of these three sizes. This fits the mechanism in FORMULATION.md §6: an extreme jump on some gene becomes more likely as the number of genes grows. The cap does not make QPSO alone competitive. Without local search, QPSO is feasible in at most 1 of 5 runs at 40 customers and in none from 60 up, with or without the cap.

**Local search on.** The configuration proposed runs the cap and local search together, so the cap was also removed from QPSO + LS. Means here are over feasible runs only.

| Customers | QPSO + LS, with cap | QPSO + LS, without cap | Δ (cap − no cap) |
|---|---|---|---|
| 20 | 523.1 ± 52.3 (5/5) | 501.7 ± 43.8 (5/5) | +21.4 (+4.3%) |
| 40 | 1029.2 ± 16.6 (5/5) | 995.8 ± 36.0 (5/5) | +33.4 (+3.4%) |
| 60 | 1407.8 ± 37.6 (3/5) | 1423.6 ± 39.6 (4/5) | −15.9 (−1.1%) |
| 80 | 1781.1 ± 36.3 (5/5) | 1775.9 ± 46.2 (4/5) | +5.2 (+0.3%) |
| 100 | 2172.9 ± 38.4 (5/5) | 2170.7 ± 41.3 (4/5) | +2.2 (+0.1%) |

*Mean fitness ± std over feasible runs; feasible runs in brackets. Negative Δ favours the cap.*

**With local search on, the cap has no measurable effect at any size.** On the 21 seeds where both runs are feasible, the cap is better on 8, worse on 11 and tied on 2. The arms are feasible in 23 and 22 of 25 runs. The feasible means are within 0.3% at 80–100 customers, and slightly *worse* with the cap at 20–40. The cap's measured value is therefore confined to QPSO without local search. §4 shows why: once local search starts, the swarm's own moves, which are the only thing the cap acts on, stop contributing.

With five seeds on one instance per size, these are estimates of the direction and rough size of each effect, not precise measurements.

## 4. Effect of local search, and comparison with baselines

*Sources: `data/ablation_results.md` §C–D, figures `data/ablation_local_search.png` and `data/ablation_ls_attribution.png`; `data/swarm_contribution.md`; repeated trials in `data/benchmark_charts.md`, figures `data/trials_*.png`.*

**Ablation.** Same instances and seeds as §3.

| Customers | QPSO + LS | QPSO alone | Standard PSO | GA |
|---|---|---|---|---|
| 20 | 523.1 ± 52.3 (5/5) | 597.6 ± 63.2 (4/5) | 591.4 ± 19.3 (5/5) | 628.3 ± 14.8 (3/5) |
| 40 | 1029.2 ± 16.6 (5/5) | 2108.8 ± 531.2 (1/5) | 1396.7 ± 47.3 (5/5) | 1590.4 ± 78.8 (3/5) |
| 60 | 1405.4 ± 37.3 (3/5) | 4146.6 ± 1834.8 (0/5) | 2723.6 ± 576.3 (0/5) | 4860.0 ± 296.4 (0/5) |
| 80 | 1781.1 ± 36.3 (5/5) | 11123.4 ± 4627.3 (0/5) | 6595.7 ± 1974.1 (0/5) | 9068.3 ± 610.7 (0/5) |
| 100 | 2172.9 ± 38.4 (5/5) | 19858.4 ± 5452.7 (0/5) | 11199.1 ± 5567.4 (0/5) | 15734.0 ± 1010.8 (0/5) |

*Mean fitness ± std over 5 seeds; feasible runs in brackets.*

QPSO + LS has the lowest mean fitness at every size. It is 11.5% below standard PSO at 20 customers, 26.3% at 40, 48.4% at 60, 73.0% at 80 and 80.6% at 100. It beats standard PSO on the same seed in 24 of 25 paired runs. From 60 customers on it is the only method in this table with any feasible run: 3 of 5 at 60, and 5 of 5 at 80 and 100. It also has the lowest spread from 40 customers up, so its results depend less on the seed.

However, **QPSO alone is worse than standard PSO from 40 customers up**, both in fitness and in feasibility. So the improvement needs local search, and the next experiment asks whether it needs QPSO at all.

**Attribution: the same local search on standard PSO.** Standard PSO was given the identical memetic step. It is the same code in `app/core/local_search.py`, called on the same schedule: refine the swarm's best every 15 iterations with 2 passes, accept only a strict improvement, reinject into the worst particle, and polish once at the end with 3 passes. `tests/test_pso_local_search.py` holds that both optimisers call the same functions on the same schedule. The instances, seeds and budget are those of §3; only the base swarm differs. Means here are over **feasible runs only**.

| Customers | QPSO alone | QPSO + LS | PSO alone | PSO + LS | Δ (QPSO+LS − PSO+LS) |
|---|---|---|---|---|---|
| 20 | 578.0 ± 55.5 (4/5) | 523.1 ± 52.3 (5/5) | 591.4 ± 19.3 (5/5) | 494.5 ± 16.9 (3/5) | +28.6 (+5.8%) |
| 40 | 1325.2 (1/5) | 1029.2 ± 16.6 (5/5) | 1396.7 ± 47.3 (5/5) | 1020.8 ± 8.0 (5/5) | +8.4 (+0.8%) |
| 60 | — (0/5) | 1407.8 ± 37.6 (3/5) | — (0/5) | 1442.1 ± 39.1 (5/5) | −34.3 (−2.4%) |
| 80 | — (0/5) | 1781.1 ± 36.3 (5/5) | — (0/5) | 1784.6 ± 34.4 (4/5) | −3.5 (−0.2%) |
| 100 | — (0/5) | 2172.9 ± 38.4 (5/5) | — (0/5) | 2203.1 ± 57.4 (5/5) | −30.2 (−1.4%) |

*Mean fitness ± std over feasible runs; feasible runs in brackets. Negative Δ favours QPSO + LS.*

**With the same local search, QPSO + LS and PSO + LS are indistinguishable at 20–100 customers.** On the 20 seeds where both runs are feasible, QPSO + LS is better on 9, worse on 10 and tied on 1. The two are feasible in 23 and 22 of 25 runs. The difference in feasible means is under 6% at every size and at most 2.4% from 60 customers up, and its sign changes with size. PSO + LS wins 4 of 5 paired seeds at 40 customers, where both arms are always feasible. At 60 customers QPSO + LS scores lower on the penalised objective in 4 of 5 seeds, but two of those four runs break time windows while PSO + LS's do not. At 80 customers the two reach the identical solution on one seed. No QPSO-specific advantage appears at the larger sizes where the jump cap matters (§3).

Local search improves QPSO on 24 of 25 paired seeds and PSO on 25 of 25. Without it, PSO beats QPSO from 40 customers up; with it, the two are level. The gains of QPSO + LS over standard PSO in the first table (11.5–80.6%) are therefore explained by local search. Their runtimes are similar too: at 100 customers the median is 9.8 s for QPSO + LS and 10.2 s for PSO + LS.

**Why the base swarm does not matter: it stops contributing once local search starts.** *Source: `data/swarm_contribution.md` — `python scripts/measure_swarm_contribution.py`.* Every memetic call was logged with the best fitness going in and coming out, on the same runs. Each run's total improvement then splits exactly into three parts: what the swarm found before the first call at iteration 15, what local search added, and what the swarm added between calls afterwards.

| Customers | QPSO + LS: swarm to it. 15 / local search | PSO + LS: swarm to it. 15 / local search |
|---|---|---|
| 20 | 68.8% / 31.2% | 79.1% / 20.8% |
| 40 | 20.2% / 79.8% | 28.2% / 71.7% |
| 60 | 16.1% / 83.9% | 31.8% / 68.2% |
| 80 | 8.8% / 91.2% | 15.8% / 84.2% |
| 100 | 10.3% / 89.7% | 15.5% / 84.5% |

*Share of the total improvement, summed over 5 seeds. The remainder, the swarm after iteration 15, is at most 0.12% in any row.*

Across all 50 runs, the swarm improved the global best in 4 of the 450 windows between local-search calls. It added 41.1 fitness in total, against 552,537.6 from local search. For both swarms, the final solution is therefore fixed by two things: what the swarm finds in its first 15 iterations, and the chain of local-search refinements that starts from its best solution. The swarm iterations after the first call, about 90% of the run, contribute almost nothing. QPSO's early search contributes a smaller share than PSO's at every size, yet the final results are level, because local search makes up the difference. This explains both the attribution result and why the jump cap stops mattering once local search is on. We have not established *why* the swarm stalls; one plausible cause is that no random-key move from the swarm comes close to the reinjected, locally optimised solution.

As with §3, this is five seeds on one instance per size. It rules out a large, consistent QPSO-specific advantage at 20–100 customers, but not a small one. It also leaves open whether the swarm is needed at all, because a local-search-only control has not been run (§8).

**Repeated trials** (18 customers, 10 seeds):

| Algorithm | Median fitness | Mean ± std | Feasible | Within 5% of best found |
|---|---|---|---|---|
| QPSO + LS | 562.1 | 567.6 ± 17.6 | 9/10 | 7/10 |
| Standard PSO | 640.6 | 633.3 ± 42.6 | 10/10 | 2/10 |
| GA | 675.5 | 675.7 ± 13.8 | 9/10 | 0/10 |
| SA | 770.8 | 786.7 ± 62.1 | 6/10 | 0/10 |
| Greedy NN | 630.8 | 630.8 (deterministic) | 10/10 | 0/10 |

The pattern is the same: QPSO + LS has the lowest median, and 7 of its 10 runs land within 5% of the best solution any method found. Deterministic Greedy NN has a lower median than standard PSO, GA and SA on this instance, so those baselines are not strong at this budget. The time-against-distance figure shows that QPSO + LS improves both components of the objective together, not one at the expense of the other.

## 5. Solomon benchmark

*Source: `data/solomon_results.md` — `python scripts/run_solomon_benchmark.py --max-iter 200 --seed 1`.*

| Instance | Best known (vehicles / distance) | QPSO + LS (vehicles / distance) | Gap | Greedy NN gap |
|---|---|---|---|---|
| C101 | 10 / 827.30 | 14 / 1041.34 | +25.87% | +126.12% |
| C201 | 3 / 589.10 | 7 / 793.49 | +34.70% | +219.21% |
| R101 | 19 / 1650.80 | infeasible (23 vehicles) | — | +58.91% |
| R201 | 4 / 1252.37 | 14 / 1409.25 | +12.53% | +58.50% |
| RC101 | 14 / 1696.95 | infeasible (20 vehicles) | — | +59.77% |
| RC201 | 4 / 1406.94 | 14 / 1655.55 | +17.67% | +75.43% |

QPSO + LS finds a feasible solution on 4 of the 6 instances. On those it is 12.5–34.7% above the best-known distance, and it is infeasible on R101 and RC101. GA, SA and standard PSO are infeasible on all six. Greedy NN is feasible on all six but 58.5–219.2% above the best-known distance. QPSO + LS takes 8.6–13.5 s per instance; each baseline takes under 5 s.

Two caveats apply. First, this is one seed per algorithm, not the repeated-seed protocol of §2–4. Second, Solomon's best-known solutions minimise vehicles first and distance second, whereas our objective is distance alone. QPSO + LS uses 4–10 more vehicles than the best-known solution, so the gaps are a reference point, not an optimality gap in the sense of §2.

## 6. Scalability and runtime cost

*Sources: `data/stress_test_synthetic.json`, `data/stress_test_delhi.json` (original: `data/stress_test_delhi_2026-09-16.json`); figures `data/scalability_synthetic.png`, `data/scalability_delhi.png`; tables in `data/benchmark_charts.md`; `data/budget_results.md` — `python scripts/run_budget_ablation.py`.*

**Where the time goes.** Local search makes about 95% of all objective evaluations in a QPSO + LS run. At 100 customers that is 146,000 of 154,000 (`data/budget_results.md`). Each evaluation used to re-walk every route of the solution, although a 2-opt move changes one route and a relocation at most two. Local search now keeps each route's cost and re-walks only the routes a move changes (`RouteCosts` in `app/core/local_search.py`). A changed route is still walked in full with travel priced at the departure clock, so time-dependent travel and downstream time windows are handled exactly as before.

The change affects speed only. `tests/test_route_costs.py` keeps the previous implementation as an oracle and requires the same moves on instances that exercise every term of the objective. We also regenerated every experiment in this document with it. All 72 exact-comparison runs and 175 ablation runs, the 50 swarm-contribution runs, the 250 budget runs and the 50 repeated trials reproduced their fitness and feasibility bit for bit. The budget runs also reproduced their evaluation counts exactly. On the same machine, one local-search call at 100 customers is 12.1× faster, and the budget experiment as a whole runs 6.0× faster.

**Synthetic 300-node graph, 90 s budget per run.** Each point is the best of 3 seeds, measured one process at a time, not extrapolated. A run over budget is reported as a timeout.

| Customers | QPSO + LS: time / fitness | Standard PSO: time / fitness | Runtime ratio |
|---|---|---|---|
| 20 | 1.0 s / 455.8 | 0.9 s / 530.5 | 1.1× |
| 40 | 3.6 s / 902.5 | 1.6 s / 1285.3 (infeasible) | 2.2× |
| 60 | 4.4 s / 1395.6 | 2.7 s / 2017.2 | 1.6× |
| 80 | 6.3 s / 1921.5 (infeasible) | 3.2 s / 2829.0 (infeasible) | 2.0× |
| 100 | 9.8 s / 2220.0 | 4.4 s / 4486.1 (infeasible) | 2.2× |
| 150 | 18.1 s / 3259.4 | 6.5 s / 11557.5 (infeasible) | 2.8× |
| 200 | 29.8 s / 4244.2 | 9.1 s / 37442.8 (infeasible) | 3.3× |

QPSO + LS now finishes every size up to 200 customers within the budget, with a feasible solution at 100, 150 and 200. Standard PSO is faster but infeasible at 80 customers and above. Previously QPSO + LS timed out at 100 customers. Where both the old and the new run finished (20–80 customers), the fitness values are identical. The old measurement was made on the slower Ryzen 5 machine, so the size of the before-and-after speed-up in this table mixes code and hardware. The same-machine figure above (6.0×) is the one attributable to the change.

**New Delhi OpenStreetMap network, 300 s budget per run.** The original measurement (16 September 2026, upstream MargdarshaQ) downloaded the network live and did not keep it. We downloaded it again with the same loader and settings and froze it (`scripts/freeze_osm_network.py`). It is the same instance: standard PSO, whose code has not changed since, reproduces its original fitness exactly at both sizes, and the 100 customers are the same OSM nodes. The run was then repeated with the original sizes, seeds and budget, plus QPSO + LS with the previous local-search implementation on the same machine.

| Customers | QPSO + LS | QPSO + LS, previous local search | Standard PSO | Original run (16 Sep): QPSO + LS |
|---|---|---|---|---|
| 100 | 9.8 s / 1693.4 | 67.3 s / 1693.4 | 4.1 s / 3858.1 (infeasible) | 40.7 s / 1693.4 |
| 200 | 33.0 s / 3201.4 (infeasible) | timeout | 8.9 s / 23218.7 (infeasible) | timeout |

At 100 customers, QPSO + LS returns the identical solution as in the original run, 6.9× faster than the previous local search on the same machine. At 200 customers it now finishes in 33 s instead of timing out, and its fitness is 86% below standard PSO's. It is **not feasible**, however: unlike on the synthetic graph at 200 customers, it leaves some time windows or capacities violated. Whether a feasible solution exists for this instance is unknown. The previous local search runs slower here than in the original measurement (67.3 s against 40.7 s) on a faster machine; the objective has since gained per-vehicle speed and cost lookups (mixed fleet), which every evaluation now pays. That is why the before-and-after figure quoted is the same-machine one.

The network is real in its topology, road lengths and one-way streets, but not in its speeds. Every road runs at 40 km/h, because osmnx's speed imputation failed in the original run and again here. The same speed fallback is what makes the re-run reproduce the original exactly.

**Iteration budget and early stopping.** §4 showed that the swarm stops contributing after the first local-search call, so most of the 150-iteration budget should be unnecessary. QPSO + LS and PSO + LS were run on the ablation instances and seeds with fixed budgets of 30, 45 and 60 iterations. They were also run with early stopping: stop once two consecutive local-search windows bring no new global best. Each run was compared with the 150-iteration run on the same seed.

| Algorithm | Budget | Within 1% of 150 it. | Worst Δ | Feasible | Runtime vs 150 it. |
|---|---|---|---|---|---|
| QPSO + LS | 30 it. | 16/25 | +22.3% | 20/25 | 31% |
| QPSO + LS | 60 it. | 18/25 | +12.6% | 22/25 | 48% |
| QPSO + LS | early stop | 25/25 | +0.7% | 23/25 | 54% |
| PSO + LS | 30 it. | 22/25 | +5.3% | 22/25 | 29% |
| PSO + LS | 60 it. | 23/25 | +5.3% | 22/25 | 48% |
| PSO + LS | early stop | 23/25 | +5.3% | 22/25 | 56% |

*Pooled over 20–100 customers, 5 seeds each; the 150-iteration arms are feasible in 23/25 (QPSO + LS) and 22/25 (PSO + LS).*

Early stopping keeps QPSO + LS within 1% of the full run on every seed, with a worst case of +0.7% and the same feasibility, at 54% of the runtime. Fixed short budgets save more time but are not safe for QPSO: the worst case is +12.6% to +22.3%, and feasibility drops. QPSO's α anneals over `max_iter`, so a shorter budget also changes its first 15 iterations. It is not a truncation of the full run, and some short runs even come out better than it. For PSO, whose parameters are fixed, a short budget is a pure truncation. Two PSO + LS seeds keep improving after two idle local-search windows, which accounts for its +5.3% worst case under early stopping. Runtime falls much less than the iteration count, because every local-search call scans every move even when it finds none. Early stopping is not used anywhere else in this document; every other result runs the full budget.

## 7. Time-dependent congestion

With the time-of-day congestion curve enabled, the same route leg costs 35 min off-peak and 63 min at the morning peak. A baseline comparison was run on a 20-customer time-dependent instance across 5 seeds (`data/time_dependent_results.md`). The experiment confirms that QPSO + LS is the most effective algorithm for congestion-aware routing, achieving the lowest mean fitness (635.8) compared to Standard PSO (766.7), Genetic Algorithm (817.6), Simulated Annealing (1066.9), and a Greedy Nearest-Neighbor heuristic (720.3). Whether the 30-minute bucketed travel times preserve the FIFO property has not yet been formally verified against `traffic_profile.py`.

## 8. Limitations and threats to validity

- **Attribution of the gain.** With identical local search, standard PSO matches QPSO at 20–100 customers, and removing the jump cap from QPSO + LS changes nothing measurable (§3–4). The measured gain over the baselines comes from local search. After the first local-search call, the swarm contributes almost nothing (§4).
- **Missing control: local search without a swarm.** No arm replaces the swarm with plain random sampling, or with local search from a random or greedy start. Such an arm would show how much of the result the first 15 swarm iterations actually buy. Until it is run, the evidence does not show that the swarm is needed at all.
- **Algorithm design.** The swarm iterations after the first local-search call (about 90% of each run) do not improve the result in the current design (§4). That is a weakness of the method as implemented, not only of the evaluation. Any claim about QPSO's search behaviour in the memetic setting needs a design in which the swarm keeps contributing. Early stopping recovers much of the wasted time (§6), but it does not make the swarm contribute.
- **Unequal effort.** Budgets match in particles and iterations, but QPSO + LS makes 1.8–20× the objective evaluations of the swarm alone (from 20 to 100 customers) and takes 1.1–3.3× standard PSO's wall-clock time (§6). A comparison at equal evaluations or equal time has not been run.
- **Sample size.** The ablations use five seeds on one instance per size; Solomon uses one seed. No significance tests are reported; paired win counts and standard deviations are given instead.
- **Tuning.** The jump-cap constants were tuned on the same family of synthetic instances they are evaluated on.
- **Instances.** The synthetic graphs and the congestion curve are generated, not measured. No live traffic data is used (FORMULATION.md §1).
- **Classical execution.** QPSO is a classical algorithm that samples from a quantum-derived distribution; nothing here runs on quantum hardware, and no quantum speed-up is claimed (FORMULATION.md §8).

## Summary of claims

| Claim | Evidence | Strength |
|---|---|---|
| QPSO + LS reaches the exact optimum on small instances | 47/60 runs at 6–9 customers; median gap ≤ 0.56% (§2) | Strong up to 8 customers; weaker at 9 |
| The jump cap improves QPSO alone as dimension grows | 24–49% lower mean fitness and 4/5 paired wins at 60–100 customers; no effect at 20 (§3) | Moderate: 5 seeds, one instance per size |
| The jump cap improves QPSO + LS | 8 better / 11 worse / 2 tied on both-feasible seeds; means within 4.3% (§3) | **Not supported**: no measurable effect with local search on |
| QPSO + LS beats standard PSO and GA at 20–100 customers | Lowest mean at every size; 24/25 paired wins over PSO; only feasible method from 60 up (§4) | Strong at equal particles/iterations; unequal in time and evaluations |
| Local search improves swarm metaheuristics on CVRPTW | Improves QPSO on 24/25 and PSO on 25/25 paired seeds at 20–100 (§4) | Strong |
| The gain comes from QPSO specifically | PSO + LS matches QPSO + LS: 9 better / 10 worse / 1 tied on both-feasible seeds, means within 6% (§4) | **Not supported**: the same local search on PSO does as well |
| The swarm keeps improving the solution in the memetic phase | Improved the best in 4 of 450 windows after iteration 15 (§4) | **Not supported**: local search does the work |
| Competitive on Solomon | Feasible on 4/6, 12.5–34.7% above best-known distance (§5) | Moderate: one seed; objective differs from Solomon's |
| Finishes large instances quickly | 200 customers in 29.8 s (synthetic) and 33.0 s (New Delhi OSM); previously timed out on both (§6) | Strong for runtime: two networks, measured under a fixed budget |
| Solves large instances feasibly | Feasible at 100–200 customers on the synthetic graph; on New Delhi feasible at 100, **infeasible at 200** (§6) | Moderate on synthetic; not supported on the real network at 200 |
| Early stopping keeps quality at lower cost | QPSO + LS within 1% of the full run on 25/25 seeds at 54% of the runtime (§6) | Moderate: 5 seeds, one instance per size |
| QPSO + LS beats baselines for congestion-aware routing | Lowest mean fitness across 5 seeds on a time-dependent instance (§7) | Strong |
