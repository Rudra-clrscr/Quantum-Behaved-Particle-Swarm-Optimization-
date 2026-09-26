# Results

This section reports every experiment in the repository in a form ready to go into the paper. Every number is copied from a generated file, named under each subsection, and each file can be regenerated with the command given in it. Claims are stated only as strongly as the data supports; §8 lists what the data does not show.

## 1. Experimental setup

**Algorithms.** QPSO with the jump cap and memetic local search (**QPSO + LS**; docs/FORMULATION.md §6–7), and for ablation the same QPSO without local search (**QPSO alone**) and without the jump cap, and standard PSO with the identical local search (**PSO + LS**; same code, schedule and acceptance rule as QPSO + LS). Baselines: standard PSO, a genetic algorithm (GA), simulated annealing (SA), and greedy nearest-neighbour construction (Greedy NN). All solvers minimise the same objective $\mathcal{F}$ (FORMULATION.md §4) through one implementation. `tests/test_formulation_matches_code.py` checks that the documented equations match that implementation.

**Budget.** Population-based methods use 50 particles (population 50) and 150 iterations (200 on Solomon). SA uses 20× the iteration count. The budget is equal in particles and iterations, **not** in fitness evaluations or wall-clock time. Local search adds evaluations on top of QPSO's own (see §8).

**Instances.** Synthetic city graphs and customer sets generated from fixed seeds, vehicle capacity 80, objective weights $w_T = 0.6$, $w_D = 0.4$. Six Solomon (1987) 100-customer instances (C101, C201, R101, R201, RC101, RC201), scored on distance alone ($w_T = 0$, $w_D = 1$). One locked New Delhi OpenStreetMap extract, used for scalability only.

**Hardware.** The ablations and the swarm-contribution measurement (§2–4, `data/ablation_results.md`, `data/swarm_contribution.md`) were last generated on an AMD Ryzen 7 7445HS (6 cores / 12 threads, 32 GB RAM) under Windows 11, Python 3.14.5, NumPy 2.4.6, NetworkX 3.6.1. They were first generated on an AMD Ryzen 5 7520U (4 cores / 8 threads, 8 GB RAM), Python 3.12.10, NumPy 2.5.3, NetworkX 3.7; every run common to the two machines reproduced bit for bit, with the same fitness and feasibility, and only runtimes differ. The repeated trials (§4) and the synthetic scalability runs (§6) come from the Ryzen 5 machine. The Solomon results (§5) and the Delhi scalability run (§6) were produced earlier, on different machines. Compare runtimes within a table, not across tables.

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

Local search improves QPSO on 24 of 25 paired seeds and PSO on 25 of 25. Without it, PSO beats QPSO from 40 customers up; with it, the two are level. The gains of QPSO + LS over standard PSO in the first table (11.5–80.6%) are therefore explained by local search. Their runtimes are similar too: at 100 customers the median is 66.0 s for QPSO + LS and 60.5 s for PSO + LS.

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

QPSO + LS finds a feasible solution on 4 of the 6 instances. On those it is 12.5–34.7% above the best-known distance, and it is infeasible on R101 and RC101. GA, SA and standard PSO are infeasible on all six. Greedy NN is feasible on all six but 58.5–219.2% above the best-known distance. QPSO + LS takes 58.7–74.4 s per instance; each baseline takes under 5 s.

Two caveats apply. First, this is one seed per algorithm, not the repeated-seed protocol of §2–4. Second, Solomon's best-known solutions minimise vehicles first and distance second, whereas our objective is distance alone. QPSO + LS uses 4–10 more vehicles than the best-known solution, so the gaps are a reference point, not an optimality gap in the sense of §2.

## 6. Scalability and runtime cost

*Sources: `data/stress_test_synthetic.json`, `data/stress_test_delhi.json`; figures `data/scalability_synthetic.png`, `data/scalability_delhi.png`; tables in `data/benchmark_charts.md`.*

Each point is the best of 3 seeds. Runtimes are measured one process at a time, not extrapolated. A run that exceeds the budget is reported as a timeout.

**Synthetic 300-node graph, 90 s budget per run:**

| Customers | QPSO + LS: time / fitness | Standard PSO: time / fitness | Runtime ratio |
|---|---|---|---|
| 20 | 3.2 s / 455.8 | 1.6 s / 530.5 | 2.0× |
| 40 | 16.1 s / 902.5 | 3.7 s / 1285.3 (infeasible) | 4.3× |
| 60 | 33.5 s / 1395.6 | 7.4 s / 2017.2 | 4.5× |
| 80 | 87.3 s / 1921.5 (infeasible) | 9.8 s / 2829.0 (infeasible) | 8.9× |
| 100 | timeout (all 3 seeds) | 8.8 s / 4486.1 (infeasible) | — |

**New Delhi OSM extract, 300 s budget per run:** at 100 customers, QPSO + LS took 40.7 s for fitness 1693.4 (feasible), against 2.6 s and 3858.1 (infeasible) for standard PSO. At 200 customers, QPSO + LS did not finish; standard PSO finished in 14.4 s with an infeasible solution (23218.7).

Better solutions come at a large runtime cost that grows with instance size. The cost comes mostly from local search, whose operators re-evaluate the full objective for every candidate move: in the ablation (§4), median runtime at 60 customers is 3.0 s for QPSO alone and 17.9 s with local search, and 2.7 s against 16.9 s for standard PSO; at 100 customers it is 4.6 s against 66.0 s for QPSO. Because of it, QPSO + LS in its current Python implementation is not practical beyond about 80 customers under a 90 s budget on the machine and 300-node graph above. The ablation did finish 100 customers (on a 150-node graph and the faster machine in §1, with runs sharing the CPU), so this limit depends on hardware and graph size. The figures in this section are the ones measured under a fixed budget.

## 7. Time-dependent congestion (preliminary)

*Source: README, "Time-dependent congestion — scope and limitations".*

With the time-of-day congestion curve enabled, the same route leg costs 35 min off-peak and 63 min at the morning peak. On one 12-customer instance, a plan optimised while ignoring time-dependence is 27 min (5.8%) worse than a time-aware plan when both are priced with time-varying travel times. This is a single instance and a single run, with no baseline comparison under time-dependence. Whether the 30-minute bucketed travel times preserve the FIFO property has not been checked. We report it as a demonstration of the formulation, not as a validated result.

## 8. Limitations and threats to validity

- **Attribution of the gain.** With identical local search, standard PSO matches QPSO at 20–100 customers, and removing the jump cap from QPSO + LS changes nothing measurable (§3–4). The measured gain over the baselines comes from local search. After the first local-search call, the swarm contributes almost nothing (§4).
- **Missing control: local search without a swarm.** No arm replaces the swarm with plain random sampling, or with local search from a random or greedy start. Such an arm would show how much of the result the first 15 swarm iterations actually buy. Until it is run, the evidence does not show that the swarm is needed at all.
- **Algorithm design.** The swarm iterations after the first local-search call (about 90% of each run) do not improve the result in the current design (§4). That is a weakness of the method as implemented, not only of the evaluation. Any claim about QPSO's search behaviour in the memetic setting needs a design in which the swarm keeps contributing.
- **Unequal effort.** Budgets match in particles and iterations, but QPSO + LS spends extra fitness evaluations and 2–9× the wall-clock time (§6). A comparison at equal evaluations or equal time has not been run.
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
| Scales to large instances | Timeout at 100 (synthetic, 90 s) and 200 (Delhi, 300 s) (§6) | **Not supported**: runtime is the main limitation |
| Time-dependent routing helps | 5.8% on one instance (§7) | Preliminary |
