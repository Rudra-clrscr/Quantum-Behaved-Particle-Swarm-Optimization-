# Time-Dependent Baseline Comparison

This experiment measures whether QPSO+LS remains the strongest algorithm when
routing over a time-dependent congestion profile. Evaluated over 5 seeds on a
20-customer synthetic instance.

| Algorithm | Mean Fitness | Best | Feasible Rate | Median Runtime (ms) |
|---|---|---|---|---|
| QPSO + local search | 635.8 ± 23.6 | 602.9 | 100% | 1362 |
| Standard PSO | 766.7 ± 54.2 | 713.4 | 100% | 1101 |
| Genetic Algorithm | 817.6 ± 39.3 | 775.4 | 80% | 1321 |
| Simulated Annealing | 1066.9 ± 92.9 | 931.3 | 40% | 508 |
| Greedy Nearest-Neighbor | 720.3 ± 0.0 | 720.3 | 100% | 1 |
