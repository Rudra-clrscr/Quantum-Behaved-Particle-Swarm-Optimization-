# Mathematical Formulation — Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)

This document provides the formal mathematical model, constraint equations, decision variable definitions, and quantum-behaved particle swarm optimization (QPSO) formulation for the congestion-aware CVRPTW solver in this repository.

---

## 1. Problem Description & Graph Network Model

Let the urban road network be represented as a weighted directed graph:
$$G = (V, E)$$
where:
- $V = \{0\} \cup C$ is the set of all vertices, with vertex $0$ denoting the central depot and $C = \{1, 2, \dots, N\}$ representing the set of customer delivery locations.
- $E = \{(u, v) \mid u, v \in V, u \neq v\}$ is the set of traversable road segments.

### Network Edge Weights & Traffic Congestion
Each road edge $(u, v) \in E$ has:
- Physical road distance: $D(u, v) \ge 0$ (kilometers)
- Base free-flow travel time: $T_0(u, v) \ge 0$ (minutes)
- Congestion friction factor: $C(u, v) \ge 1.0$, a static per-edge value. Scripted incidents can raise it on individual edges for a set period. No live traffic data is used.

The effective travel time along edge $(u, v)$ is:
$$T(u, v) = T_0(u, v) \cdot C(u, v)$$

For the optional time-dependent extension, time is split into fixed buckets $b \in \{0, 1, \dots, B-1\}$ (30-minute intervals over operating horizon $H$). Each edge's factor is scaled by a fixed time-of-day multiplier $\mu(\cdot) \ge 1$, which is evaluated at the bucket midpoint $\tau_b$:
$$T(u, v, t) = T_0(u, v) \cdot C(u, v) \cdot \mu(\tau_{\text{bucket}(t)})$$
Here $\mu$ is a twin-peak commuter curve (a sum of two Gaussians for the morning and evening peaks). Its amplitudes are a generic urban calibration, not measured traffic counts.

For any two problem locations $i, j \in V$, Dijkstra's algorithm over $G$ gives the minimum travel time $t_{ij}$. The distance $d_{ij}$ is the length of that minimum-time path, which need not be the shortest path by distance.

---

## 2. Customer & Fleet Parameters

- **Customer demands**: Each customer $i \in C$ has a positive demand $q_i > 0$, with depot demand $q_0 = 0$.
- **Time windows**: Each customer $i \in C$ specifies a service time window $[e_i, l_i]$, where:
  - $e_i \ge 0$ is the earliest arrival time (ready time). If a vehicle arrives at $t < e_i$, it waits until $e_i$.
  - $l_i \ge e_i$ is the latest acceptable service start time (due time / SLA deadline).
- **Service duration**: Customer $i$ requires service duration $s_i \ge 0$ minutes. For depot, $s_0 = 0$.
- **Vehicle fleet**: up to $K$ vehicles, each with capacity $Q$. $K$ is a ceiling, not a quota: a vehicle may be left unused (but see $\rho$ below).
- **Mixed fleet (optional)**: two per-vehicle factors, both defaulting to $1.0$. With the defaults, the fleet is uniform and the objective reduces to the plain case:
  - $\sigma_k > 0$ is the **speed factor**. It multiplies vehicle $k$'s travel time on every leg, so values below $1.0$ mean faster than the network's base speed. It scales congested travel time rather than replacing it.
  - $\kappa_k > 0$ is the **relative cost per kilometre**, weighting the distance driven by vehicle $k$. Only the ratio between vehicles matters.
- **Fleet utilisation mode**: a flag $\rho \in \{0, 1\}$ (`require_all_vehicles`, default $0$). When $\rho = 1$, all $K$ vehicles must be dispatched; see the idle-vehicle penalty in §4.3.
- **Depot operating window**: The depot operates within $[e_0, l_0]$, where $l_0$ represents the planning horizon $H$.

---

## 3. Decision Variables & Chromosome Encoding

### Classical Binary Decision Variables
For completeness, the standard three-index vehicle-flow formulation defines:
$$x_{ijk} = \begin{cases} 1 & \text{if vehicle } k \text{ traverses arc } (i, j) \\ 0 & \text{otherwise} \end{cases} \quad \forall i, j \in V, k \in \{1, \dots, K\}$$
$$y_{ik} = \begin{cases} 1 & \text{if customer } i \text{ is served by vehicle } k \\ 0 & \text{otherwise} \end{cases} \quad \forall i \in C, k \in \{1, \dots, K\}$$
$$t_{ik} \ge 0 \quad \text{arrival time of vehicle } k \text{ at node } i$$

### Continuous Random-Key Encoding (for Metaheuristic Swarm)
Combinatorial permutations are mapped to continuous particle positions $X \in \mathbb{R}^N$ using a random-key representation:
$$X = [x_1, x_2, \dots, x_N], \quad x_i \in [0, K)$$
For customer $i \in \{1, \dots, N\}$:
- **Vehicle Assignment**: $k_i = \lfloor x_i \rfloor \in \{0, 1, \dots, K-1\}$
- **In-Route Visiting Priority**: $p_i = x_i - \lfloor x_i \rfloor \in [0, 1)$

Decoding sorts the customers assigned to vehicle $k$ in ascending order of their priorities $p_i$, constructing the tour:
$$R_k = (0, c_{k,1}, c_{k,2}, \dots, c_{k, m_k}, 0)$$

---

## 4. Objective Function & Penalty Formulation

This section states the function that is actually minimised. Every solver in this repository reaches the objective through the single implementation `evaluate_solution` in [`app/core/vrp_problem.py`](../app/core/vrp_problem.py): QPSO, GA, SA, standard PSO, the greedy baseline and the exact solver. They therefore all optimise the same model. [`tests/test_formulation_matches_code.py`](../tests/test_formulation_matches_code.py) implements the equations below independently and checks that they reproduce the code's fitness exactly.

### 4.1 Route clock

A route for vehicle $k$ is the ordered customer sequence $R_k = (c_{k,1}, \dots, c_{k,m_k})$, driven as $0 \to c_{k,1} \to \dots \to c_{k,m_k} \to 0$. The vehicle leaves the depot at $\tau_{k,0} = 0$. Writing $c_{k,0} = 0$ for the depot, for each $j \in \{1, \dots, m_k\}$:

$$\theta_{k,j} = \sigma_k \cdot t\big(c_{k,j-1},\, c_{k,j},\, \tau_{k,j-1}\big) \qquad \text{(travel time, priced at the moment of departure)}$$

$$a_{k,j} = \tau_{k,j-1} + \theta_{k,j} \qquad \text{(arrival)}$$

$$w_{k,j} = \max\big(0,\; e_{c_{k,j}} - a_{k,j}\big) \qquad \text{(wait, if early)}$$

$$\beta_{k,j} = \max\big(a_{k,j},\; e_{c_{k,j}}\big) = a_{k,j} + w_{k,j} \qquad \text{(service start)}$$

$$\tau_{k,j} = \beta_{k,j} + s_{c_{k,j}} \qquad \text{(departure, after service)}$$

The return leg is $\theta_{k,m_k+1} = \sigma_k \cdot t(c_{k,m_k}, 0, \tau_{k,m_k})$, with no wait and no service at the depot.

The travel time $t(\cdot,\cdot,\tau)$ is evaluated **at the departure time**. With time-dependence enabled, the lookup uses the travel-time matrix for bucket $b(\tau) = \min\big(\lfloor \tau / \Delta \rfloor,\, B-1\big)$, where $\Delta = 30$ minutes. The whole leg is priced at that bucket, even if the drive crosses into the next one. With time-dependence disabled (the default, and the setting for every result except the time-dependent demonstration), this reduces to the single static matrix $t_{ij}$.

### 4.2 Accumulated quantities

$$T_{\text{total}} = \sum_{k=1}^{K} \left[ \sum_{j=1}^{m_k} \big(\theta_{k,j} + w_{k,j}\big) \;+\; \theta_{k,m_k+1} \right]$$

$$D_{\text{total}} = \sum_{k=1}^{K} \kappa_k \left[ \sum_{j=1}^{m_k} d\big(c_{k,j-1}, c_{k,j}\big) \;+\; d\big(c_{k,m_k}, 0\big) \right]$$

- $T_{\text{total}}$ **includes waiting time**, not just driving. A vehicle that arrives early and waits for the window to open is charged for the wait. Service time is not included.
- $D_{\text{total}}$ is a **weighted** distance. For a uniform fleet ($\kappa_k = 1$) it is distance in kilometres. For a mixed fleet it is a relative cost.

### 4.3 Penalties

Constraints are handled as soft penalties, so the swarm can pass through infeasible regions of the search space instead of being blocked by them.

**Capacity**, per vehicle:

$$\mathcal{P}_{\text{cap}} = \sum_{k=1}^{K} \max\left(0,\; \sum_{j=1}^{m_k} q_{c_{k,j}} - Q \right)$$

**Time window.** Lateness is measured from the **service start** $\beta_{k,j}$, not from arrival. A vehicle that arrives early and waits is still late if the window opens only after the deadline has passed:

$$\mathcal{P}_{\text{time}} = \sum_{k=1}^{K} \sum_{j=1}^{m_k} \max\big(0,\; \beta_{k,j} - l_{c_{k,j}}\big) \;+\; 1000 \cdot \big|U\big|$$

Here $U$ is the set of legs whose travel time is infinite (the node is unreachable). Each such leg adds a flat $1000$, contributes no time or distance, and does not advance the vehicle's clock.

**Idle vehicle**, only when every vehicle must be dispatched ($\rho = 1$):

$$\mathcal{P}_{\text{idle}} = \rho \cdot \big|\{\, k : R_k = \varnothing \,\}\big|$$

### 4.4 Fitness

$$\boxed{\;\min \; \mathcal{F} \;=\; w_T \cdot T_{\text{total}} \;+\; w_D \cdot D_{\text{total}} \;+\; \lambda_{\text{cap}} \cdot \mathcal{P}_{\text{cap}} \;+\; \lambda_{\text{time}} \cdot \mathcal{P}_{\text{time}} \;+\; \lambda_{\text{idle}} \cdot \mathcal{P}_{\text{idle}}\;}$$

| symbol | meaning | default |
|---|---|---|
| $w_T$ | weight on fleet time | $0.6$ |
| $w_D$ | weight on distance driven | $0.4$ |
| $\lambda_{\text{cap}}$ | capacity violation multiplier | $50.0$ |
| $\lambda_{\text{time}}$ | lateness multiplier | $10.0$ |
| $\lambda_{\text{idle}}$ | idle-vehicle multiplier | $200.0$ |

The objective is a **weighted sum of two objectives**, fleet time and distance driven, not travel time alone. The weights belong to the problem instance, not to a solver. On the synthetic instances they are $w_T = 0.6$, $w_D = 0.4$. Solomon's CVRPTW benchmark is scored on total distance, so instances loaded from it set $w_T = 0$, $w_D = 1$, and the optimiser minimises the same quantity the benchmark reports.

A solution is reported **feasible** when both hard-constraint penalties are zero to numerical tolerance:

$$\mathcal{P}_{\text{cap}} < 10^{-6} \quad \text{and} \quad \mathcal{P}_{\text{time}} < 10^{-6}$$

$\mathcal{P}_{\text{idle}}$ does not affect feasibility: leaving a vehicle unused is a preference, not a constraint violation.

---

## 5. Exact Constraints (Reference MIP Formulation)

This MIP is given for reference only; no code in this repository solves it. The exact baseline in [`app/core/exact_vrp.py`](../app/core/exact_vrp.py) instead minimises the penalised objective $\mathcal{F}$ of §4 exactly: it enumerates every ordering of every customer subset, then assigns the subsets to vehicles by dynamic programming over subsets. "Optimum" in the results therefore means the minimum of $\mathcal{F}$, which coincides with the MIP optimum when the minimum of $\mathcal{F}$ is feasible.

1. **Routing and Single Visit**:
   $$\sum_{k=1}^K \sum_{j \in V, j \neq i} x_{ijk} = 1 \quad \forall i \in C$$
2. **Flow Conservation**:
   $$\sum_{j \in V, j \neq p} x_{jpk} - \sum_{j \in V, j \neq p} x_{pjk} = 0 \quad \forall p \in C, \forall k \in \{1, \dots, K\}$$
3. **Depot Departure and Return**:
   $$\sum_{j \in C} x_{0jk} \le 1, \quad \sum_{i \in C} x_{i0k} \le 1 \quad \forall k \in \{1, \dots, K\}$$
4. **Capacity Limits**:
   $$\sum_{i \in C} q_i \sum_{j \in V, j \neq i} x_{ijk} \le Q \quad \forall k \in \{1, \dots, K\}$$
5. **Time Window Precedence**:
   $$t_{ik} + s_i + t_{ij} - M(1 - x_{ijk}) \le t_{jk} \quad \forall i \in V, j \in C, i \neq j, \forall k$$
   $$e_i \le t_{ik} \le l_i \quad \forall i \in C, \forall k$$

---

## 6. Quantum-Behaved Particle Swarm Optimization (QPSO)

### Quantum Delta-Potential-Well Mechanics
In classical PSO, a particle moves along Newtonian trajectories with position $x$ and velocity $v$. In QPSO, particles exhibit quantum behavior bound by a delta potential well at the local attractor point $P$.

According to the Schrödinger equation for a particle in a 1D delta potential well:
$$\psi(y) = \frac{1}{\sqrt{L}} e^{-|y|/L}$$
where $L$ is the characteristic length of the potential well. The probability density function of position is:
$$Q(y) = |\psi(y)|^2 = \frac{1}{L} e^{-2|y|/L}$$

Using the inverse transform sampling method with uniform random variable $u \sim U(0, 1)$:
$$y = \pm \frac{L}{2} \ln\left(\frac{1}{u}\right)$$

### Algorithmic Update Rules
For particle $m \in \{1, \dots, M\}$ on dimension $j \in \{1, \dots, N\}$ at iteration $t$:

1. **Mean Best Position ($mbest$)**:
   The center of gravity of all individual personal best positions:
   $$mbest_j(t) = \frac{1}{M} \sum_{m=1}^M pbest_{m,j}(t)$$

2. **Local Attractor ($P$)**:
   A stochastic combination of personal best $pbest_m$ and global swarm best $gbest$:
   $$P_{m,j}(t) = \phi_{m,j}(t) \cdot pbest_{m,j}(t) + (1 - \phi_{m,j}(t)) \cdot gbest_j(t), \quad \phi_{m,j} \sim U(0, 1)$$

3. **Position Update**:
   $$X_{m,j}(t+1) = P_{m,j}(t) \pm \alpha(t) \cdot |mbest_j(t) - X_{m,j}(t)| \cdot \ln\left(\frac{1}{u}\right), \quad u \sim U(10^{-6}, 1)$$
   The sign $\pm$ is chosen with equal probability ($p = 0.5$). $\phi$, $u$ and the sign are drawn independently for every particle and dimension. The new position is clipped to the chromosome range $[0, K)$.

4. **Contraction-Expansion Coefficient ($\alpha$)**:
   Decreases linearly over the iterations, from exploration early in the run to exploitation late in it:
   $$\alpha(t) = \alpha_{\text{start}} - \frac{t}{T_{\max} - 1} (\alpha_{\text{start}} - \alpha_{\text{end}}), \quad t = 0, \dots, T_{\max} - 1$$
   Values used: $\alpha_{\text{start}} = 1.2$, $\alpha_{\text{end}} = 0.35$.

### High-Dimensional Instability Fix (Jump-Cap)
The excursion term $\ln(1/u)$ is unbounded as $u \to 0$. The chromosome has one gene per customer, so as $N$ grows, the chance that at least one gene draws a very large jump in a given iteration also grows. One such jump can reassign a customer and break an otherwise good route.

The jump is therefore capped, with a cap that tightens as $N$ grows:
$$\text{jump} = \min\left(\ln\left(\frac{1}{u}\right),\; c(N)\right), \qquad c(N) = \max\left(0.3,\; \frac{3}{1 + N/20}\right)$$
The constants $3$, $20$ and the floor $0.3$ were tuned by experiment on the CVRPTW benchmark; they are not derived from theory. The cap makes the search more stable in practice (see the scalability results) but does not come with a convergence guarantee. [`tests/test_qpso_jump_cap.py`](../tests/test_qpso_jump_cap.py) checks that the bound is applied.

---

## 7. Memetic Hybridization (Lamarckian Local Search)

QPSO's sampling explores globally. Local route improvements are better made by dedicated neighbourhood operators, so the global best is periodically refined by local search ([`app/core/local_search.py`](../app/core/local_search.py)).

Every $L = 15$ iterations (skipping iteration 0), the global best is decoded into routes and refined by alternating the two operators below. Refinement stops after 2 passes or at the first pass that brings no improvement. Both operators accept a move only if it lowers the full fitness $\mathcal{F}$ of §4, so penalties count, not just distance.
1. **2-opt (intra-route)**: reverses a segment $(i, \dots, j)$ of one vehicle's route. The first improving reversal found is accepted.
2. **Relocation (single-customer Or-opt, inter- and intra-route)**: tries moving each customer to every position in every route, including its own, and applies the best improving move for that customer. Only single customers are moved, not blocks of consecutive customers.

If refinement improves on $gbest$, the refined routes are encoded back into a chromosome and replace $gbest$. They also replace the position and personal best of the particle with the worst personal best (Lamarckian learning), so the swarm continues from the improved solution. One final refinement of 3 passes is applied to the best solution after the last iteration.

The benefit is measured, not assumed: on 40–60-customer instances, local search closes the gap between QPSO and standard PSO (see the README results). Instances of other sizes have not been evaluated to the same standard.

---

## 8. Discussion and Limitations: Classical Execution of a Quantum-Behaved Algorithm

We call the algorithm *quantum-behaved* PSO, following Sun et al.'s original QPSO terminology. The looser term *quantum-inspired* is a common synonym. Either way, no quantum computation takes place: the whole method runs on a classical CPU. The only quantum element is the sampling distribution. Each particle's next position is drawn from the probability density $Q(y) = |\psi(y)|^2$ of a particle in a 1D delta potential well (Section 6), using ordinary pseudo-random numbers. This heavy-tailed, attractor-centred distribution replaces PSO's velocity update. It is the source of the algorithm's exploration behaviour, and it also causes the dimension-dependent jump instability that the jump-cap corrects. Any gains reported here therefore come from a classical stochastic search operator and should be compared with other classical metaheuristics (GA, SA, standard PSO), not with quantum hardware. We make no claim of quantum speed-up.

The congestion model has two further limits. First, edge congestion factors are static or come from a fixed time-of-day curve; live traffic data is not ingested. Second, the fleet is homogeneous in capacity, and all experiments solve the problem once, from the depot, before departure.

**Future work.** The penalty formulation of Section 4 could be cast as a QUBO or Ising Hamiltonian and solved with QAOA or quantum annealing. Other extensions are driving the congestion factors from observed traffic data and supporting mid-route re-planning with per-vehicle residual capacities $Q_k$.
