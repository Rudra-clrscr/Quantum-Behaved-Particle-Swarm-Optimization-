from app.core.graph_model import generate_synthetic_city_graph
from app.core.vrp_problem import generate_synthetic_vrp
from app.core.qpso_vrp import QPSOVRPOptimizer
from app.core.classical_baselines_vrp import run_ga_vrp, run_sa_vrp, run_standard_pso_vrp, run_greedy_nn_vrp

net = generate_synthetic_city_graph(n_nodes=30, seed=42)
kw = dict(n_customers=12, depot=0, vehicle_capacity=80, seed=1)
td_problem = generate_synthetic_vrp(net, time_dependent=True, **kw)

aware = QPSOVRPOptimizer(td_problem, n_particles=30, max_iter=60, seed=1).optimize().best_solution
ga_td = run_ga_vrp(td_problem, pop_size=30, max_iter=60, seed=1).best_solution
sa_td = run_sa_vrp(td_problem, max_iter=60*20, seed=1).best_solution
pso_td = run_standard_pso_vrp(td_problem, n_particles=30, max_iter=60, seed=1).best_solution
greedy_td = run_greedy_nn_vrp(td_problem).best_solution

print(f"QPSO: {aware.total_time}")
print(f"GA: {ga_td.total_time}")
print(f"SA: {sa_td.total_time}")
print(f"PSO: {pso_td.total_time}")
print(f"Greedy: {greedy_td.total_time}")
