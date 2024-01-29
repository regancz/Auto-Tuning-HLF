import numpy as np
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from CopilotFabric.server.service import logger
from Model.mopso import p_objective
from Model.mopso.Mopso import Mopso
from Model.mospsa.optimizer import spsa_maximize_optimize, get_hlf_boundary_2metirc
from Model.mospsa.test_ngsa2_2target import FabricProblem


def aspsa_maximize():
    boundary = get_hlf_boundary_2metirc()
    random_input = np.random.uniform(low=boundary['Lower'].values, high=boundary['Upper'].values)
    max_iter = 40
    best_params_optimize, objective_values_optimize, best_objective_optimize, param_ledger_preferred, param_ledger_count, metric_ledger = spsa_maximize_optimize(
        random_input,
        rho=1,
        num_iterations=max_iter,
        a=10, c=2, A=1,
        alpha=0.602,
        gamma=0.101,
        low=boundary[
            'Lower'].values,
        high=boundary[
            'Upper'].values)
    logger.info("aspsa_optimize best_params:", best_params_optimize)
    logger.info("aspsa_optimize metric:", best_objective_optimize)


def moaspsa_minimize():
    problem = FabricProblem()
    algorithm = NSGA2(
        pop_size=40,  # 设置更大的种群规模，以确保获得更多的解
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 40)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    X = res.X
    # X[-1:, 0], X[-1:, 1]


def mopso_maximize():
    logger.info("start to execute mopso job")
    particals = 200
    cycle_ = 10
    mesh_div = 10
    thresh = 300
    boundary = p_objective.get_hlf_boundary()
    max_ = boundary['Upper'].values
    min_ = boundary['Lower'].values
    mopso_ = Mopso(particals, max_, min_, thresh, mesh_div)
    pareto_in, pareto_fitness = mopso_.done(cycle_)
    logger.info("mopso job finish")
    data = {}
    for name, row in zip(boundary['Name'], np.round(pareto_in)[:, -1]):
        data[name] = row.tolist()
    # with open('./output.yaml', 'w') as yaml_file:
    #     yaml.dump(data, yaml_file, default_flow_style=False)
    logger.info("upload to nacos")
    return data
