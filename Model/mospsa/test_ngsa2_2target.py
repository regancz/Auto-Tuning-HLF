import joblib
import numpy as np
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from Model.mopso.p_objective import get_hlf_boundary


def main():
    problem = FabricProblem()
    problemSingle = FabricProblemSingle()
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
    F = res.F

    xl, xu = problemSingle.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.xlabel("PreferredMaxBytes")
    plt.ylabel("MaxMessageCount")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(np.flip(np.abs(F[:, 0])), np.flip(F[:, 1]), s=30, facecolors='none', edgecolors='blue')
    plt.xlabel("Throughput")
    plt.ylabel("Latency")
    plt.show()


def mainLF():
    problem = FabricProblemLF()
    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 200)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    X = res.X
    F = res.F
    xl, xu = problem.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.xlabel("PreferredMaxBytes")
    plt.ylabel("MaxMessageCount")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(np.flip(np.abs(F[:, 0])), np.flip(F[:, 1]), s=30, facecolors='none', edgecolors='blue')
    plt.xlabel("Error Rate")
    plt.ylabel("Latency")
    plt.show()


def main1():
    problem = FabricProblem()
    problemSingle = FabricProblemSingle()

    # NSGA-II for Single Objective Optimization
    algorithm_single = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination_single = get_termination("n_gen", 40)
    history_single = []

    def callback_single(algorithm):
        history_single.append(algorithm.pop.get("X"))

    # Single Objective Optimization
    minimize(problemSingle,
             algorithm_single,
             termination_single,
             seed=1,
             callback=callback_single,
             save_history=True,
             verbose=True)

    # Top 40 solutions from Single Objective Optimization
    top_X_single = history_single[-40:]

    # NSGA-II for Multi-Objective Optimization
    algorithm_multi = NSGA2(
        pop_size=100,  # Increase population size for better diversity
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination_multi = get_termination("n_gen", 40)
    history_multi = []

    def callback_multi(algorithm):
        history_multi.append(algorithm.pop.get("X"))

    # Multi-Objective Optimization
    res_pareto = minimize(problem,
                          algorithm_multi,
                          termination_multi,
                          seed=1,
                          callback=callback_multi,
                          save_history=True,
                          verbose=True)

    # Pareto Front solutions
    X_pareto = res_pareto.X

    # Plotting on a single figure
    xl, xu = problemSingle.bounds()
    plt.figure(figsize=(10, 7))

    # Plot Top 40 solutions from Single Objective Optimization
    for X in top_X_single:
        plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')

    # Plot Pareto Front solutions from Multi-Objective Optimization
    plt.scatter(X_pareto[:, 0], X_pareto[:, 1], s=30, marker='+', label='Multi-Objective')
    plt.scatter([], [], s=30, facecolors='none', edgecolors='r',
                label='Single Objective')

    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.xlabel("PreferredMaxBytes")
    plt.ylabel("MaxMessageCount")
    plt.legend()
    plt.show()


def multiObj():
    problem = FabricProblem()
    problemSingle = FabricProblemSingle()
    problemThroughputBlockcutter = FabricProblemThroughputBlockcutter()
    problemThroughputRWPS = FabricProblemThroughputRWPS()
    algorithm_multi = NSGA2(
        pop_size=100,  # Increase population size for better diversity
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination_multi = get_termination("n_gen", 100)
    res_pareto = minimize(problemThroughputRWPS,
                          algorithm_multi,
                          termination_multi,
                          seed=1,
                          save_history=True,
                          verbose=True)
    F = res_pareto.F
    plt.figure(figsize=(10, 7))
    plt.scatter(np.abs(F[:, 0]), np.abs(F[:, 1]), s=30, facecolors='none', edgecolors='blue')
    plt.xlabel("Throughput")
    plt.ylabel("RWPS")
    plt.legend()
    plt.show()


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([-2, -2]),
                         xu=np.array([2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0] ** 2 + x[1] ** 2)
        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = - 20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


class FabricProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=np.array([1, 1]),
                         xu=np.array([20, 20]))

    def _evaluate(self, x, out, *args, **kwargs):
        # create & modify & query & open & query & transfer
        cc_name = 'open'
        model_name = 'SVR'
        # 'throughput', 'avg_latency', 'disc_write'
        model_throughput = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_throughput_2metric_model.pkl')
        throughput = model_throughput.predict(x)
        model_avg_latency = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_avg_latency_2metric_model.pkl')
        avg_latency = model_avg_latency.predict(x)
        old_min = 310
        old_max = 312
        new_min = 280
        new_max = 312
        throughput = linear_mapping(throughput, old_min, old_max, new_min, new_max)
        out["F"] = np.column_stack([-throughput, avg_latency])
        out["G"] = -avg_latency


class FabricProblemLF(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl=np.array([1, 1]),
                         xu=np.array([20, 20]))

    def _evaluate(self, x, out, *args, **kwargs):
        # create & modify & query & open & query & transfer
        cc_name = 'open'
        model_name = 'SVR'
        # 'throughput', 'avg_latency', 'disc_write' 'open_error_rate_2metric_model.pkl'
        model_throughput = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_error_rate_2metric_model.pkl')
        throughput = model_throughput.predict(x)
        model_avg_latency = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_avg_latency_2metric_model.pkl')
        avg_latency = model_avg_latency.predict(x)
        old_min = 3.2
        old_max = 4
        new_min = 3.2
        new_max = 36
        throughput = linear_mapping(throughput, old_min, old_max, new_min, new_max)
        out["F"] = np.column_stack([throughput, avg_latency])
        out["G"] = np.column_stack([-throughput, -avg_latency])


class FabricProblemThroughputBlockcutter(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=np.array([1, 1]),
                         xu=np.array([20, 20]))

    def _evaluate(self, x, out, *args, **kwargs):
        # create & modify & query & open & query & transfer
        cc_name = 'open'
        model_name = 'SVR'
        # 'throughput', 'avg_latency', 'disc_write'
        model_throughput = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_throughput_2metric_model.pkl')
        throughput = model_throughput.predict(x)
        model_avg_latency = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_avg_latency_2metric_model.pkl')
        avg_latency = model_avg_latency.predict(x)
        old_min = 310
        old_max = 312
        new_min = 280
        new_max = 312
        throughput = linear_mapping(throughput, old_min, old_max, new_min, new_max)
        out["F"] = np.column_stack([-throughput, avg_latency])
        out["G"] = -avg_latency


class FabricProblemThroughputRWPS(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, xl=np.array([1, 1]),
                         xu=np.array([20, 20]))

    def _evaluate(self, x, out, *args, **kwargs):
        # create & modify & query & open & query & transfer
        cc_name = 'open'
        model_name = 'SVR'
        # 'throughput', 'avg_latency', 'disc_write'
        model_throughput = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_avg_latency_2metric_model.pkl')
        throughput = model_throughput.predict(x)
        model_disc_write = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_disc_write_2metric_model.pkl')
        disc_write = model_disc_write.predict(x)
        # old_min = 310
        # old_max = 312
        # new_min = 280
        # new_max = 312
        # throughput = linear_mapping(throughput, old_min, old_max, new_min, new_max)
        out["F"] = np.column_stack([-throughput, -disc_write])
        out["G"] = np.column_stack([-throughput, -disc_write])


class FabricProblemSingle(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=1, xl=np.array([1, 1]),
                         xu=np.array([20, 20]))

    def _evaluate(self, x, out, *args, **kwargs):
        # create & modify & query & open & query & transfer
        cc_name = 'open'
        model_name = 'SVR'
        # 'throughput', 'avg_latency', 'disc_write'
        model_throughput = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_throughput_2metric_model.pkl')
        throughput = model_throughput.predict(x)
        model_avg_latency = joblib.load(
            f'../traditional_model/spsa/{model_name}/{cc_name}_avg_latency_2metric_model.pkl')
        avg_latency = model_avg_latency.predict(x)
        old_min = 310
        old_max = 312
        new_min = 280
        new_max = 312
        throughput = linear_mapping(throughput, old_min, old_max, new_min, new_max)
        out["F"] = np.column_stack([-(throughput + 10 * avg_latency)])
        out["G"] = -avg_latency


def linear_mapping(value, old_min, old_max, new_min, new_max):
    return ((value - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min


if __name__ == "__main__":
    mainLF()
