import numpy as np
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from Model.moo.test_ngsa2_2target import FabricProblemLF


def main():
    problem = FabricProblemLF()
    algorithm = RNSGA2(
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


if __name__ == "__main__":
    main()