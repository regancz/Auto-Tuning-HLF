import numpy as np
from matplotlib import pyplot as plt
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from Model.moo.algo import nsga2, agemoea, smsemoa, dnsga2
from Model.moo.problem import Fabric3ObjProblem


def main():
    problem = Fabric3ObjProblem()
    algorithm = nsga2
    termination = get_termination("n_gen", 10)
    res = minimize(problem,
                   algorithm,
                   termination,
                   # seed=1,
                   save_history=True,
                   verbose=True)
    X = res.X
    F = res.F
    rounded_F = np.round(F, decimals=2)

    # 将两个数组水平堆叠
    data_to_write = np.column_stack((np.round(X), abs(rounded_F)))
    with open("./pareto_in.txt", "ab") as file:
        np.savetxt(file, data_to_write, fmt='%1.2f')
    print(np.mean(F[:, 0]))
    print(np.mean(F[:, 1]))
    # xl, xu = problem.bounds()
    # plt.figure(figsize=(7, 5))
    # plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    # plt.xlim(xl[0], xu[0])
    # plt.ylim(xl[1], xu[1])
    # plt.xlabel("PreferredMaxBytes")
    # plt.ylabel("MaxMessageCount")
    # plt.show()

    # plt.figure(figsize=(7, 5))
    # plt.scatter(np.flip(np.abs(F[:, 0])), np.flip(F[:, 1]), s=30, facecolors='none', edgecolors='blue')
    # plt.xlabel("TPS")
    # plt.ylabel("Latency")
    # plt.show()


if __name__ == "__main__":
    main()
