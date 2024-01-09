import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from Model.mopso.P_objective import get_hlf_boundary, P_objective
from Model.mutil_layer_prediction_model import RegressionModel



def get_hlf_boundary_constraints():
    boundary = get_hlf_boundary()
    constraints = []
    for idx in range(len(boundary)):
        lower = boundary.iloc[idx, 1]
        upper = boundary.iloc[idx, 2]
        constraints.append((lower, upper))
    return constraints


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=17, n_obj=1, n_constr=17, xl=0.0, xu=1.0)
        boundary = get_hlf_boundary()
        self.xl = boundary['Lower'].values
        self.xu = boundary['Upper'].values

    def _evaluate(self, x, out, *args, **kwargs):
        predictions_combined = []
        model = RegressionModel()
        model.load_state_dict(torch.load('../Model/bpnn/bpnn_throughput.pth'))
        for col in range(0, len(x), 1):
            peer_config = x[col, :12]
            orderer_config = x[col, :5]
            output = model(peer_config.T, orderer_config.T, None)
            predictions_combined.append(output.item())
        out["F"] = predictions_combined
        # 获取约束条件的上下限值

        # 初始化约束条件的值为0数组
        # out["G"] = np.zeros(self.n_constr)
        # # 计算约束条件的值并存储在out["G"]中
        # for col in range(0, len(x), 1):
        #     for i, (lower, upper) in enumerate(constraints):
        #         diff1 = x[i, col] - upper
        #         diff2 = lower - x[i, col]
        #         out["G"][i] = max(diff1, diff2)


def main():
    problem = MyProblem()

    # algorithm = NSGA2(pop_size=1)  # 使用NSGA-II
    algorithm = GA(pop_size=100)  # 使用GA

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 20),
                   seed=42,
                   verbose=True)

    print("最优参数:", res.X)
    print("最优目标值:", res.F)


if __name__ == "__main__":
    main()
