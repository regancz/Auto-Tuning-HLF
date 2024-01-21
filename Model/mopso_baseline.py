import joblib
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize

from Model.moopso.P_objective import get_hlf_boundary


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
        super().__init__(n_var=17, n_obj=1, xl=0.0, xu=1.0)
        boundary = get_hlf_boundary()
        self.xl = boundary['Lower'].values
        self.xu = boundary['Upper'].values

    def _evaluate(self, x, out, *args, **kwargs):
        # create & modify & query & open & query & transfer
        payload_function = 'create'
        model_name = 'XGBoost'
        # 'throughput', 'avg_latency', 'disc_write'
        for target_col in ['throughput']:
            predictions_combined = []
            # model = RegressionModel()
            # model.load_state_dict(torch.load(f'../Model/bpnn/bpnn_{payload_function}_{target_col}.pth'))
            model = joblib.load(f'./traditional_model/{model_name}/{target_col}_{payload_function}_best_model.pkl')
            # for col in range(0, len(x), 1):
            #     peer_config = x[col, :12]
            #     orderer_config = x[col, 12:]
            #     output = model.predict(x)
            #     # output = model(peer_config.T, orderer_config.T, None)
            #     # predictions_combined.append(output.item() if output.item() > 0 else -output.item())
            #     predictions_combined.append(output)
            # if target_col == 'throughput' or target_col == 'disc_write':
            #     # predictions_combined.append(output.item())
            #     predictions_combined.append(output)
            # else:
            #     # predictions_combined.append(output.item())
            #     predictions_combined.append(output)
            output = model.predict(x)
            out["F"] = np.array(output) * -1
            # if out["F"] is None:
            #     out["F"] = np.array(predictions_combined)
            # else:
            #     out["F"] = np.column_stack([out["F"], np.array(predictions_combined)])


class CustomSampling(Sampling):
    def __init__(self, initial_x=None):
        super().__init__()
        self.initial_x = initial_x

    def _do(self, problem, n_samples, **kwargs):
        if self.initial_x is not None:
            X = self.initial_x
            if len(X) != n_samples:
                raise ValueError("Length of initial_x should match n_samples.")
        else:
            X = np.random.random((n_samples, problem.n_var)).tolist()  # 使用随机生成的样本作为默认值

        # if problem.has_bounds():
        #     xl, xu = problem.bounds()
        #     assert np.all(xu >= xl)
        #     X = [[xl[i] + (xu[i] - xl[i]) * val for i, val in enumerate(sample)] for sample in X]

        return X


def main():
    problem = MyProblem()

    algorithm = NSGA2(pop_size=1)  # 使用NSGA-II
    # 你手动指定的初始 x 值（可以是一个列表、数组等）
    initial_x = []

    # 将初始 x 值转换为种群
    custom_sampling = CustomSampling(initial_x)

    # 创建遗传算法实例并传递初始种群
    # algorithm = GA(pop_size=100, sampling=FloatRandomSampling())

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 100),
                   seed=1,
                   verbose=True,
                   eliminate_duplicates=True,
                   return_least_infeasible=True)

    print("最优参数:", res.X)
    print("最优目标值:", res.F)


if __name__ == "__main__":
    main()
