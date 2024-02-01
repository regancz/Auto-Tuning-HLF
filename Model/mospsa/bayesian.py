import time

import joblib
import numpy as np
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real

from Model.mopso.p_objective import get_hlf_boundary


count = 0

def obj_fun(params):
    # create & modify & query & open & query & transfer
    payload_function = 'transfer'
    model_name = 'XGBoost'
    # 'throughput', 'avg_latency', 'disc_write'
    target_col = 'throughput'
    # model = joblib.load(f'./traditional_model/{model_name}/{target_col}_{payload_function}_best_model.pkl')
    # feature_input = np.array(params)
    # feature_input = feature_input.reshape([1, 17])
    # output = model.predict(feature_input)
    # return output[0]
    # count += 1
    print(count)
    return count

def main():
    start_time = time.time()
    space = []
    boundary = get_hlf_boundary()
    for i in range(len(boundary)):
        space.append(Real(boundary.iloc[i, 1], boundary.iloc[i, 2], name=boundary.iloc[i, 0]))
    result = gp_minimize(func=obj_fun, dimensions=space, n_calls=30, random_state=1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Train Time: {elapsed_time}")
    print("最优参数:", result.x)
    print("最优目标值:", result.fun)
    plot_convergence(result)
    plt.show()


if __name__ == "__main__":
    main()
