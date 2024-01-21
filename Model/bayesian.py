import time

import joblib
import numpy as np
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real

from Model.moopso.P_objective import get_hlf_boundary


def obj_fun(params):
    predictions_combined = None
    # model = RegressionModel()
    # model.load_state_dict(torch.load(f'../Model/bpnn/bpnn_throughput.pth'))
    # peer_config = params[:12]
    # orderer_config = params[12:]
    # output = model(peer_config, orderer_config, None)
    # create & modify & query & open & query & transfer
    payload_function = 'transfer'
    model_name = 'XGBoost'
    # 'throughput', 'avg_latency', 'disc_write'
    target_col = 'throughput'
    model = joblib.load(f'./traditional_model/{model_name}/{target_col}_{payload_function}_best_model.pkl')
    feature_input = np.array(params)
    feature_input = feature_input.reshape([1, 17])
    output = model.predict(feature_input)
    # if predictions_combined is None:
    #     predictions_combined = output
    # else:
    #     predictions_combined = np.column_stack((predictions_combined, output))
    # predictions_combined = np.array(output)
    return output[0]


def main():
    start_time = time.time()
    space = []
    boundary = get_hlf_boundary()
    for i in range(len(boundary)):
        space.append(Real(boundary.iloc[i, 1], boundary.iloc[i, 2], name=boundary.iloc[i, 0]))
    # for i in range(n_calls):
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
