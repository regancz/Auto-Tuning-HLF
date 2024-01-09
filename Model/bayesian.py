import numpy as np
import torch
from skopt import gp_minimize
from skopt.space import Real

from Model.mopso.P_objective import get_hlf_boundary
from Model.mutil_layer_prediction_model import RegressionModel


def obj_fun(params):
    predictions_combined = None
    model = RegressionModel()
    model.load_state_dict(torch.load(f'../Model/bpnn/bpnn_throughput.pth'))
    peer_config = params[:12]
    orderer_config = params[12:]
    output = model(peer_config, orderer_config, None)
    if predictions_combined is None:
        predictions_combined = output
    else:
        predictions_combined = np.column_stack((predictions_combined, output))
    return predictions_combined.item()


def main():
    space = []
    boundary = get_hlf_boundary()
    for i in range(len(boundary)):
        space.append(Real(boundary.iloc[i, 1], boundary.iloc[i, 2], name=boundary.iloc[i, 0]))
    result = gp_minimize(func=obj_fun, dimensions=space, n_calls=20, random_state=1)
    print("最优参数:", result.x)
    print("最优目标值:", result.fun)


if __name__ == "__main__":
    main()
