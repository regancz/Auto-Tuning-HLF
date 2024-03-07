import joblib
import numpy as np
import torch
from pymoo.core.problem import Problem

from Model.mopso.p_objective import get_hlf_boundary
from Model.prediction.mutil_layer_prediction_model import RegressionModel


class Fabric3ObjProblem(Problem):
    def __init__(self):
        boundary = get_hlf_boundary()
        max_ = boundary['Upper'].values
        min_ = boundary['Lower'].values
        super().__init__(n_var=17, n_obj=2, n_constr=2, xl=min_, xu=max_)

    def _evaluate(self, x, out, *args, **kwargs):
        model_name = 'bpnn'
        if model_name == 'bpnn':
            predictions_combined = None
            payload_function = 'open'
            g_tps = None
            g_latency = None
            model_name = 'bpnn'
            for target_col in ['throughput', 'avg_latency']:
                peer_config = x[:, :12]
                orderer_config = x[:, 12:]
                model = RegressionModel()
                model.load_state_dict(
                    torch.load(f'../../Model/model_dict/bpnn/moo_bpnn_{payload_function}_{target_col}.pth'))
                output = []
                # for i in range(len(input)):
                #     out = model(peer_config=peer_config[i, :], orderer_config=orderer_config[i, :], metric=input[i, :])
                #     output.append(out.item())
                output = model(peer_config=peer_config[:, :], orderer_config=orderer_config[:, :], metric=x[:, :])
                if target_col == 'throughput':
                    g_tps = output - 350
                    output = -output
                    # 350 is upper tps, must <= 0
                if target_col == 'avg_latency':
                    g_latency = -output
                if predictions_combined is None:
                    predictions_combined = output.detach().numpy()
                else:
                    predictions_combined = np.column_stack((predictions_combined, output.detach().numpy()))
            # # create & modify & query & open & query & transfer
            # # cc_name = 'open'
            # name = 'adaboost'
            # # 'throughput', 'avg_latency', 'disc_write'
            # model_throughput = joblib.load(
            #     f'../model_dict/{name}/moo_open_throughput_best_model.pkl')
            # throughput = model_throughput.predict(x)
            # model_avg_latency = joblib.load(
            #     f'../model_dict/{name}/moo_open_avg_latency_best_model.pkl')
            # avg_latency = model_avg_latency.predict(x)
            # model_disc_write = joblib.load(
            #     f'../model_dict/{name}/moo_open_disc_write_best_model.pkl')
            # disc_write = model_disc_write.predict(x)
            # old_min = 310
            # old_max = 312
            # new_min = 280
            # new_max = 312
            # tmp = model_throughput.predict(x+1231)
            # throughput = linear_mapping(throughput, old_min, old_max, new_min, new_max)
            out["F"] = predictions_combined
            out["G"] = np.column_stack([g_tps.detach().numpy(), g_latency.detach().numpy()])
        else:
            predictions_combined = None
            # create & modify & query & open & query & transfer
            payload_function = 'open'
            g_tps = None
            g_latency = None
            for target_col in ['throughput', 'avg_latency']:
                model = joblib.load(f'../../Model/model_dict/{model_name}/moo_open_{target_col}_best_model.pkl')
                prediction = model.predict(x)
                if target_col == 'throughput':
                    g_tps = prediction - 350
                    prediction = -prediction
                if target_col == 'avg_latency':
                    g_latency = -prediction
                if predictions_combined is None:
                    predictions_combined = prediction
                else:
                    predictions_combined = np.column_stack((predictions_combined, prediction))
            out["F"] = predictions_combined
            out["G"] = np.column_stack([g_tps, g_latency])


