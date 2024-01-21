import joblib
import numpy
import numpy as np
import pandas as pd
import torch
from torch import optim, nn

from Model import initialize
from Model.mutil_layer_prediction_model import RegressionModel
from Model.performance_analyze import calculate_weight


def P_objective(Operation, Problem, M, Input):
    [Output, Boundary, Coding] = P_DTLZ(Operation, Problem, M, Input)
    if Boundary == []:
        return Output
    else:
        return Output, Boundary, Coding


def P_DTLZ(Operation, Problem, M, Input):
    Boundary = []
    Coding = ""
    k = 1
    K = [5, 10, 10, 10, 10, 10, 20]
    K_select = K[k - 1]
    if Operation == "init":
        D = M + K_select - 1
        MaxValue = np.ones((1, D))
        MinValue = np.zeros((1, D))
        Population = np.random.random((Input, D))
        Population = np.multiply(Population, np.tile(MaxValue, (Input, 1))) + \
                     np.multiply((1 - Population), np.tile(MinValue, (Input, 1)))
        Boundary = np.vstack((MaxValue, MinValue))
        Coding = "Real"
        return Population, Boundary, Coding
    elif Operation == "value":
        Population = Input
        FunctionValue = np.zeros((Population.shape[0], M))
        if Problem == "DTLZ1":
            # g = 100*(K_select+np.sum( (Population[:, M-1:] - 0.5)**2 - np.cos(20*np.pi*(Population[:, M-1:] - 0.5)), axis=1, keepdims = True))
            g = 100 * (K_select + np.sum(
                (Population[:, M - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (Population[:, M - 1:] - 0.5)), axis=1))
            for i in range(M):
                FunctionValue[:, i] = 0.5 * np.multiply(np.prod(Population[:, :M - i - 1], axis=1), (1 + g))
                if i > 0:
                    FunctionValue[:, i] = np.multiply(FunctionValue[:, i], 1 - Population[:, M - i - 1])
        elif Problem == "DTLZ2":
            g = np.sum((Population[:, M - 1:] - 0.5) ** 2, axis=1)
            for i in range(M):
                FunctionValue[:, i] = (1 + g) * np.prod(np.cos(0.5 * np.pi * (Population[:, :M - i - 1])), axis=1)
                if i > 0:
                    FunctionValue[:, i] = np.multiply(FunctionValue[:, i],
                                                      np.sin(0.5 * np.pi * (Population[:, M - i - 1])))

        return FunctionValue, Boundary, Coding


def get_hlf_boundary():
    param_range = initialize.read_yaml_config('/Benchmark_Deploy_Tool/param_range.yaml')
    boundary = pd.DataFrame(columns=['Name', 'Lower', 'Upper'], index=range(17))
    boundary['Name'] = boundary['Name'].astype(str)
    boundary['Lower'] = boundary['Lower'].astype(float)
    boundary['Upper'] = boundary['Upper'].astype(float)
    idx = 0
    contained_col = ['peer_gossip_dialTimeout', 'peer_gossip_aliveTimeInterval',
                     'peer_deliveryclient_reConnectBackoffThreshold',
                     'peer_gossip_publishCertPeriod',
                     'peer_gossip_election_leaderElectionDuration', 'peer_keepalive_minInterval',
                     'peer_gossip_maxBlockCountToStore',
                     'peer_deliveryclient_connTimeout', 'peer_gossip_requestStateInfoInterval',
                     'peer_keepalive_client_timeout',
                     'peer_discovery_authCacheMaxSize', 'peer_discovery_authCachePurgeRetentionRatio',
                     'Orderer_BatchSize_PreferredMaxBytes', 'Orderer_BatchSize_MaxMessageCount',
                     'General_Authentication_TimeWindow',
                     'General_Keepalive_ServerInterval',
                     'Orderer_BatchSize_AbsoluteMaxBytes']
    # Order:6, Configtx:4, Peer:48
    for param_type in ['Peer', 'Orderer', 'Configtx']:
        for k, v in param_range['Parameters'][param_type].items():
            if k in contained_col:
                lower = v['lower']
                upper = v['upper']
                lower_value, unit = convert_to_number(str(lower))
                upper_value, unit = convert_to_number(str(upper))
                boundary.iloc[idx, 0] = k
                boundary.iloc[idx, 1] = lower_value
                boundary.iloc[idx, 2] = upper_value
                idx += 1
    # mask = boundary[:, 0] == 'peer_keepalive_minInterval'
    # print(boundary['peer_keepalive_minInterval'][0])
    return boundary


def convert_to_number(arg):
    if arg.endswith('ms') or arg.endswith('MB'):
        return int(arg[:-2]), arg[-2:]
    if arg.endswith('s') or arg.endswith('m'):
        return int(arg[:-1]), arg[-1:]
    return float(arg), ''


def model_predict_four_metric(input, model_name):
    if model_name == 'bpnn':
        predictions_combined = None
        for target_col in ['throughput', 'avg_latency', 'disc_write']:
            model = RegressionModel()
            model.load_state_dict(torch.load(f'../../Model/bpnn/bpnn_{target_col}.pth'))
            peer_config = input
            orderer_config = input
            metric = input
            bench_config = input
            output = model(peer_config, orderer_config, metric, bench_config)
            if predictions_combined is None:
                predictions_combined = output
            else:
                predictions_combined = np.column_stack((predictions_combined, output))
        return predictions_combined
    elif model_name == 'XGBoost':
        predictions_combined = None
        # create & modify & query & open & query & transfer
        payload_function = 'open'
        for target_col in ['throughput', 'avg_latency', 'disc_write']:
            model = joblib.load(f'../traditional_model/{model_name}/{target_col}_{payload_function}_best_model.pkl')
            prediction = model.predict(input)
            if predictions_combined is None:
                predictions_combined = prediction
            else:
                predictions_combined = np.column_stack((predictions_combined, prediction))
        return predictions_combined
