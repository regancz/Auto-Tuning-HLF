import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch import nn, optim

from CopilotFabric.server.service import model_config
from Model.prediction import engine
from Model.prediction.mutil_layer_prediction_model import RegressionModel


# 定义学习曲线函数
def learning_curve(n_values, a, b, curve_type):
    if curve_type == 'log':
        return [a + b * np.log(n) for n in n_values]
    elif curve_type == 'power':
        return [a * n ** b for n in n_values]
    elif curve_type == 'exponential':
        return [a * b ** n for n in n_values]
    elif curve_type == 'weiss':
        return [a + (b * n) / (n + 1) for n in n_values]


# 定义计算n*的函数
def calculate_n_star(R, S, a, b, curve_type):
    if curve_type == 'log':
        return -(R * np.abs(S) * b) / 2
    elif curve_type == 'power':
        return (-2 / (R * np.abs(S) * a * b)) ** (1 / (b - 1))
    elif curve_type == 'exponential':
        return np.log(-(2 / (R * np.abs(S) * a * np.log(b))))
    elif curve_type == 'weiss':
        return np.sqrt((-R * np.abs(S) * b) / 2)


# 定义计算Pearson相关系数的函数
def compute_correlation(true_curve, predicted_curve):
    # 使用clip函数限制值的范围，防止溢出
    true_curve = np.clip(true_curve, -1e100, 1e100)
    predicted_curve = np.clip(predicted_curve, -1e100, 1e100)

    correlation, _ = pearsonr(true_curve, predicted_curve)
    return correlation


# 选择具有最大Pearson相关系数的候选函数作为最佳拟合函数
def calculate_sample_curve(n_values, true_curve, R=1, S=200):
    best_fit_function = None
    best_correlation = -1  # 初始化为-1
    best_curve_type = None  # 保存最佳拟合函数类型

    for curve_type in ['log', 'power', 'exponential', 'weiss']:
        # 计算最优的拟合参数 a 和 b
        a, b = fit_learning_curve(n_values, true_curve, curve_type)

        # 计算拟合函数
        err_n = learning_curve(n_values, a, b, curve_type)

        # 计算Pearson相关系数
        correlation = compute_correlation(true_curve, err_n)

        if correlation > best_correlation:
            best_correlation = correlation
            best_fit_function = err_n
            best_a = a
            best_b = b
            best_curve_type = curve_type

    # 打印最佳拟合函数类型
    print(f"Best Fit Function Type: {best_curve_type}")

    # 计算 n*
    n_star = calculate_n_star(best_a, best_b, R, np.abs(S), best_curve_type)
    return best_fit_function, n_star


# 定义拟合函数，使用最小二乘法来拟合学习曲线的系数 a 和 b
def fit_learning_curve(n_values, true_curve, curve_type):
    try:
        # 增加 maxfev 的值，例如增加到 1000
        params, covariance = curve_fit(lambda n, a, b: learning_curve(n, a, b, curve_type), n_values, true_curve, maxfev=400)
        a, b = params
        print('a: ', a, ' b: ', b)
        return a, b
    except RuntimeError:
        print(f"Optimal parameters not found for curve_type: {curve_type}")
        return 48, 84


def main():
    # 'create', 'modify', 'open', 'query', 'transfer'
    for payload_method in ['create', 'modify', 'open', 'query', 'transfer']:
        for target_col in ['throughput', 'avg_latency']:
            loss_list = []
            data_size_list = []
            for data_size in range(100, 600, 50):
                df = pd.read_sql(model_config['prediction']['bpnn']['table'], con=engine)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                df_curr = df[df['bench_config'].isin([payload_method])]
                peer_config = df_curr[
                    ['peer_gossip_dialTimeout',
                     'peer_gossip_aliveTimeInterval',
                     'peer_deliveryclient_reConnectBackoffThreshold',
                     'peer_gossip_publishCertPeriod',
                     'peer_gossip_election_leaderElectionDuration',
                     'peer_keepalive_minInterval',
                     'peer_gossip_maxBlockCountToStore',
                     'peer_deliveryclient_connTimeout',
                     'peer_gossip_requestStateInfoInterval',
                     'peer_keepalive_client_timeout',
                     'peer_discovery_authCacheMaxSize',
                     'peer_discovery_authCachePurgeRetentionRatio']]
                orderer_config = df_curr[
                    ['Orderer_BatchSize_PreferredMaxBytes',
                     'Orderer_BatchSize_MaxMessageCount',
                     'Orderer_General_Authentication_TimeWindow',
                     'Orderer_General_Keepalive_ServerInterval',
                     'Orderer_BatchSize_AbsoluteMaxBytes']]
                metric = df_curr[['throughput', 'avg_latency', 'error_rate', 'disc_write', 'gossip_state_commit_duration',
                                  'broadcast_validate_duration',
                                  'blockcutter_block_fill_duration', 'broadcast_enqueue_duration']]
                peer_config_train, peer_config_test = train_test_split(peer_config, test_size=0.2, random_state=42)
                # 划分 orderer_config
                orderer_config_train, orderer_config_test = train_test_split(orderer_config, test_size=0.2, random_state=42)
                # 划分 metric
                metric_train, metric_test = train_test_split(metric, test_size=0.2, random_state=42)

                peer_config_train = peer_config_train[: data_size]
                orderer_config_train = orderer_config_train[: data_size]
                metric_train = metric_train[: data_size]

                model = RegressionModel()
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                target_df_prev = metric_train[[target_col]]
                target_tensor = torch.tensor(target_df_prev.values).float()
                repeat_times = 0
                last_loss = 0
                for epoch in range(200):
                    optimizer.zero_grad()
                    model.eval()
                    output = model(peer_config_train, orderer_config_train, metric_train)
                    criterion = nn.L1Loss()
                    loss = criterion(output / target_tensor, target_tensor / target_tensor)
                    loss.backward()
                    optimizer.step()
                    if (epoch + 1) % 100 == 0:
                        pass
                        # print(f"{target_col} Epoch {epoch + 1}: Loss: {loss.item()}")
                    if last_loss == loss.item():
                        repeat_times += 1
                    last_loss = loss.item()
                    if repeat_times == 5:
                        break
                # 使用模型进行预测
                model.eval()
                with torch.no_grad():
                    predictions = model(peer_config_test, orderer_config_test, metric_test)
                    criterion = nn.L1Loss()
                    target_df_test = metric_test[[target_col]]
                    target_tensor_test = torch.tensor(target_df_test.values).float()
                    loss = criterion(predictions / target_tensor_test, target_tensor_test / target_tensor_test)
                    # print(f'{data_size} {target_col} Test Loss: ', loss)
                data_size_list.append(data_size)
                loss_list.append(loss)
            best_fit_function, n_star = calculate_sample_curve(data_size_list, loss_list)
            print(f'{payload_method} n*: ', n_star)
        # break


if __name__ == "__main__":
    main()
