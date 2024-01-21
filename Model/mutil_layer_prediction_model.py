import datetime
import logging
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from torch import optim
from xgboost import XGBRegressor
from tqdm import tqdm

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, pred, true):
        # calculate Error
        return torch.mean(torch.abs((pred - true) / true))

    def backward(self, output, target):
        # 计算损失关于输入的梯度
        diff = torch.sign(output - target)
        return diff / output.size(0)  # 返回平均梯度


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, hidden1, hidden2, hidden3):
        loss = torch.mean(hidden1 + hidden2 + hidden3)  # 自定义损失
        return loss


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Sequential(
            # nn.BatchNorm1d(12, affine=False),
            nn.Linear(12, 1),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )
        self.hidden2 = nn.Sequential(
            # nn.BatchNorm1d(6, affine=False),
            nn.Linear(6, 1),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )
        self.hidden3 = nn.Sequential(
            # nn.BatchNorm1d(14, affine=False),
            nn.Linear(14, 1),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )

        self.output = nn.Linear(3, 1)

    # dim: 12, 5, 4 + 4
    def forward(self, peer_config, orderer_config, metric):
        # extra_config_df = orderer_config_df[['Orderer_BatchSize_PreferredMaxBytes',
        #                                      'Orderer_BatchSize_MaxMessageCount',
        #                                      'Orderer_General_Authentication_TimeWindow']]
        if isinstance(peer_config, list):
            peer_config_tensor = torch.tensor(peer_config, dtype=torch.float32).reshape([1, 12])
        elif isinstance(peer_config, pd.DataFrame):
            peer_config_tensor = torch.tensor(peer_config.values, dtype=torch.float32)
        elif isinstance(peer_config, np.ndarray):
            peer_config_tensor = torch.tensor(peer_config, dtype=torch.float32).reshape([1, 12])
        else:
            raise ValueError("Invalid type for peer_config. Should be list or DataFrame.")

        if isinstance(orderer_config, list):
            orderer_config_tensor = torch.tensor(orderer_config, dtype=torch.float32).reshape([1, 5])
        elif isinstance(orderer_config, pd.DataFrame):
            orderer_config_tensor = torch.tensor(orderer_config.values, dtype=torch.float32)
        elif isinstance(orderer_config, np.ndarray):
            orderer_config_tensor = torch.tensor(orderer_config, dtype=torch.float32).reshape([1, 5])
        else:
            raise ValueError("Invalid type for orderer_config. Should be list or DataFrame.")
        # peer_config_tensor = torch.tensor(peer_config_df.values).float()
        # orderer_config_tensor = torch.tensor(orderer_config_df.values).float()
        # extra_config_tensor = torch.tensor(extra_config_df.values).float()
        # bench_config_tensor = torch.tensor(bench_config_df.values).float()

        # metric_df = metric[['gossip_state_commit_duration',
        #                     'broadcast_validate_duration',
        #                     'blockcutter_block_fill_duration',
        #                     'broadcast_enqueue_duration']]
        # minmax_scaler = MinMaxScaler()
        # metric_scaled = minmax_scaler.fit_transform(metric_df)
        # metric_tensor = torch.tensor(metric_scaled).float()

        # combined_input1 = torch.cat((peer_config_tensor), dim=1)
        out_hidden1 = self.hidden1(peer_config_tensor)
        combined_input2 = torch.cat((out_hidden1, orderer_config_tensor), dim=1)
        out_hidden2 = self.hidden2(combined_input2)
        combined_input3 = torch.cat((out_hidden1, out_hidden2, peer_config_tensor), dim=1)
        out_hidden3 = self.hidden3(combined_input3)
        combined_input4 = torch.cat((out_hidden1, out_hidden2, out_hidden3), dim=1)
        output = self.output(combined_input4)

        return output


class SLRegressionModel(nn.Module):
    def __init__(self):
        super(SLRegressionModel, self).__init__()
        self.hidden1 = nn.Sequential(
            # nn.BatchNorm1d(12, affine=False),
            nn.Linear(17, 3),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )
        # self.hidden2 = nn.Sequential(
        #     # nn.BatchNorm1d(6, affine=False),
        #     nn.Linear(11, 1),
        #     nn.ReLU(),
        #     # nn.Dropout(0.3),
        # )
        # self.hidden3 = nn.Sequential(
        #     # nn.BatchNorm1d(14, affine=False),
        #     nn.Linear(19, 1),
        #     nn.ReLU(),
        #     # nn.Dropout(0.3),
        # )

        self.output = nn.Linear(3, 1)

    # dim: 12, 5, 4 + 4
    def forward(self, peer_config_df, orderer_config_df, metric):
        extra_config_df = orderer_config_df[['Orderer_BatchSize_PreferredMaxBytes',
                                             'Orderer_BatchSize_MaxMessageCount',
                                             'Orderer_General_Authentication_TimeWindow']]
        # peer_config = torch.tensor(peer_config_df.values)
        # orderer_config = torch.tensor(orderer_config_df.values)
        # extra_config = torch.tensor(extra_config_df.values)
        peer_config_tensor = torch.tensor(peer_config_df.values).float()
        orderer_config_tensor = torch.tensor(orderer_config_df.values).float()
        extra_config_tensor = torch.tensor(extra_config_df.values).float()
        # bench_config_tensor = torch.tensor(bench_config_df.values).float()

        metric_df = metric[['gossip_state_commit_duration',
                            'broadcast_validate_duration',
                            'blockcutter_block_fill_duration',
                            'broadcast_enqueue_duration']]
        minmax_scaler = MinMaxScaler()
        metric_scaled = minmax_scaler.fit_transform(metric_df)
        metric_tensor = torch.tensor(metric_scaled).float()

        combined_input1 = torch.cat((peer_config_tensor, orderer_config_tensor), dim=1)
        out_hidden1 = self.hidden1(combined_input1)
        # relu(out_hidden1, inplace=True)
        # combined_input2 = torch.cat((bench_config_tensor, out_hidden1, orderer_config_tensor), dim=1)
        # out_hidden2 = self.hidden2(combined_input2)
        # combined_input3 = torch.cat((bench_config_tensor, out_hidden1, out_hidden2, peer_config_tensor), dim=1)
        # out_hidden3 = self.hidden3(combined_input3)
        # combined_input4 = torch.cat((bench_config_tensor, out_hidden1, out_hidden2, out_hidden3, metric_tensor), dim=1)
        output = self.output(out_hidden1)

        return output


def train_and_predict_with_metrics(peer_config, orderer_config, metric, weight, payload_function):
    # 选择特征和目标列
    # features = pd.concat([peer_config, orderer_config, metric[['gossip_state_commit_duration',
    #                                                            'broadcast_validate_duration',
    #                                                            'blockcutter_block_fill_duration',
    #                                                            'broadcast_enqueue_duration']]], axis=1)
    # features = pd.concat([peer_config, orderer_config], axis=1)
    features = orderer_config[['Orderer_BatchSize_PreferredMaxBytes', 'Orderer_BatchSize_MaxMessageCount']]
    for target_col in ['throughput', 'avg_latency', 'disc_write']:
        target = metric[target_col]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # y_train_scaler = StandardScaler()
        # y_train_scaled = y_train_scaler.fit_transform(y_train.values.reshape(-1, 1))
        scoring = {'MAE': make_scorer(mean_absolute_error),
                   'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))}
        models = [XGBRegressor(), SVR(), AdaBoostRegressor(), KNeighborsRegressor()]
        model_names = ['XGBoost', 'SVR', 'AdaBoost', 'KNeighbors']
        for model, name in zip(models, model_names):
            start_time = time.time()
            param_grid = {}
            if name == 'XGBoost':
                param_grid = {
                    # 每一步迭代的步长，很重要。太大了运行准确率不高，太小了运行速度慢。我们一般使用比默认值小一点，0.1左右就很好。
                    'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
                    # 我们常用3-10之间的数字
                    'max_depth': [3, 4, 5, 7, 9],
                    # 这是生成的最大树的数目，也是最大的迭代次数。
                    'n_estimators': [50, 100, 150, 200],
                    # 0.5-1，0.5代表平均采样，防止过拟合. 范围: (0,1]，注意不可取0
                    # 'subsample': [0.3, 0.6, 0.8, 0.9, 1.0],
                    # 用来控制每棵随机采样的列数的占比(每一列是一个特征) 典型值：0.5-1范围: (0,1]
                    # 'colsample_bytree': [0.3, 0.5, 0.8],
                    # 'gamma': [0, 0.1, 0.2],
                    # 'reg_lambda': [1, 1.5, 2],
                    # 'reg_alpha': [0, 0.1, 0.5],
                    # 'min_child_weight': [1, 3, 5],
                    # 'max_delta_step': [0, 1, 2],
                    # 'scale_pos_weight': [1, 2, 3],
                    # # DART booster 参数
                    # 'sample_type': ['uniform', 'weighted'],
                    # 'normalize_type': ['tree', 'forest'],
                    # 'rate_drop': [0.1, 0.2, 0.3],
                    # 'skip_drop': [0.1, 0.2, 0.3],
                    # # 交叉验证参数
                    # 'num_boost_round': [50, 100, 150],
                    # 'early_stopping_rounds': [5, 10, 15],
                }
            elif name == 'SVR':
                param_grid = {
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': [0.1, 0.2, 0.3],
                    'C': [0.1, 1, 10],
                    # 'epsilon': [0.1, 0.2, 0.3],
                    # 'degree': [2, 3, 4],
                    # 'tol': [0.001, 0.01, 0.1],
                    # 'shrinking': [True, False],
                }
            elif name == 'AdaBoost':
                param_grid = {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
                    # 'loss': ["linear", "square", "exponential"]
                    # 添加 AdaBoost 模型的参数定义
                }
            elif name == 'KNeighbors':
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    # 'metric': ['euclidean', 'manhattan', 'minkowski'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [20, 40, 50],
                    # 'p': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 当使用 Minkowski 距离时考虑的参数
                }
            grid = GridSearchCV(model, param_grid, scoring=scoring, refit='RMSE', cv=5)
            grid.fit(X_train_scaled, y_train.values)
            end_time = time.time()
            elapsed_time = end_time - start_time
            best_params = grid.best_params_
            best_model = grid.best_estimator_
            predictions = best_model.predict(X_test_scaled)
            # transform_predictions = y_train_scaler.inverse_transform(predictions.reshape(-1, 1))
            joblib.dump(best_model,
                        f'./traditional_model/{name}/{target_col}_{payload_function}_2metric_best_model.pkl')
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print(f"Model: {name} {target_col}")
            print(f"Best parameters: {best_params}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}")
            print(f"Train Time: {elapsed_time}")
            print("")
            # break
    return  # 返回其它结果（这里你可以补充其他返回结果）


def train_model_for_sampling(df):
    model_names = ['XGBoost', 'SVR', 'AdaBoost', 'KNeighbors']
    for name in model_names:
        if name == 'XGBoost':
            start_point = 400
            increment = 400
            end_point = 2001
            for data_size in range(start_point, end_point, increment):
                train_traditional_model_for_loop(df, data_size, name)
        elif name == 'SVR':
            start_point = 200
            increment = 200
            end_point = 2001
            for data_size in range(start_point, end_point, increment):
                train_traditional_model_for_loop(df, data_size, name)
        elif name == 'AdaBoost':
            start_point = 400
            increment = 400
            end_point = 2001
            for data_size in range(start_point, end_point, increment):
                train_traditional_model_for_loop(df, data_size, name)
        elif name == 'KNeighbors':
            train_traditional_model_for_loop(df, 1500, name)
            train_traditional_model_for_loop(df, 2000, name)
        else:
            break
    return


def train_traditional_model_for_loop(df, data_size, name):
    df = df[:data_size]
    peer_config = df[
        ['peer_keepalive_minInterval',
         'peer_keepalive_client_timeout',
         'peer_gossip_maxBlockCountToStore',
         'peer_gossip_requestStateInfoInterval',
         'peer_gossip_publishCertPeriod',
         'peer_gossip_dialTimeout',
         'peer_gossip_aliveTimeInterval',
         'peer_gossip_election_leaderElectionDuration',
         'peer_deliveryclient_connTimeout',
         'peer_deliveryclient_reConnectBackoffThreshold',
         'peer_discovery_authCacheMaxSize',
         'peer_discovery_authCachePurgeRetentionRatio']]
    orderer_config = df[
        ['Orderer_General_Authentication_TimeWindow',
         'Orderer_General_Keepalive_ServerInterval',
         'Orderer_BatchSize_MaxMessageCount',
         'Orderer_BatchSize_AbsoluteMaxBytes',
         'Orderer_BatchSize_PreferredMaxBytes']]
    metric = df[['throughput', 'avg_latency', 'error_rate', 'disc_write', 'gossip_state_commit_duration',
                 'broadcast_validate_duration',
                 'blockcutter_block_fill_duration', 'broadcast_enqueue_duration']]
    features = pd.concat([peer_config, orderer_config], axis=1)
    for target_col in ['throughput', 'avg_latency', 'disc_write']:
        target = metric[target_col]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scoring = {'Error': make_scorer(mean_absolute_error)}
        model = None
        param_grid = {}
        if name == 'XGBoost':
            model = XGBRegressor()
            param_grid = {
                'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
                'max_depth': [3, 4, 5, 7, 9],
                'n_estimators': [50, 100, 150, 200],
                'subsample': [0.3, 0.6, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.3, 0.5, 0.8],
            }
        elif name == 'SVR':
            model = SVR()
            param_grid = {
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': [0.1, 0.2, 0.3],
                'C': [0.1, 1, 10],
            }
        elif name == 'AdaBoost':
            model = AdaBoostRegressor()
            param_grid = {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
            }
        elif name == 'KNeighbors':
            model = KNeighborsRegressor()
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [20, 40, 50],
            }
        grid = GridSearchCV(model, param_grid, scoring=scoring, refit='Error', cv=5)
        grid.fit(X_train_scaled, y_train.values)
        best_params = grid.best_params_
        best_model = grid.best_estimator_
        predictions = best_model.predict(X_test_scaled)
        # joblib.dump(best_model, f'./traditional_model/{name}/{target_col}_best_model.pkl')
        mae = mean_absolute_error(y_test / y_test, predictions / y_test)
        total_cost = 2 * data_size + mae * 5000
        logger = logging.getLogger(f"{name}_{target_col}_{data_size}")
        file_handler = logging.FileHandler(f'./log/sampling/{name}_{target_col}_{data_size}.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Model: {name} Metric: {target_col} DataSize: {data_size}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Error: {mae}")
        logger.info(f"TotalCost: {total_cost}")
        print(f"Model: {name} Metric: {target_col} DataSize: {data_size}")
        print(f"Best parameters: {best_params}")
        print(f"Error: {mae}")
        print(f"TotalCost: {total_cost}")
        print("")


def train_bpnn_for_sampling(df, data_size, target_col):
    df = df[~df['bench_config'].isin(['query'])]
    df = df[:data_size]
    peer_config = df[
        ['peer_gossip_dialTimeout', 'peer_gossip_aliveTimeInterval', 'peer_deliveryclient_reConnectBackoffThreshold',
         'peer_gossip_publishCertPeriod',
         'peer_gossip_election_leaderElectionDuration', 'peer_keepalive_minInterval',
         'peer_gossip_maxBlockCountToStore',
         'peer_deliveryclient_connTimeout', 'peer_gossip_requestStateInfoInterval', 'peer_keepalive_client_timeout',
         'peer_discovery_authCacheMaxSize', 'peer_discovery_authCachePurgeRetentionRatio']]
    orderer_config = df[
        ['Orderer_BatchSize_PreferredMaxBytes', 'Orderer_BatchSize_MaxMessageCount',
         'Orderer_General_Authentication_TimeWindow',
         'Orderer_General_Keepalive_ServerInterval',
         'Orderer_BatchSize_AbsoluteMaxBytes']]
    metric = df[['throughput', 'avg_latency', 'error_rate', 'disc_write', 'gossip_state_commit_duration',
                 'broadcast_validate_duration',
                 'blockcutter_block_fill_duration', 'broadcast_enqueue_duration']]
    model = RegressionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    target_df_prev = metric[[target_col]]
    target_tensor = torch.tensor(target_df_prev.values).float()
    log_file = f'./log/sampling/bpnn/{target_col}_{data_size}_adam_test.log'
    logger = logging.getLogger(f'bpnn_{target_col}_{data_size}_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    logger.addHandler(file_handler)
    repeat_times = 0
    last_loss = 0
    optimizer.zero_grad()
    for epoch in range(5000):
        output = model(peer_config, orderer_config, metric)
        criterion = MAELoss()
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        total_cost = 2 * data_size + 5000 * loss.item()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if (epoch + 1) % 100 == 0:
            logger.info(
                f"{timestamp} Epoch {epoch + 1} Metric: {target_col} DataSize: {data_size} Error: {loss.item()} TotalCost: {total_cost}")
        if last_loss == loss.item():
            repeat_times += 1
        last_loss = loss.item()
        if repeat_times == 5:
            break


def train_boost_2metric(df, cc_name, name, model, param_grid):
    # df = df[df['bench_config'].isin([cc_name])]
    df = df[(df['blockcutter_block_fill_duration'] != 0) & ~df['blockcutter_block_fill_duration'].isna()]
    # 在DataFrame中对列进行归一化
    df = normalize_column(df.copy(), 'Orderer_BatchSize_PreferredMaxBytes', 1, 20)
    df = normalize_column(df.copy(), 'Orderer_BatchSize_MaxMessageCount', 1, 20)

    orderer_config = df[
        [
            'Orderer_BatchSize_PreferredMaxBytes_normalized',
            'Orderer_BatchSize_MaxMessageCount_normalized',
            # 'Orderer_BatchSize_PreferredMaxBytes',
            # 'Orderer_BatchSize_MaxMessageCount',
            # 'Orderer_General_Authentication_TimeWindow',
            # 'Orderer_General_Keepalive_ServerInterval',
            # 'Orderer_BatchSize_AbsoluteMaxBytes'
        ]]

    metric = df[['throughput', 'avg_latency', 'disc_write', 'error_rate', 'blockcutter_block_fill_duration']]
    # model = XGBRegressor()
    # param_grid = {
    #     'learning_rate': [2],
    #     'max_depth': [3],
    #     'n_estimators': [200],
    #     'subsample': [0.3],
    #     'colsample_bytree': [0.3],
    # }

    # model = SVR()
    # param_grid = {
    #     'kernel': ['linear', 'rbf', 'poly'],
    #     'gamma': [0.1, 0.2, 0.3],
    #     'C': [0.1, 1, 10],
    # }

    # {'colsample_bytree': 0.3, 'learning_rate': 2, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.3}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    log_prefix = 'boost_2metric'
    log_file = f"F:/Project/PythonProject/Auto-Tuning-HLF/Model/log/spsa/{log_prefix}_{timestamp}.log"

    logger = logging.getLogger(f'{log_prefix}_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    features = orderer_config
    for target_col in tqdm(['blockcutter_block_fill_duration']):
        X_train, X_test, y_train, y_test = train_test_split(features, metric[target_col], test_size=0.2, random_state=42)
        # x_scaler = MinMaxScaler()
        # X_train_scaled = x_scaler.fit_transform(X_train)
        # X_test_scaled = x_scaler.transform(X_test)

        # normalized_value = normalize_value(x_value, x_min_original, x_max_original, a_target, b_target)
        scoring = {'Error': make_scorer(mean_absolute_error)}
        # model = None
        # param_grid = {}
        grid = GridSearchCV(model, param_grid, scoring=scoring, refit='Error', cv=5, verbose=2)
        grid.fit(X_train, y_train.values)
        best_params = grid.best_params_
        best_model = grid.best_estimator_
        predictions = best_model.predict(X_test)
        # print(predictions)
        joblib.dump(best_model, f'./traditional_model/spsa/{name}/{cc_name}_{target_col}_2metric_model.pkl')
        # mae = mean_absolute_error(y_test / y_test, predictions / y_test)
        mae = mean_absolute_error(y_test, predictions)
        logger.info(f"Model: {name} Metric: {target_col}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Error: {mae}")


def train_model_for_spsa(df, cc_name):
    model_names = [
        # 'XGBoost',
        'SVR',
        # 'AdaBoost',
        # 'KNeighbors'
    ]
    for name in model_names:
        if name == 'XGBoost':
            model = XGBRegressor()
            param_grid = {
                'learning_rate': [0.1, 0.5, 1.2, 2],
                'max_depth': [2, 3, 4],
                'n_estimators': [150, 200],
                'subsample': [0.3, 0.6, 0.8, 1.0],
                'colsample_bytree': [0.3, 0.5, 0.8],
            }
            train_boost_2metric(df, cc_name=cc_name, model=model, param_grid=param_grid, name=name)
        elif name == 'SVR':
            model = SVR()
            param_grid = {
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': [0.1, 0.2, 0.3],
                'C': [0.1, 1, 10],
            }
            train_boost_2metric(df, cc_name=cc_name, model=model, param_grid=param_grid, name=name)
        elif name == 'AdaBoost':
            model = AdaBoostRegressor()
            param_grid = {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
            }
            train_boost_2metric(df, cc_name=cc_name, model=model, param_grid=param_grid, name=name)
        elif name == 'KNeighbors':
            model = KNeighborsRegressor()
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [20, 40, 50],
            }
            train_boost_2metric(df, cc_name=cc_name, model=model, param_grid=param_grid, name=name)
    return


def normalize_column(df, column_name, a, b):
    x_min = df[column_name].min()
    x_max = df[column_name].max()
    df.loc[:, column_name + '_normalized'] = ((df[column_name] - x_min) * (b - a)) / (x_max - x_min) + a
    return df
