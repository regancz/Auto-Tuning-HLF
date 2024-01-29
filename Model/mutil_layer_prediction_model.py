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
        return torch.mean(torch.abs((pred - true) / true))

    def backward(self, output, target):
        diff = torch.sign(output - target)
        return diff / output.size(0)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, hidden1, hidden2, hidden3):
        loss = torch.mean(hidden1 + hidden2 + hidden3)
        return loss


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.BatchNorm1d(12, affine=False),
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
            num_rows, num_columns = peer_config.shape
            peer_config_tensor = torch.tensor(peer_config, dtype=torch.float32).reshape([num_rows, num_columns])
        else:
            raise ValueError("Invalid type for peer_config. Should be list or DataFrame.")

        if isinstance(orderer_config, list):
            orderer_config_tensor = torch.tensor(orderer_config, dtype=torch.float32).reshape([1, 5])
        elif isinstance(orderer_config, pd.DataFrame):
            orderer_config_tensor = torch.tensor(orderer_config.values, dtype=torch.float32)
        elif isinstance(orderer_config, np.ndarray):
            num_rows, num_columns = orderer_config.shape
            orderer_config_tensor = torch.tensor(orderer_config, dtype=torch.float32).reshape([num_rows, num_columns])
        else:
            raise ValueError("Invalid type for orderer_config. Should be list or DataFrame.")
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
            nn.Linear(17, 3),
            nn.ReLU(),
        )
        self.output = nn.Linear(3, 1)

    # dim: 12, 5, 4 + 4
    def forward(self, peer_config_df, orderer_config_df, metric):
        peer_config_tensor = torch.tensor(peer_config_df.values).float()
        orderer_config_tensor = torch.tensor(orderer_config_df.values).float()
        combined_input1 = torch.cat((peer_config_tensor, orderer_config_tensor), dim=1)
        out_hidden1 = self.hidden1(combined_input1)
        output = self.output(out_hidden1)
        return output


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


def normalize_column(df, column_name, a, b):
    x_min = df[column_name].min()
    x_max = df[column_name].max()
    df.loc[:, column_name + '_normalized'] = ((df[column_name] - x_min) * (b - a)) / (x_max - x_min) + a
    return df
