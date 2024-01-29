import time
import warnings

import joblib
import pandas as pd
import torch
from minio.commonconfig import SnowballObject
from sklearn.ensemble import AdaBoostRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import optim, nn
from xgboost import XGBRegressor

from CopilotFabric.server.api import engine
from CopilotFabric.server.service import logger, minio_client, model_config
from Model.initialize import read_yaml_config
from Model.mutil_layer_prediction_model import RegressionModel


def bpnn_predict_model():
    df = pd.read_sql(model_config['prediction']['bpnn']['table'], con=engine)
    for payload_method in ['create', 'modify', 'open', 'query', 'transfer']:
        for target_col in ['throughput', 'avg_latency', 'disc_write']:
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
                 'peer_discovery_authCachePurgeRetentionRatio', ]]
            orderer_config = df_curr[
                ['Orderer_BatchSize_PreferredMaxBytes',
                 'Orderer_BatchSize_MaxMessageCount',
                 'Orderer_General_Authentication_TimeWindow',
                 'Orderer_General_Keepalive_ServerInterval',
                 'Orderer_BatchSize_AbsoluteMaxBytes']]
            metric = df_curr[['throughput', 'avg_latency', 'error_rate', 'disc_write', 'gossip_state_commit_duration',
                              'broadcast_validate_duration',
                              'blockcutter_block_fill_duration', 'broadcast_enqueue_duration']]
            model = RegressionModel()
            # model = SLRegressionModel()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            target_df_prev = metric[[target_col]]
            target_tensor = torch.tensor(target_df_prev.values).float()
            start_time = time.time()
            repeat_times = 0
            last_loss = 0
            for epoch in range(101):
                optimizer.zero_grad()
                model.eval()
                output = model(peer_config, orderer_config, metric)
                criterion = nn.L1Loss()
                # mse_loss = criterion(torch.expm1(output), torch.expm1(target_tensor))
                loss = criterion(output, target_tensor)
                # loss = torch.sqrt(loss)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}: Loss: {loss.item()}")
                if last_loss == loss.item():
                    repeat_times += 1
                last_loss = loss.item()
                if repeat_times == 5:
                    torch.save(model.state_dict(),
                               f'F:/Project/PythonProject/Auto-Tuning-HLF/Model/model_dict/bpnn/bpnn_{payload_method}_{target_col}.pth')
                    minio_client.upload_snowball_objects(
                        "copilotfabric",
                        [
                            SnowballObject(f"/model_dict/bpnn/bpnn_{payload_method}_{target_col}",
                                           filename=f'F:/Project/PythonProject/Auto-Tuning-HLF/Model/model_dict/bpnn/bpnn_{payload_method}_{target_col}.pth'),
                        ],
                    )
                    break
            torch.save(model.state_dict(),
                       f'F:/Project/PythonProject/Auto-Tuning-HLF/Model/model_dict/bpnn/bpnn_{payload_method}_{target_col}.pth')
            minio_client.upload_snowball_objects(
                "copilotfabric",
                [
                    SnowballObject(f"/model_dict/bpnn/bpnn_{payload_method}_{target_col}",
                                   filename=f'F:/Project/PythonProject/Auto-Tuning-HLF/Model/model_dict/bpnn/bpnn_{payload_method}_{target_col}.pth'),
                ],
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"bpnn Model Train Time: {elapsed_time} Loss: {last_loss}")


def baseline_predict_model():
    for payload_function in ['create', 'modify', 'query', 'open', 'query', 'transfer']:
        df = pd.read_sql(model_config['prediction']['bpnn']['table'], con=engine)
        df = df.head(100)
        df = df[df['bench_config'].isin([payload_function])]
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
        for name in ['xgboost', 'svr', 'adaboost', 'kneighbors']:
            for target_col in ['throughput', 'avg_latency', 'disc_write']:
                target = metric[target_col]
                X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                scoring = {'Error': make_scorer(mean_absolute_error)}
                model = None
                param_grid = {}
                if name == 'xgboost':
                    model = XGBRegressor()
                    param_grid = {
                        # 'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
                        'max_depth': [3, 4, 5, 9],
                        'n_estimators': [150, 200],
                        'subsample': [0.3, 0.6, 0.8, 0.9, 1.0],
                        'colsample_bytree': [0.3, 0.5, 0.8],
                    }
                elif name == 'svr':
                    model = SVR()
                    param_grid = {
                        'kernel': ['linear', 'rbf', 'poly'],
                        # 'gamma': [0.1, 0.2, 0.3],
                        # 'C': [0.1, 1, 10],
                    }
                elif name == 'adaboost':
                    model = AdaBoostRegressor()
                    param_grid = {
                        'n_estimators': [50, 100, 150, 200],
                        # 'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
                    }
                elif name == 'kneighbors':
                    model = KNeighborsRegressor()
                    param_grid = {
                        'n_neighbors': [3, 5, 7, 9],
                        # 'weights': ['uniform', 'distance'],
                        # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        # 'leaf_size': [20, 40, 50],
                    }
                grid = GridSearchCV(model, param_grid, scoring=scoring, refit='Error', cv=5)
                grid.fit(X_train_scaled, y_train.values)
                best_params = grid.best_params_
                best_model = grid.best_estimator_
                predictions = best_model.predict(X_test_scaled)
                joblib.dump(best_model, f'F:/Project/PythonProject/Auto-Tuning-HLF/Model/model_dict/{name}/{name}_{payload_function}_{target_col}.pkl')
                mae = mean_absolute_error(y_test, predictions)
                logger.info(f"Model: {name} Metric: {target_col} Best parameters: {best_params} MAE: {mae}")
                print(f"Model: {name} Metric: {target_col} Best parameters: {best_params} MAE: {mae}")
                minio_client.upload_snowball_objects(
                    "copilotfabric",
                    [
                        SnowballObject(f"/model_dict/{name}/{name}_{payload_function}_{target_col}",
                                       filename=f'F:/Project/PythonProject/Auto-Tuning-HLF/Model/model_dict/{name}/{name}_{payload_function}_{target_col}.pkl')
                    ],
                )
