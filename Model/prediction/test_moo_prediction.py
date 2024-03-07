import time

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from Model.prediction import engine


# 计算 MAPE
def calculate_mape(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.ravel()
    elif isinstance(y_true, np.ndarray):
        y_true = y_true.ravel()
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 计算 RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    df = pd.read_sql('dataset', con=engine)
    df = df[df['error_rate'] <= 10]
    # df = df[df['bench_config'].isin(['query'])]
    # encoder = OneHotEncoder(sparse=False, dtype=int)
    # 将 bench_config 列转换为字符串类型
    # df['bench_config'] = df['bench_config'].astype(str)
    # 对 bench_config 列进行独热编码
    # encoded_data = encoder.fit_transform(df[['bench_config']])
    # print("原始数据中的缺失值数量：", df[['bench_config']].isnull().sum())
    # print("编码后的数据形状：", encoded_data.shape)
    # # 获取独热编码后的特征列名称
    # feature_names = encoder.get_feature_names_out(['bench_config'])
    # # 将编码后的数据转为 DataFrame，并拼接到原始数据中
    # encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
    # df = pd.concat([df, encoded_df], axis=1)
    df = pd.get_dummies(df, columns=['bench_config'], dtype=int)
    metric = df[['throughput', 'avg_latency', 'error_rate', 'disc_write', 'gossip_state_commit_duration',
                 'broadcast_validate_duration',
                 'blockcutter_block_fill_duration', 'broadcast_enqueue_duration']]
    # 准备数据
    features = df.drop(columns=['id', 'performance_id', 'config_id', 'stage', 'config_id',
                                'broadcast_enqueue_duration', 'blockcutter_block_fill_duration',
                                'broadcast_validate_duration', 'gossip_state_commit_duration',
                                'SmartBFT_RequestBatchMaxCount', 'SmartBFT_RequestBatchMaxInterval',
                                'SmartBFT_RequestForwardTimeout',
                                'SmartBFT_RequestComplainTimeout', 'SmartBFT_RequestAutoRemoveTimeout',
                                'SmartBFT_ViewChangeResendInterval',
                                'SmartBFT_ViewChangeTimeout', 'SmartBFT_LeaderHeartbeatTimeout',
                                'SmartBFT_CollectTimeout',
                                'SmartBFT_IncomingMessageBufferSize', 'SmartBFT_RequestPoolSize',
                                'SmartBFT_LeaderHeartbeatCount',
                                'peer_gossip_pvtData_transientstoreMaxBlockRetention',
                                'peer_gossip_pvtData_pushAckTimeout', 'peer_gossip_pvtData_btlPullMargin',
                                'peer_gossip_pvtData_reconcileBatchSize',
                                'peer_gossip_pvtData_reconcileSleepInterval', 'throughput', 'avg_latency', 'error_rate',
                                'disc_write', 'bench_config_create',
                                'bench_config_modify',
                                'bench_config_open',
                                'bench_config_query',
                                'bench_config_transfer'
                                ])
    # features.columns = features.columns.astype(str)
    name = 'adaboost'
    model = None
    param_grid = {}
    if name == 'xgboost':
        model = XGBRegressor()
        param_grid = {
            'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
            'max_depth': [3, 4, 5, 9],
            'n_estimators': [150, 200],
            'subsample': [0.3, 0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.3, 0.5, 0.8],
        }
    elif name == 'svr':
        model = SVR()
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': [0.1, 0.2, 0.3],
            'C': [0.1, 1, 10],
        }
    elif name == 'adaboost':
        model = AdaBoostRegressor()
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.1, 0.5, 0.8, 1.2, 2],
        }
    elif name == 'kneighbors':
        model = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 40, 50],
        }
    elif name == 'rf':
        rfr = RandomForestRegressor()
    # throughput avg_latency
    target_col = 'throughput'
    target = metric[[target_col]]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    if name != 'adaboost' and name != 'xgboost':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    loss_name = 'mape'
    loss_func = None
    if loss_name == 'mape':
        loss_func = calculate_mape
    elif loss_name == 'r2':
        loss_func = r2_score
    elif loss_name == 'rmse':
        loss_func = calculate_rmse

    scoring = {'Error': make_scorer(loss_func)}

    start_time = time.time()
    grid = GridSearchCV(model, param_grid, scoring=scoring, refit='Error', cv=5)
    grid.fit(X_train, y_train.values)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{name} Model Train Time: {elapsed_time}")
    best_params = grid.best_params_
    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)
    loss = loss_func(y_test, predictions)

    print(f"Model: {name} Metric: {target_col}")
    print(f"Best parameters: {best_params}")
    print(f"{loss_name}: {loss}")
    print(predictions)


if __name__ == "__main__":
    main()
