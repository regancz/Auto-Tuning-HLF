import logging
import time

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from Model import initialize, storage, param_identification, performance_analyze
from Model.mutil_layer_prediction_model import RegressionModel, CustomLoss, train_and_predict_with_metrics, MAELoss, \
    train_bpnn_for_sampling
from Model.performance_analyze import get_dataset_lasso, calculate_weight

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark-Deploy-Tool/config.yaml')
    mysql_connection, engine = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])
    df = pd.read_sql('dataset', con=engine)
    for data_size in range(400, 2001, 400):
        for target_col in ['throughput', 'disc_write']:
            train_bpnn_for_sampling(df, data_size, target_col)
