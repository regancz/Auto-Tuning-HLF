import warnings

import pandas as pd
import torch
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from Model import initialize, storage, param_identification, performance_analyze
from Model.mutil_layer_prediction_model import RegressionModel, CustomLoss, train_and_predict_with_metrics, \
    train_model_for_sampling
from Model.performance_analyze import get_dataset_lasso, calculate_weight

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark_Deploy_Tool/config.yaml')
    mysql_connection, engine = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])
    df = pd.read_sql('dataset', con=engine)
    df = df[~df['bench_config'].isin(['query'])]
    df = df[:400]
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    train_model_for_sampling(df)
