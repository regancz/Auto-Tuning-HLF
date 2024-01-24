import warnings

import pandas as pd
import torch
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from Model import initialize, storage, param_identification, performance_analyze
from Model.mutil_layer_prediction_model import RegressionModel, CustomLoss, train_and_predict_with_metrics, \
    train_model_for_sampling, train_model_for_spsa
from Model.performance_analyze import get_dataset_lasso, calculate_weight

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark_Deploy_Tool/config.yaml')
    mysql_connection, engine = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])
    # df = pd.read_sql('dataset', con=engine)
    # df = df[~df['bench_config'].isin(['query'])]

    # config_df = pd.read_sql('SELECT id, Orderer_BatchSize_MaxMessageCount, Orderer_BatchSize_PreferredMaxBytes FROM config_parameter_spsa', con=engine)
    # performance_df = pd.read_sql(
    #     "SELECT config_id, throughput, bench_config, avg_latency, fail FROM performance_metric_spsa WHERE succ <> 0", con=engine)
    #
    # # 将两个表格通过config_id进行合并
    # merged_df = pd.merge(config_df, performance_df, left_on='id', right_on='config_id')
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    df = pd.read_sql('dataset_spsa', con=engine)
    cc_name = 'open'
    df = df[df['bench_config'].isin([cc_name])]
    df.drop(['bench_config'], axis=1, inplace=True)
    df_avg = df.groupby(['performance_id', 'config_id']).mean().reset_index()

    train_model_for_spsa(df_avg, cc_name=cc_name)
