import warnings

import pandas as pd
import torch
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from Model import initialize, storage, param_identification, performance_analyze
from Model.mutil_layer_prediction_model import RegressionModel, CustomLoss, train_and_predict_with_metrics
from Model.performance_analyze import get_dataset_lasso, calculate_weight

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark-Deploy-Tool/config.yaml')
    mysql_connection, engine = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])
    # train model
    df = pd.read_sql('dataset', con=engine)
    df = df[~df['bench_config'].isin(['query'])]
    # bench_config = pd.get_dummies(df, columns=['bench_config'])[
    #     ['bench_config_create', 'bench_config_modify', 'bench_config_open',
    #      'bench_config_query', 'bench_config_transfer']].astype(int)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    performance_df = df[['avg_latency', 'throughput', 'disc_write']]
    weight = calculate_weight(performance_df)
    # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    #   y = column_or_1d(y, warn=True)
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    train_and_predict_with_metrics(peer_config, orderer_config, metric, weight)
