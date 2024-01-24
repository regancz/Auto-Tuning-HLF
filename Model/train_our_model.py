import logging
import time

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from Model.initialize import read_yaml_config, mysql_connect
from Model.mutil_layer_prediction_model import RegressionModel, CustomLoss, train_and_predict_with_metrics
from Model.performance_analyze import get_dataset_lasso, calculate_weight

if __name__ == "__main__":
    configParameters = read_yaml_config('../Benchmark_Deploy_Tool/config.yaml')
    mysql_connection, engine = mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])
    df = pd.read_sql('dataset', con=engine)
    # create & modify & query & open & query & transfer
    payload_method = 'transfer'
    target_col = 'disc_write'
    df = df[df['bench_config'].isin([payload_method])]
    # bench_config = pd.get_dummies(df, columns=['bench_config'])[['bench_config_create', 'bench_config_modify', 'bench_config_open',
    #                                                              'bench_config_query', 'bench_config_transfer']].astype(int)
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
    # custom_criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    target_df_prev = metric[[target_col]]
    target_tensor = torch.tensor(target_df_prev.values).float()
    threshold_loss = 100
    start_time = time.time()
    repeat_times = 0
    last_loss = 0
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(peer_config, orderer_config, metric)
        # custom_loss = custom_criterion(output, target_tensor, loss_hidden1, loss_hidden2, loss_hidden3)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        # mse_loss = criterion(torch.expm1(output), torch.expm1(target_tensor))
        loss = criterion(output, target_tensor)
        # loss = torch.sqrt(loss)
        # total_loss = mse_loss + custom_loss
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(
                # f"Epoch {epoch + 1}: MSE Loss: {mse_loss.item()}, Custom Loss: {custom_loss.item()}, Total Loss: {total_loss.item()}")
                f"Epoch {epoch + 1}: MAE Loss: {loss.item()}")
        # if mse_loss.item() <= 20:
        #     torch.save(model.state_dict(), './bpnn/bpnn_param.pth')
        #     torch.save(model, './bpnn/bpnn_model.pth')
        #     break
        if last_loss == loss.item():
            repeat_times += 1
        last_loss = loss.item()
        # if loss.item() <= 100:
        #     end_time = time.time()
        #     elapsed_time = end_time - start_time
        #     print(f'proposed Model Epoch {epoch}: MAE Loss: {loss.item()}')
        #     logging.info(f"proposed Model Train Time: {elapsed_time}")
        if repeat_times == 5:
            torch.save(model.state_dict(), f'./bpnn/bpnn_{payload_method}_{target_col}.pth')
            # torch.save(model, f'./bpnn/bpnn_{target_col}.pth')
            break
    torch.save(model.state_dict(), f'./bpnn/bpnn_{payload_method}_{target_col}.pth')
