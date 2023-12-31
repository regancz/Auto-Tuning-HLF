import logging
import time

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

from Model import initialize, storage, param_identification, performance_analyze
from Model.mutil_layer_prediction_model import RegressionModel, CustomLoss, train_and_predict_with_metrics, \
    SLRegressionModel, MAELoss
from Model.performance_analyze import get_dataset_lasso, calculate_weight

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark-Deploy-Tool/config.yaml')
    mysql_connection, engine = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])

    # get_dataset_lasso(engine)

    # parameter_data, parameter_rows = storage.query_config_parameter_by_table(mysql_connection)
    # performance_data, performance_rows = storage.query_performance_metric_by_table(mysql_connection)
    # param_identification.lasso_test(parameter_rows, performance_rows)
    # selected_params, selected_feats = param_identification.feature_selection(parameter_data, performance_data,
    #                                                                          alpha=0.1, method='lasso',
    #                                                                          sort_method='feature_importance')
    # print("Selected Parameters:", selected_params)
    # print("Selected Features:", selected_feats)

    # storage.prepare_data(mysql_connection)
    # storage.update_resource_monitor(mysql_connection)
    # storage.calculate_error_rate(mysql_connection)
    # performance_analyze.aggregated_lasso_dataset(mysql_connection, engine)
    # performance_analyze.get_dataset_lasso(engine)

    # df = performance_analyze.get_performance_metric(mysql_connection)
    # df = df.drop('id', axis=1)
    # weight = performance_analyze.calculate_weight(df)
    # print(weight)
    # param_identification.lasso_test(parameter_rows, performance_rows)



    # train model
    df = pd.read_sql('dataset', con=engine)
    df = df[~df['bench_config'].isin(['query'])]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # xgboost
    # mse_xgb, r2_xgb, mse_svr, r2_svr = train_and_predict_with_metrics(peer_config, orderer_config, metric)
    # print(mse_xgb, r2_xgb, mse_svr, r2_svr)

    # nn
    model = SLRegressionModel()
    # custom_criterion = CustomLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # target_df = metric[['throughput', 'avg_latency', 'error_rate', 'disc_write']]

    # performance_df = df[['avg_latency', 'throughput', 'error_rate', 'disc_write']]
    # weight = calculate_weight(performance_df)

    # target_df = metric['throughput'] * weight['throughput'] + metric['avg_latency'] * weight['avg_latency'] + \
    #     metric['error_rate'] * weight['error_rate'] + metric['disc_write'] * weight['disc_write']
    # target_df = target_df.to_frame()
    target_df_prev = metric[['avg_latency']]

    # scaler = MinMaxScaler()
    # target_normalized = scaler.fit_transform(target_df)
    # target_tensor = torch.tensor(target_normalized).float()

    target_tensor = torch.tensor(target_df_prev.values).float()
    # target_tensor = torch.log1p(target_tensor)
    logging.basicConfig(filename='./log/slbpnn_avg_latency_rmse.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    threshold_loss = 100
    start_time = time.time()
    repeat_times = 0
    last_loss = 0
    for epoch in range(1000000000000000000):
        optimizer.zero_grad()
        output = model(peer_config, orderer_config, metric)
        # custom_loss = custom_criterion(output, target_tensor, loss_hidden1, loss_hidden2, loss_hidden3)
        criterion = nn.MSELoss()
        # criterion = MAELoss()
        # criterion = nn.L1Loss()
        # output_array = output.detach().numpy()
        # output_normalized = scaler.fit_transform(output_array)
        # output_tensor = torch.tensor(output_normalized).float()

        # mse_loss = criterion(torch.expm1(output), torch.expm1(target_tensor))
        loss = criterion(output, target_tensor)
        loss = torch.sqrt(loss)
        # loss = criterion(output, target_tensor)
        # total_loss = mse_loss + custom_loss
        # rmse_loss.backward()
        loss.backward()
        # mae_loss
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(
                # f"Epoch {epoch + 1}: MSE Loss: {mse_loss.item()}, Custom Loss: {custom_loss.item()}, Total Loss: {total_loss.item()}")
                f"Epoch {epoch + 1}: RMSE Loss: {loss.item()}")
        # if mse_loss.item() <= 20:
        #     torch.save(model.state_dict(), './bpnn/bpnn_param.pth')
        #     torch.save(model, './bpnn/bpnn_model.pth')
        #     break
        if last_loss == loss.item():
            repeat_times += 1
        last_loss = loss.item()
        if loss.item() <= 100:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logging.info(f'SLBPNN Epoch {epoch}: RMSE Loss: {loss.item()}')
            logging.info(f"SLBPNN Train Time: {elapsed_time}")
        if repeat_times == 5:
            break
