import time

import pandas as pd
import torch
from torch import optim, nn

from Model.prediction import engine
from Model.prediction.mutil_layer_prediction_model import train_model_for_sampling, RegressionModel


def main():
    df = pd.read_sql('dataset', con=engine)
    # ['create', 'modify', 'query', 'open', 'query', 'transfer']
    payload_function = 'open'
    df = df[df['bench_config'].isin([payload_function])]
    df = df[df['error_rate'] <= 10]
    # bpnn_predict_model()
    train_model_for_sampling(df)


def bpnn_predict_model():
    df = pd.read_sql('dataset', con=engine)
    df = df[df['error_rate'] <= 10]
    # 'create', 'modify', 'open', 'query', 'transfer'
    for payload_method in ['transfer']:
        # for target_col in ['throughput', 'avg_latency', 'disc_write']:
        for target_col in ['disc_write']:
            # df_curr = df[df['bench_config'].isin([payload_method])]
            df_curr = df
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
                 'peer_discovery_authCachePurgeRetentionRatio']]
            orderer_config = df_curr[
                ['Orderer_General_Authentication_TimeWindow',
                 'Orderer_General_Keepalive_ServerInterval',
                 'Orderer_BatchSize_PreferredMaxBytes',
                 'Orderer_BatchSize_MaxMessageCount',
                 'Orderer_BatchSize_AbsoluteMaxBytes']]
            metric = df_curr[['throughput', 'avg_latency', 'error_rate', 'disc_write', 'gossip_state_commit_duration',
                              'broadcast_validate_duration',
                              'blockcutter_block_fill_duration', 'broadcast_enqueue_duration']]
            model = RegressionModel()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            target_df_prev = metric[[target_col]]
            target_tensor = torch.tensor(target_df_prev.values).float()
            repeat_times = 0
            last_loss = 0
            start_time = time.time()
            for epoch in range(2000000):
                optimizer.zero_grad()
                model.eval()
                output = model(peer_config, orderer_config, metric)
                criterion = nn.MSELoss()
                # mse_loss = criterion(torch.expm1(output), torch.expm1(target_tensor))
                loss = criterion(output, target_tensor)
                # loss = torch.sqrt(loss)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}: Loss: {loss.item()}")
                if last_loss == loss.item():
                    repeat_times += 1
                last_loss = loss.item()
                if repeat_times == 5:
                    # torch.save(model.state_dict(),
                    #            f'F:/Project/PythonProject/Auto-Tuning-HLF/Model/model_dict/bpnn/moo_bpnn_{payload_method}_{target_col}.pth')
                    break
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"bpnn Model Train {target_col} Time: {elapsed_time} Loss: {last_loss}")
            # torch.save(model.state_dict(),
            #            f'F:/Project/PythonProject/Auto-Tuning-HLF/Model/model_dict/bpnn/moo_bpnn_{payload_method}_{target_col}.pth')


if __name__ == "__main__":
    bpnn_predict_model()
