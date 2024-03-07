import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim

from Model.prediction import engine


# 计算 MAPE
def calculate_mape(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100


# 计算 R2
def calculate_r2(y_true, y_pred):
    mean_y_true = torch.mean(y_true)
    total_variance = torch.sum((y_true - mean_y_true) ** 2)
    explained_variance = torch.sum((y_pred - mean_y_true) ** 2)
    r2 = explained_variance / total_variance
    return r2


# 计算 RMSE
def calculate_rmse(y_true, y_pred):
    return torch.sqrt(nn.functional.mse_loss(y_pred, y_true))


class CNN(nn.Module):
    def __init__(self, fc1_len):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1632, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个维度以适应Conv1d的输入要求
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        # self.fc1 = nn.Linear(fc1_len, 64)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    global loss
    df = pd.read_sql('dataset', con=engine)
    df = df[df['error_rate'] <= 10]
    # df = df[~df['bench_config'].isin(['query'])]
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
                                'disc_write', 'bench_config'
                                ])
    # features = pd.concat([peer_config, orderer_config], axis=1)
    # throughput avg_latency
    target_col = 'throughput'
    target = metric[[target_col]]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features.values, target.values, test_size=0.2, random_state=42)

    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 初始化模型和优化器
    model = CNN(fc1_len=len(X_train))
    loss_name = 'mape'
    loss_func = None
    if loss_name == 'mape':
        loss_func = calculate_mape
    elif loss_name == 'r2':
        loss_func = calculate_r2
    elif loss_name == 'rmse':
        loss_func = calculate_rmse
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 200
    batch_size = 32
    start_time = time.time()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_func(batch_y.unsqueeze(1), outputs)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()} Train Time: {time.time() - start_time}")
        # if loss.item() < 5:
        #     break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"cnn Model Train {target_col} Time: {elapsed_time} Loss: {loss.item()}")
    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predicted_throughput = predictions.squeeze().numpy()
        print(f'{loss_name} ', loss_func(y_test, predictions.squeeze()))

    # 打印预测结果
    # print(predicted_throughput)


if __name__ == "__main__":
    main()
