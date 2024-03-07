import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from Model.prediction import engine


def main():
    df = pd.read_sql('dataset', con=engine)
    df = df[df['error_rate'] <= 50]
    cc_df = df[df['bench_config'].isin(['query'])]
    # df = df.drop(columns=['id', 'performance_id', 'config_id', 'stage', 'config_id',
    #                       'broadcast_enqueue_duration', 'blockcutter_block_fill_duration',
    #                       'broadcast_validate_duration', 'gossip_state_commit_duration',
    #                       'SmartBFT_RequestBatchMaxCount', 'SmartBFT_RequestBatchMaxInterval', 'SmartBFT_RequestForwardTimeout',
    #                       'SmartBFT_RequestComplainTimeout', 'SmartBFT_RequestAutoRemoveTimeout', 'SmartBFT_ViewChangeResendInterval',
    #                       'SmartBFT_ViewChangeTimeout', 'SmartBFT_LeaderHeartbeatTimeout', 'SmartBFT_CollectTimeout',
    #                       'SmartBFT_IncomingMessageBufferSize', 'SmartBFT_RequestPoolSize', 'SmartBFT_LeaderHeartbeatCount',
    #                       'peer_gossip_pvtData_transientstoreMaxBlockRetention',
    #                       'peer_gossip_pvtData_pushAckTimeout', 'peer_gossip_pvtData_btlPullMargin',
    #                       'peer_gossip_pvtData_reconcileBatchSize',
    #                       'peer_gossip_pvtData_reconcileSleepInterval',
    #                       ])
    cc_df = cc_df[['Orderer_BatchSize_PreferredMaxBytes', 'Orderer_BatchSize_MaxMessageCount', 'Orderer_BatchTimeout',
                   'peer_authentication_timewindow',
                   'peer_gossip_election_leaderElectionDuration', 'peer_discovery_authCacheMaxSize',
                   'Metrics_Statsd_WriteInterval', 'peer_gossip_election_leaderAliveThreshold',
                   'peer_deliveryclient_reconnectTotalTimeThreshold', 'avg_latency', 'throughput', 'error_rate',
                   'disc_write']]
    cc_df = cc_df.rename(columns={
        'Orderer_BatchSize_PreferredMaxBytes': 'PreferredMaxBytes',
        'Orderer_BatchSize_MaxMessageCount': 'MaxMessageCount',
        'Orderer_BatchTimeout': 'BatchTimeout',
        'peer_authentication_timewindow': 'Timewindow',
        'peer_gossip_election_leaderElectionDuration': 'ElectionDuration',
        'peer_discovery_authCacheMaxSize': 'CacheMaxSize',
        'Metrics_Statsd_WriteInterval': 'WriteInterval',
        'peer_gossip_election_leaderAliveThreshold': 'AliveThreshold',
        'peer_deliveryclient_reconnectTotalTimeThreshold': 'ReconnectThreshold',
        'avg_latency': 'Latency',
        'throughput': 'TPS',
        'error_rate': 'Error',
        'disc_write': 'RWPS'
    })

    # performance_df = cc_df[['avg_latency', 'throughput', 'error_rate', 'disc_write']]
    # 计算特征之间的相关性矩阵
    correlation_matrix = cc_df.corr()

    # 使用热力图可视化相关性矩阵
    plt.figure(figsize=(18, 14))
    ax = sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", mask=(correlation_matrix == 0))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # plt.title("Correlation Heatmap")
    plt.show()


def main1():
    df = pd.read_sql('dataset', con=engine)
    df = df[df['error_rate'] <= 50]
    # 'create', 'modify', 'open', 'query', 'transfer'
    # cc_df = df[df['bench_config'].isin(['open'])]
    cc_df = pd.get_dummies(df, columns=['bench_config'])
    # df = df.drop(columns=['id', 'performance_id', 'config_id', 'stage', 'config_id',
    #                       'broadcast_enqueue_duration', 'blockcutter_block_fill_duration',
    #                       'broadcast_validate_duration', 'gossip_state_commit_duration',
    #                       'SmartBFT_RequestBatchMaxCount', 'SmartBFT_RequestBatchMaxInterval', 'SmartBFT_RequestForwardTimeout',
    #                       'SmartBFT_RequestComplainTimeout', 'SmartBFT_RequestAutoRemoveTimeout', 'SmartBFT_ViewChangeResendInterval',
    #                       'SmartBFT_ViewChangeTimeout', 'SmartBFT_LeaderHeartbeatTimeout', 'SmartBFT_CollectTimeout',
    #                       'SmartBFT_IncomingMessageBufferSize', 'SmartBFT_RequestPoolSize', 'SmartBFT_LeaderHeartbeatCount',
    #                       'peer_gossip_pvtData_transientstoreMaxBlockRetention',
    #                       'peer_gossip_pvtData_pushAckTimeout', 'peer_gossip_pvtData_btlPullMargin',
    #                       'peer_gossip_pvtData_reconcileBatchSize',
    #                       'peer_gossip_pvtData_reconcileSleepInterval',
    #                       ])
    cc_df = cc_df[['Orderer_BatchSize_PreferredMaxBytes',
                   'Orderer_BatchSize_MaxMessageCount',
                   'Orderer_BatchTimeout',
                   'peer_authentication_timewindow',
                   'peer_gossip_election_leaderElectionDuration',
                   'peer_discovery_authCacheMaxSize',
                   'Metrics_Statsd_WriteInterval',
                   'peer_gossip_election_leaderAliveThreshold',
                   'peer_deliveryclient_reconnectTotalTimeThreshold',
                   'avg_latency',
                   'throughput',
                   'error_rate',
                   'disc_write',
                   'gossip_state_commit_duration',
                   'broadcast_validate_duration',
                   'blockcutter_block_fill_duration',
                   'broadcast_enqueue_duration',
                   'bench_config_create',
                   'bench_config_modify',
                   'bench_config_open',
                   'bench_config_query',
                   'bench_config_transfer']]
    # 'create', 'modify', 'open', 'query', 'transfer'
    cc_df = cc_df.rename(columns={
        'Orderer_BatchSize_PreferredMaxBytes': 'PreferredMaxBytes',
        'Orderer_BatchSize_MaxMessageCount': 'MaxMessageCount',
        'Orderer_BatchTimeout': 'BatchTimeout',
        'peer_authentication_timewindow': 'Timewindow',
        'peer_gossip_election_leaderElectionDuration': 'ElectionDuration',
        'peer_discovery_authCacheMaxSize': 'CacheMaxSize',
        'Metrics_Statsd_WriteInterval': 'WriteInterval',
        'peer_gossip_election_leaderAliveThreshold': 'AliveThreshold',
        'peer_deliveryclient_reconnectTotalTimeThreshold': 'ReconnectThreshold',
        'avg_latency': 'Latency',
        'throughput': 'TPS',
        'error_rate': 'Error',
        'disc_write': 'RWPS',
        'gossip_state_commit_duration': 'StateCommit',
        'broadcast_validate_duration': 'Validate',
        'blockcutter_block_fill_duration': 'BlockFill',
        'broadcast_enqueue_duration': 'Enqueue',
        'bench_config_create': 'create',
        'bench_config_modify': 'modify',
        'bench_config_open': 'open',
        'bench_config_query': 'query',
        'bench_config_transfer': 'transfer'
    })

    # performance_df = cc_df[['avg_latency', 'throughput', 'error_rate', 'disc_write']]
    # 计算特征之间的相关性矩阵
    correlation_matrix = cc_df.corr()

    # 使用热力图可视化相关性矩阵
    plt.figure(figsize=(18, 14))
    ax = sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", mask=(correlation_matrix == 0))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # plt.title("Correlation Heatmap")
    plt.show()


if __name__ == "__main__":
    main()
