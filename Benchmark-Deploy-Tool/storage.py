import uuid
import paramiko
from bs4 import BeautifulSoup
import pymysql.cursors
from pymysql import IntegrityError


def get_metrics(mysql_connection, ssh_client, config_id, performance_id):
    try:
        stdin, stdout, stderr = ssh_client.exec_command(
            'cat /home/charles/Project/Blockchain/report.html')
        html_content = stdout.read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        summary_name = []
        summary_table = soup.find('table')
        for row in summary_table.find_all('tr'):
            cols = row.find_all('td')
            cols = [col.text.strip() for col in cols]
            if cols:
                summary_name.append(cols[0])
        performancePrefix = 'Performance metrics for '
        resourcePrefix = 'Resource utilization for '
        for idx, name in enumerate(summary_name):
            performance_col = extract_data_from_table(soup, performancePrefix, name)
            resource_col = extract_data_from_table(soup, resourcePrefix, name)
            insert_metric_to_database(performance_col, resource_col, mysql_connection, config_id, performance_id, idx)
    finally:
        print(f"Already finish storage")


def extract_data_from_table(soup, prefix, name):
    col = []
    table_h3 = soup.find('h3', text=prefix + name)
    if table_h3:
        table = table_h3.find_next('table')
        if table:
            for row in table.find_all('tr'):
                cols = row.find_all('td')
                cols = [col.text.strip() for col in cols]
                if cols:
                    col.append(cols[:])
    if prefix == 'Resource utilization for ':
        return col[1:]
    return col


def insert_metric_to_database(performance_col, resource_col, connection, config_id, performance_id, idx):
    # report.html contains NaN or -
    performance_col = [['0' if elem == '-' or elem == 'NaN' else elem for elem in sublist] for sublist in performance_col]
    resource_col = [['0' if elem == '-' or elem == 'NaN' else elem for elem in sublist] for sublist in resource_col]
    for col in performance_col:
        col.append(performance_id + '-' + str(idx))
        col.append(config_id)
    if resource_col:
        for col in resource_col:
            col.append(performance_id + '-' + str(idx))

    try:
        with connection.cursor() as cursor:
            performance_metric_sql = "INSERT INTO performance_metric (bench_config, succ, fail, send_rate, max_latency, min_latency, avg_latency, throughput, id, config_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            resource_monitor_sql = "INSERT INTO resource_monitor (metric, prometheus_query, name, value, performance_id) VALUES (%s, %s, %s, %s, %s)"
            for col in performance_col:
                connection.ping(reconnect=True)
                cursor.execute(performance_metric_sql, col)
            if resource_col:
                for col in resource_col:
                    connection.ping(reconnect=True)
                    cursor.execute(resource_monitor_sql, col)
    # except pymysql.Error as e:
    #     print("could not close connection error pymysql %d: %s" % (e.args[0], e.args[1]))

        connection.commit()
    finally:
        print("insert_metric_to_database done")
        # connection.close()


def insert_bft_config_to_database(connection, order_param, configtx_param, bft_configtx_param, peer_param, config_id):
    data = [val for param in (order_param, configtx_param, bft_configtx_param, peer_param) for val in param.values()]
    data.append(config_id)
    data = [str(elem).replace('s', '').replace('M', '').replace('B', '').replace('m', '').replace(' ', '').replace('K', '').strip() if isinstance(elem,str) else str(
            elem) for elem in data]
    try:
        with connection.cursor() as cursor:
            parameter_sql = "INSERT INTO config_parameter (Orderer_General_Authentication_TimeWindow, Orderer_General_Cluster_SendBufferSize, " \
                                   "Orderer_General_Keepalive_ServerInterval, Orderer_General_Keepalive_ServerMinInterval, " \
                                   "Orderer_General_Keepalive_ServerTimeout, Metrics_Statsd_WriteInterval, Orderer_BatchTimeout, " \
                                   "Orderer_BatchSize_MaxMessageCount, Orderer_BatchSize_AbsoluteMaxBytes, Orderer_BatchSize_PreferredMaxBytes, " \
                                   "SmartBFT_RequestBatchMaxCount, SmartBFT_RequestBatchMaxInterval, SmartBFT_RequestForwardTimeout, " \
                                   "SmartBFT_RequestComplainTimeout, SmartBFT_RequestAutoRemoveTimeout, SmartBFT_ViewChangeResendInterval, " \
                                   "SmartBFT_ViewChangeTimeout, SmartBFT_LeaderHeartbeatTimeout, SmartBFT_CollectTimeout, " \
                                   "SmartBFT_IncomingMessageBufferSize, SmartBFT_RequestPoolSize, SmartBFT_LeaderHeartbeatCount, " \
                                   "peer_keepalive_minInterval, peer_keepalive_client_interval, peer_keepalive_client_timeout, " \
                                   "peer_keepalive_deliveryClient_interval, peer_keepalive_deliveryClient_timeout, " \
                                   "peer_gossip_membershipTrackerInterval, peer_gossip_maxBlockCountToStore, " \
                                   "peer_gossip_maxPropagationBurstLatency, peer_gossip_maxPropagationBurstSize, " \
                                   "peer_gossip_propagateIterations, peer_gossip_propagatePeerNum, " \
                                   "peer_gossip_pullInterval, peer_gossip_pullPeerNum, " \
                                   "peer_gossip_requestStateInfoInterval, peer_gossip_publishStateInfoInterval, " \
                                   "peer_gossip_publishCertPeriod, peer_gossip_dialTimeout, " \
                                   "peer_gossip_connTimeout, peer_gossip_recvBuffSize, " \
                                   "peer_gossip_sendBuffSize, peer_gossip_digestWaitTime, " \
                                   "peer_gossip_requestWaitTime, peer_gossip_responseWaitTime, " \
                                   "peer_gossip_aliveTimeInterval, peer_gossip_aliveExpirationTimeout, " \
                                   "peer_gossip_reconnectInterval, peer_gossip_election_startupGracePeriod, " \
                                   "peer_gossip_election_membershipSampleInterval, peer_gossip_election_leaderAliveThreshold, " \
                                   "peer_gossip_election_leaderElectionDuration, peer_gossip_pvtData_pullRetryThreshold, " \
                                   "peer_gossip_pvtData_transientstoreMaxBlockRetention, " \
                                   "peer_gossip_pvtData_pushAckTimeout, peer_gossip_pvtData_btlPullMargin, " \
                                   "peer_gossip_pvtData_reconcileBatchSize, " \
                                   "peer_gossip_pvtData_reconcileSleepInterval, peer_gossip_state_checkInterval, " \
                                   "peer_gossip_state_responseTimeout, " \
                                   "peer_gossip_state_batchSize, peer_gossip_state_blockBufferSize, " \
                                   "peer_gossip_state_maxRetries, peer_authentication_timewindow, peer_client_connTimeout, " \
                                   "peer_deliveryclient_reconnectTotalTimeThreshold, peer_deliveryclient_connTimeout, " \
                                   "peer_deliveryclient_reConnectBackoffThreshold, peer_discovery_authCacheMaxSize, " \
                                   "peer_discovery_authCachePurgeRetentionRatio, id) VALUES " \
                                   "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                                   "%s, %s, %s, % s, % s, % s, % s, % s, % s, % s, % s, % s, % s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                                   "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            connection.ping(reconnect=True)
            cursor.execute(parameter_sql, data)
        connection.commit()
    except pymysql.Error as e:
        print("could not close connection error pymysql %d: %s" % (e.args[0], e.args[1]))
    finally:
        print("insert_metric_to_database done")
        # connection.close()


def insert_config_to_database(connection, order_param, configtx_param, peer_param, config_id):
    data = [val for param in (order_param, configtx_param, peer_param) for val in param.values()]
    data.append(config_id)
    data = [str(elem).replace('s', '').replace('M', '').replace('B', '').replace('m', '').replace(' ', '').replace('K', '').strip() if isinstance(elem,str) else str(
            elem) for elem in data]
    try:
        with connection.cursor() as cursor:
            parameter_sql = "INSERT INTO config_parameter (Orderer_General_Authentication_TimeWindow, Orderer_General_Cluster_SendBufferSize, " \
                                   "Orderer_General_Keepalive_ServerInterval, Orderer_General_Keepalive_ServerMinInterval, " \
                                   "Orderer_General_Keepalive_ServerTimeout, Metrics_Statsd_WriteInterval, Orderer_BatchTimeout, " \
                                   "Orderer_BatchSize_MaxMessageCount, Orderer_BatchSize_AbsoluteMaxBytes, Orderer_BatchSize_PreferredMaxBytes, " \
                                   "peer_keepalive_minInterval, peer_keepalive_client_interval, peer_keepalive_client_timeout, " \
                                   "peer_keepalive_deliveryClient_interval, peer_keepalive_deliveryClient_timeout, " \
                                   "peer_gossip_membershipTrackerInterval, peer_gossip_maxBlockCountToStore, " \
                                   "peer_gossip_maxPropagationBurstLatency, peer_gossip_maxPropagationBurstSize, " \
                                   "peer_gossip_propagateIterations, peer_gossip_propagatePeerNum, " \
                                   "peer_gossip_pullInterval, peer_gossip_pullPeerNum, " \
                                   "peer_gossip_requestStateInfoInterval, peer_gossip_publishStateInfoInterval, " \
                                   "peer_gossip_publishCertPeriod, peer_gossip_dialTimeout, " \
                                   "peer_gossip_connTimeout, peer_gossip_recvBuffSize, " \
                                   "peer_gossip_sendBuffSize, peer_gossip_digestWaitTime, " \
                                   "peer_gossip_requestWaitTime, peer_gossip_responseWaitTime, " \
                                   "peer_gossip_aliveTimeInterval, peer_gossip_aliveExpirationTimeout, " \
                                   "peer_gossip_reconnectInterval, peer_gossip_election_startupGracePeriod, " \
                                   "peer_gossip_election_membershipSampleInterval, peer_gossip_election_leaderAliveThreshold, " \
                                   "peer_gossip_election_leaderElectionDuration, peer_gossip_pvtData_pullRetryThreshold, " \
                                   "peer_gossip_pvtData_transientstoreMaxBlockRetention, " \
                                   "peer_gossip_pvtData_pushAckTimeout, peer_gossip_pvtData_btlPullMargin, " \
                                   "peer_gossip_pvtData_reconcileBatchSize, " \
                                   "peer_gossip_pvtData_reconcileSleepInterval, peer_gossip_state_checkInterval, " \
                                   "peer_gossip_state_responseTimeout, " \
                                   "peer_gossip_state_batchSize, peer_gossip_state_blockBufferSize, " \
                                   "peer_gossip_state_maxRetries, peer_authentication_timewindow, peer_client_connTimeout, " \
                                   "peer_deliveryclient_reconnectTotalTimeThreshold, peer_deliveryclient_connTimeout, " \
                                   "peer_deliveryclient_reConnectBackoffThreshold, peer_discovery_authCacheMaxSize, " \
                                   "peer_discovery_authCachePurgeRetentionRatio, id) VALUES " \
                                   "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                                   "%s, %s, %s, % s, % s, % s, % s, % s, % s, % s, % s, % s, % s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, " \
                                   "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            connection.ping(reconnect=True)
            cursor.execute(parameter_sql, data)
        connection.commit()
    except pymysql.Error as e:
        print("could not close connection error pymysql %d: %s" % (e.args[0], e.args[1]))
    finally:
        print("insert_metric_to_database done")
