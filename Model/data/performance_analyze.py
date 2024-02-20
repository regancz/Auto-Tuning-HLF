import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Model.data import param_identification


def plot_error_rate_vs_tps(mysql_connect):
    try:
        cursor = mysql_connect.cursor()
        cursor.execute("SELECT error_rate, throughput FROM performance_metric_copy2")
        rows = cursor.fetchall()

        # 将查询结果存储到 Pandas DataFrame 中
        data = pd.DataFrame(rows, columns=['error_rate', 'throughput'])
        print(data['error_rate'])
        # 绘制相关性图
        plt.figure(figsize=(8, 6))
        plt.scatter(data['error_rate'], data['throughput'])
        plt.xlabel('Error Rate')
        plt.ylabel('TPS')
        plt.title('Correlation between Error Rate and TPS')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 关闭游标
        cursor.close()


def plot_tps_by_custom_id(mysql_connect):
    try:
        cursor = mysql_connect.cursor()
        cursor.execute("SELECT throughput FROM performance_metric_copy2 ")  # 请替换为你的表名
        rows = cursor.fetchall()

        # 创建一个自定义的 id 序列（从 1 开始）
        custom_id = list(range(1, len(rows) + 1))

        # 将查询结果存储到 Pandas DataFrame 中，并将自定义的 id 合并
        data = pd.DataFrame(rows, columns=['throughput'])
        data = data.sort_values(by='throughput')
        data['id'] = custom_id

        # 按 tps 排序
        data = data.sort_values(by='id')

        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.bar(data['id'], data['throughput'])
        plt.xlabel('ID')
        plt.ylabel('TPS')
        plt.title('TPS by Custom ID')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 关闭游标
        cursor.close()


def aggregate_monitor_metric(mysql_connection):
    try:
        cursor = mysql_connection.cursor()
        # 按metric属性分组，再在每个分组内，针对name属性以order, peer开头的行计算平均值
        query = ("""
            INSERT INTO resource_monitor_aggregated_spsa (performance_id, metric, prometheus_query, name, avg_value, stage)
            SELECT performance_id, metric, prometheus_query, 'order' AS name, AVG(value), stage
            FROM resource_monitor_spsa
            WHERE name LIKE 'order%'
            GROUP BY metric, performance_id

            UNION ALL

            SELECT performance_id, metric, prometheus_query, 'peer' AS name, AVG(value), stage
            FROM resource_monitor_spsa
            WHERE name LIKE 'peer%'
            GROUP BY metric, performance_id
        """)
        cursor.execute(query)
        mysql_connection.commit()

        print("Data aggregated and written to resource_monitor_copy2 successfully.")

    # except Exception as e:
    #     print(f"Error: {e}")

    finally:
        # 关闭游标
        cursor.close()


def get_performance_metric(mysql_connect):
    try:
        cursor = mysql_connect.cursor()
        cursor.execute(
            "SELECT id, avg_latency, throughput, error_rate FROM performance_metric_copy2 WHERE stage IS NULL AND bench_config <> 'query'")
        rows = cursor.fetchall()
        metric = pd.DataFrame(rows, columns=['id', 'avg_latency', 'throughput', 'error_rate'])
        query = "SELECT performance_id, avg_value FROM resource_monitor_copy2_aggregated WHERE performance_id IN %s AND metric = 'Disc Write (MB)' AND stage IS NULL"
        performance_ids = metric['id'].tolist()
        cursor.execute(query, (performance_ids,))
        disc_write_rows = cursor.fetchall()
        disc_write = pd.DataFrame(disc_write_rows, columns=['performance_id', 'avg_value'])
        disc_write.columns = ['id', 'disc_write']
        df = pd.merge(metric, disc_write, how='left', on='id')
        mysql_connect.commit()

        print("Data aggregated and written to resource_monitor_copy2 successfully.")
        return df

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 关闭游标
        cursor.close()


def aggregated_lasso_dataset(mysql_connect, engine):
    try:
        cursor = mysql_connect.cursor()
        # cursor fetch不会保留前一次的结果
        # 查询三个表进行聚合
        cursor.execute(
            "SELECT * FROM config_parameter_spsa WHERE peer_discovery_authCachePurgeRetentionRatio < 1")
        config_rows = cursor.fetchall()
        config = pd.DataFrame(config_rows)
        config.rename(columns={'id': 'config_id'}, inplace=True)
        cursor.execute(
            "SELECT id, config_id, avg_latency, throughput, bench_config, error_rate FROM performance_metric_spsa WHERE stage IS NULL")
        performance_rows = cursor.fetchall()
        performance = pd.DataFrame(performance_rows,
                                   columns=['id', 'config_id', 'avg_latency', 'throughput', 'bench_config',
                                            'error_rate'])
        performance.rename(columns={'id': 'performance_id'}, inplace=True)
        dataset = pd.merge(config, performance, how='inner', on='config_id')

        cursor.execute(
            "SELECT performance_id, avg_value FROM resource_monitor_aggregated_spsa WHERE metric = 'Disc Write (MB)' AND stage IS NULL")
        disc_write_rows = cursor.fetchall()
        disc_write = pd.DataFrame(disc_write_rows,
                                  columns=['performance_id', 'avg_value'])
        disc_write.rename(columns={'avg_value': 'disc_write'}, inplace=True)

        cursor.execute(
            "SELECT performance_id, avg_value FROM resource_monitor_aggregated_spsa WHERE metric = 'gossip_state_commit_duration' AND stage IS NULL")
        gossip_rows = cursor.fetchall()
        gossip = pd.DataFrame(gossip_rows,
                              columns=['performance_id', 'avg_value'])
        gossip.rename(columns={'avg_value': 'gossip_state_commit_duration'}, inplace=True)

        cursor.execute(
            "SELECT performance_id, avg_value FROM resource_monitor_aggregated_spsa WHERE metric = 'broadcast_validate_duration' AND stage IS NULL")
        broadcast_validate_duration_rows = cursor.fetchall()
        broadcast_validate_duration = pd.DataFrame(broadcast_validate_duration_rows,
                                                   columns=['performance_id', 'avg_value'])
        broadcast_validate_duration.rename(columns={'avg_value': 'broadcast_validate_duration'}, inplace=True)

        cursor.execute(
            "SELECT performance_id, avg_value FROM resource_monitor_aggregated_spsa WHERE metric = 'blockcutter_block_fill_duration' AND stage IS NULL")
        blockcutter_rows = cursor.fetchall()
        blockcutter = pd.DataFrame(blockcutter_rows,
                                   columns=['performance_id', 'avg_value'])
        blockcutter.rename(columns={'avg_value': 'blockcutter_block_fill_duration'}, inplace=True)

        cursor.execute(
            "SELECT performance_id, avg_value FROM resource_monitor_aggregated_spsa WHERE metric = 'broadcast_enqueue_duration' AND stage IS NULL")
        broadcast_enqueue_duration_rows = cursor.fetchall()
        broadcast_enqueue_duration = pd.DataFrame(broadcast_enqueue_duration_rows,
                                                  columns=['performance_id', 'avg_value'])
        broadcast_enqueue_duration.rename(columns={'avg_value': 'broadcast_enqueue_duration'}, inplace=True)

        dataset = pd.merge(dataset, disc_write, how='left', on='performance_id')
        dataset = pd.merge(dataset, gossip, how='left', on='performance_id')
        dataset = pd.merge(dataset, broadcast_validate_duration, how='left', on='performance_id')
        dataset = pd.merge(dataset, blockcutter, how='left', on='performance_id')
        dataset = pd.merge(dataset, broadcast_enqueue_duration, how='left', on='performance_id')
        dataset.rename(columns={'id': 'config_id'}, inplace=True)
        # dataset.drop(columns=['performance_id'])
        dataset.dropna(subset=['config_id', 'performance_id'], inplace=True)
        dataset.to_sql('dataset_spsa', con=engine, if_exists='append', index=False)
        mysql_connect.commit()
        print("Data aggregated and written to dataset_copy1 successfully.")
        # return dataset

    # except Exception as e:
    #     print(f"Error: {e}")

    finally:
        # 关闭游标
        cursor.close()


def calculate_weight(df):
    # 删除所有包含 NaN 值的行
    df_cleaned = df.dropna()
    # 校验 df_cleaned 是否为空
    if df_cleaned.empty:
        raise ValueError("DataFrame contains only NaN values or is empty")
    # 数据归一化
    df_normalized = (df_cleaned - df_cleaned.min()) / (df_cleaned.max() - df_cleaned.min())
    # 确保数据不含零值，避免除零错误
    df_normalized = df_normalized.replace(0, np.finfo(float).eps)
    # 计算信息熵
    entropy = (-1) * (1 / np.log(len(df_cleaned))) * ((df_normalized * np.log(df_normalized)).sum())
    # 计算权重
    weight = (1 - entropy) / sum(1 - entropy)
    return weight


def get_dataset_lasso(engine):
    # dataset_spsa
    df = pd.read_sql('dataset', con=engine)
    df = df[df['error_rate'] <= 10]
    # 'broadcast_enqueue_duration', 'blockcutter_block_fill_duration', 'broadcast_validate_duration', 'gossip_state_commit_duration'
    df = df.drop(columns=['id', 'performance_id', 'config_id', 'stage', 'error_rate', 'config_id', 'broadcast_enqueue_duration', 'blockcutter_block_fill_duration', 'broadcast_validate_duration', 'gossip_state_commit_duration'])
    performance_df = df[['avg_latency', 'throughput', 'disc_write']]
    config_df = df.drop(performance_df.columns, axis=1)
    weight = calculate_weight(performance_df)
    param_identification.lasso_test(config_df, performance_df, weight)
    # return df
