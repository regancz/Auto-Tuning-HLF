import uuid

import numpy as np
import paramiko
from bs4 import BeautifulSoup
import pymysql.cursors


def query_config_parameter_by_table(mysql_connect):
    cursor = mysql_connect.cursor()
    try:
        # 执行查询语句获取指定 id 的多行数据
        query_col = "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'config_parameter' AND COLUMN_NAME NOT IN ('id')"
        cursor.execute(query_col)
        col_rows = cursor.fetchall()
        col_name = []
        for d in col_rows:
            for val in d.items():
                col_name.append(val[1])
        columns_str = ', '.join(col_name)
        query = f"SELECT {columns_str} FROM config_parameter LIMIT 100"
        cursor = mysql_connect.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        data_array = []
        for d in rows:
            data_col = []
            for val in d.items():
                data_col.append(val[1])
            data_array.append(data_col)
        return data_array, rows

    except Exception as e:
        print(f"Error querying data: {e}")
        return None

    finally:
        cursor.close()


def query_performance_metric_by_table(mysql_connect):
    cursor = mysql_connect.cursor()
    try:
        # 执行查询语句获取指定 id 的多行数据
        # query = "SELECT succ, fail, send_rate, max_latency, min_latency, avg_latency, throughput FROM performance_metric LIMIT 100"
        query = "SELECT throughput FROM performance_metric LIMIT 100"
        cursor.execute(query)
        rows = cursor.fetchall()
        data_array = []
        for d in rows:
            data_col = []
            for val in d.items():
                data_col.append(val[1])
            data_array.append(data_col)
        return data_array, rows

    except Exception as e:
        print(f"Error querying data: {e}")
        return None

    finally:
        cursor.close()


def prepare_data(mysql_connect):
    try:
        cursor = mysql_connect.cursor()
        # 从 config_parameter 中选择满足条件的rows
        cursor.execute("SELECT id FROM config_parameter WHERE peer_discovery_authCachePurgeRetentionRatio >= 1")
        selected_ids = [row['id'] for row in cursor.fetchall()]
        for id_value in selected_ids:
            cursor.execute("UPDATE performance_metric_copy2 SET stage = 0 WHERE config_id = %s", id_value)
            cursor.execute("SELECT id FROM performance_metric_copy2 WHERE config_id = %s ", id_value)
        selected_performance_id = [row['id'] for row in cursor.fetchall()]
        # for table in ['performance_metric_copy2', 'resource_monitor_copy2']:
        for id_value in selected_performance_id:
            cursor.execute("UPDATE resource_monitor_copy2 SET stage = 0 WHERE performance_id = %s", id_value)

        # 提交事务
        mysql_connect.commit()

    # except Exception as e:
    #     # 如果发生异常，回滚事务
    #     mysql_connect.rollback()
    #     print(f"Error: {e}")

    finally:
        # 关闭游标
        cursor.close()


def update_resource_monitor(mysql_connect):
    try:
        cursor = mysql_connect.cursor()
        cursor.execute("SELECT id, metric, prometheus_query FROM resource_monitor ORDER BY id")
        rows = cursor.fetchall()

        prev_metric = None
        prev_prometheus_query = None

        for row in rows:
            id_val, metric, prometheus_query = row['id'], row['metric'], row['prometheus_query']
            if metric is None or metric == '' or prometheus_query is None or prometheus_query == '':
                metric = prev_metric if metric == '' or metric is None else metric
                prometheus_query = prev_prometheus_query if prometheus_query == '' or prometheus_query is None else prometheus_query
                cursor.execute("UPDATE resource_monitor_copy2 SET metric = %s, prometheus_query = %s WHERE id = %s",
                               (metric, prometheus_query, id_val))
            prev_metric = metric
            prev_prometheus_query = prometheus_query

        # 提交事务
        mysql_connect.commit()

    except Exception as e:
        # 如果发生异常，回滚事务
        mysql_connect.rollback()
        print(f"Error: {e}")

    finally:
        # 关闭游标
        cursor.close()


def calculate_error_rate(mysql_connect):
    try:
        cursor = mysql_connect.cursor()
        cursor.execute("SELECT succ, fail FROM performance_metric_copy2")
        rows = cursor.fetchall()

        for row in rows:
            succ, fail = row['succ'], row['fail']
            error_rate = round(fail / (fail + succ) * 100, 3)
            # 将计算出的错误率更新到数据库的新列 error_rate 中，假设有一列叫做 error_rate
            cursor.execute("UPDATE performance_metric_copy2 SET error_rate = %s WHERE succ = %s AND fail = %s",
                           (error_rate, succ, fail))

        # 提交事务
        mysql_connect.commit()

    except Exception as e:
        # 如果发生异常，回滚事务
        mysql_connect.rollback()
        print(f"Error: {e}")

    finally:
        # 关闭游标
        cursor.close()
