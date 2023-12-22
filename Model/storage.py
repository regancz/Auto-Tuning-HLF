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