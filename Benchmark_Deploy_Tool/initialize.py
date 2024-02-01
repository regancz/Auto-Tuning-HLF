import uuid
from datetime import datetime

import paramiko
import pymysql
import yaml
import time


def ssh_connect(host, port, username, password):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=host, port=port, username=username, password=password, timeout=1800)
    return ssh_client


def mysql_connect(host, port, user, password, db):
    connection = pymysql.connect(host=host,
                                 port=port,
                                 user=user,
                                 password=password,
                                 db=db,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    return connection


def read_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data


def get_timestamp():
    return int(time.time() * 1000)


def til_next_millis(last_timestamp):
    timestamp = get_timestamp()
    while timestamp <= last_timestamp:
        timestamp = get_timestamp()
    return timestamp


class SnowflakeIdGenerator:
    def __init__(self, datacenter_id, worker_id):
        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.epoch = 1288834974657
        self.sequence = 0
        self.worker_id_bits = 5
        self.datacenter_id_bits = 5
        self.max_worker_id = -1 ^ (-1 << self.worker_id_bits)
        self.max_datacenter_id = -1 ^ (-1 << self.datacenter_id_bits)
        self.sequence_bits = 12
        self.last_timestamp = -1

        self.worker_id_shift = self.sequence_bits
        self.datacenter_id_shift = self.sequence_bits + self.worker_id_bits
        self.timestamp_left_shift = self.sequence_bits + self.worker_id_bits + self.datacenter_id_bits
        self.sequence_mask = -1 ^ (-1 << self.sequence_bits)

    def generate_id(self):
        timestamp = get_timestamp()

        if timestamp < self.last_timestamp:
            raise Exception("Clock moved backwards. Refusing to generate id for %d milliseconds" % (
                    self.last_timestamp - timestamp))

        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & self.sequence_mask
            if self.sequence == 0:
                timestamp = til_next_millis(self.last_timestamp)
        else:
            self.sequence = 0

        self.last_timestamp = timestamp

        return ((timestamp - self.epoch) << self.timestamp_left_shift) | (
                self.datacenter_id << self.datacenter_id_shift) | (
                       self.worker_id << self.worker_id_shift) | self.sequence

    def extract_date_from_id(self, snowflake_id):
        timestamp = (snowflake_id >> self.timestamp_left_shift) + self.epoch
        date = datetime.fromtimestamp(timestamp / 1000)  # 转换为秒级时间戳并创建日期对象
        return date.strftime("%Y-%m-%d")  # 返回格式化后的日期字符串
