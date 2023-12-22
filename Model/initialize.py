import paramiko
import pymysql
import yaml


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
