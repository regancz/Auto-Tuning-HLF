from Benchmark_Deploy_Tool.initialize import ssh_connect, read_yaml_config
from Model.initialize import mysql_connect

config_parameters = read_yaml_config('/Benchmark_Deploy_Tool/config.yaml')
ssh_client = ssh_connect(config_parameters['SSH']['Host'], config_parameters['SSH']['Port'],
                         config_parameters['SSH']['Username'], config_parameters['SSH']['Password'])
mysql_connection, engine = mysql_connect(config_parameters['Database']['Mysql']['Host'],
                                         config_parameters['Database']['Mysql']['Port'],
                                         config_parameters['Database']['Mysql']['User'],
                                         config_parameters['Database']['Mysql']['Password'],
                                         config_parameters['Database']['Mysql']['Database'])
