from Model.initialize import read_yaml_config, mysql_connect

config_parameters = read_yaml_config('F:/Project/PythonProject/Auto-Tuning-HLF/Benchmark_Deploy_Tool/config.yaml')
mysql_connection, engine = mysql_connect(config_parameters['Database']['Mysql']['Host'],
                                         config_parameters['Database']['Mysql']['Port'],
                                         config_parameters['Database']['Mysql']['User'],
                                         config_parameters['Database']['Mysql']['Password'],
                                         config_parameters['Database']['Mysql']['Database'])
