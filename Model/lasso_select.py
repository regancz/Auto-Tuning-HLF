from Model import initialize
from Model.performance_analyze import get_dataset_lasso

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark_Deploy_Tool/config.yaml')
    mysql_connection, engine = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])

    get_dataset_lasso(engine)
