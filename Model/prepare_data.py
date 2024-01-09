from Model import initialize
from Model.performance_analyze import aggregated_lasso_dataset, aggregate_monitor_metric
from Model.storage import update_resource_monitor

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark-Deploy-Tool/config.yaml')
    mysql_connection, engine = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])
    aggregated_lasso_dataset(mysql_connection, engine)
    # aggregate_monitor_metric(mysql_connection)