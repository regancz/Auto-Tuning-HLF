from Model import initialize, storage, performance_analyze
from Model.performance_analyze import aggregated_lasso_dataset, aggregate_monitor_metric
from Model.storage import update_resource_monitor

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark_Deploy_Tool/config.yaml')
    mysql_connection, engine = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                        configParameters['Database']['Mysql']['Port'],
                                                        configParameters['Database']['Mysql']['User'],
                                                        configParameters['Database']['Mysql']['Password'],
                                                        configParameters['Database']['Mysql']['Database'])
    # aggregated_lasso_dataset(mysql_connection, engine)


    # storage.prepare_data(mysql_connection)
    # storage.update_resource_monitor(mysql_connection)
    # storage.calculate_error_rate(mysql_connection)
    # aggregate_monitor_metric(mysql_connection)
    performance_analyze.aggregated_lasso_dataset(mysql_connection, engine)
    # performance_analyze.get_dataset_lasso(engine)
