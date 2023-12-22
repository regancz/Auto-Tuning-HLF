from Model import initialize, storage, param_identification

if __name__ == "__main__":
    configParameters = initialize.read_yaml_config('../Benchmark-Deploy-Tool/config.yaml')
    mysql_connection = initialize.mysql_connect(configParameters['Database']['Mysql']['Host'],
                                                configParameters['Database']['Mysql']['Port'],
                                                configParameters['Database']['Mysql']['User'],
                                                configParameters['Database']['Mysql']['Password'],
                                                configParameters['Database']['Mysql']['Database'])
    parameter_data, parameter_rows = storage.query_config_parameter_by_table(mysql_connection)
    performance_data, performance_rows = storage.query_performance_metric_by_table(mysql_connection)
    param_identification.lasso_test(parameter_rows, performance_rows)
    # selected_params, selected_feats = param_identification.feature_selection(parameter_data, performance_data,
    #                                                                          alpha=0.1, method='lasso',
    #                                                                          sort_method='feature_importance')
    # print("Selected Parameters:", selected_params)
    # print("Selected Features:", selected_feats)
