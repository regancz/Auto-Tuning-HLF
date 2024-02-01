from Benchmark_Deploy_Tool import initialize

config_parameters = initialize.read_yaml_config('../../Benchmark_Deploy_Tool/config.yaml')
ssh_client = initialize.ssh_connect(config_parameters['SSH']['Host'], config_parameters['SSH']['Port'],
                                    config_parameters['SSH']['Username'], config_parameters['SSH']['Password'])
connection = initialize.mysql_connect(config_parameters['Database']['Mysql']['Host'],
                                            config_parameters['Database']['Mysql']['Port'],
                                            config_parameters['Database']['Mysql']['User'],
                                            config_parameters['Database']['Mysql']['Password'],
                                            config_parameters['Database']['Mysql']['Database'])
performance_id = 'a7e084a5-b42d-11ee-9035-005056c00008-0'
try:
    with connection.cursor() as cursor:
        connection.ping(reconnect=True)
        metric_name = 'throughput'
        query = f"SELECT {metric_name} FROM performance_metric WHERE id = %s"
        # query = f"SELECT {columns_str} FROM config_parameter LIMIT 100"
        cursor.execute(query, (performance_id))
        col_rows = cursor.fetchall()
        metric_val = []
        for d in col_rows:
            for val in d.items():
                metric_val.append(val[1])
        connection.commit()
# except pymysql.Error as e:
#     print("could not close connection error pymysql %d: %s" % (e.args[0], e.args[1]))
finally:
    print("insert_metric_to_database done")