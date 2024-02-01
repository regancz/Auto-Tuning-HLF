from Benchmark_Deploy_Tool.deploy_fabric import run_fabric
from Model.initialize import read_yaml_config, ssh_connect, mysql_connect


def main():
    config_parameters = read_yaml_config('../../Benchmark_Deploy_Tool/config.yaml')
    ssh_client = ssh_connect(config_parameters['SSH']['Host'], config_parameters['SSH']['Port'],
                                        config_parameters['SSH']['Username'], config_parameters['SSH']['Password'])
    mysql_connection = mysql_connect(config_parameters['Database']['Mysql']['Host'],
                                                config_parameters['Database']['Mysql']['Port'],
                                                config_parameters['Database']['Mysql']['User'],
                                                config_parameters['Database']['Mysql']['Password'],
                                                config_parameters['Database']['Mysql']['Database'])
    # run_fabric(ssh_client, mysql_connection, config_parameters, 'benchmark', 'Configtx')
    run_fabric(ssh_client, mysql_connection, config_parameters, 'default', 'Configtx')
    run_fabric(ssh_client, mysql_connection, config_parameters, 'default', 'Configtx')
    run_fabric(ssh_client, mysql_connection, config_parameters, 'default', 'Configtx')
    run_fabric(ssh_client, mysql_connection, config_parameters, 'default', 'Configtx')
    run_fabric(ssh_client, mysql_connection, config_parameters, 'default', 'Configtx')
    ssh_client.close()


if __name__ == "__main__":
    main()