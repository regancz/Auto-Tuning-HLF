import datetime
import logging
import uuid

import numpy as np

import deploy_fabric
import storage
import initialize
import config
import deploy_caliper
import helper


def main():
    config_parameters = initialize.read_yaml_config('config.yaml')
    ssh_client = initialize.ssh_connect(config_parameters['SSH']['Host'], config_parameters['SSH']['Port'],
                                        config_parameters['SSH']['Username'], config_parameters['SSH']['Password'])
    mysql_connection = initialize.mysql_connect(config_parameters['Database']['Mysql']['Host'],
                                                config_parameters['Database']['Mysql']['Port'],
                                                config_parameters['Database']['Mysql']['User'],
                                                config_parameters['Database']['Mysql']['Password'],
                                                config_parameters['Database']['Mysql']['Database'])
    deploy_fabric.run_fabric(ssh_client, mysql_connection, config_parameters, 'default', '')
    ssh_client.close()


if __name__ == "__main__":
    main()
