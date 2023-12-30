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

if __name__ == "__main__":
    updates_connection_org1 = {
        'peers-peer0.org1.example.com-url': 'grpcs://192.168.3.39:7051',
        'certificateAuthorities-ca.org1.example.com-url': 'https://192.168.3.39:7054',
    }
    config_parameters = initialize.read_yaml_config('config.yaml')
    param_range = initialize.read_yaml_config('param_range.yaml')
    ssh_client = initialize.ssh_connect(config_parameters['SSH']['Host'], config_parameters['SSH']['Port'],
                                        config_parameters['SSH']['Username'], config_parameters['SSH']['Password'])
    mysql_connection = initialize.mysql_connect(config_parameters['Database']['Mysql']['Host'],
                                                config_parameters['Database']['Mysql']['Port'],
                                                config_parameters['Database']['Mysql']['User'],
                                                config_parameters['Database']['Mysql']['Password'],
                                                config_parameters['Database']['Mysql']['Database'])
    cc_names = ['simple', 'smallbank']
    for k, v in param_range['Parameters']['Peer'].items():
        lower = v['lower']
        upper = v['upper']
        step = v['step']
        lower_value, unit = helper.convert_to_number(str(lower))
        upper_value, unit = helper.convert_to_number(str(upper))
        # unit = lower[-1] if lower[-1].isalpha() else ''
        for i in np.arange(lower_value, upper_value + 1, step):
            new_value = (str(i) + unit) if unit else str(i)
            updates = {k: new_value}
            config_id = str(uuid.uuid1())
            config.modify_param_yaml(ssh_client, config_parameters['ConfigPath']['Peer'], updates)
            for cc_name in cc_names:
                performance_id = str(uuid.uuid1())

                deploy_fabric.deploy_fabric_and_log(ssh_client, cc_name)
                config.modify_connection_yaml(ssh_client, config_parameters['ConfigPath']['ConnectionOrg1'], updates_connection_org1)
                deploy_caliper.run_caliper_and_log(ssh_client, cc_name)
                order_param = config.get_config(ssh_client, config_parameters['ConfigPath']['Orderer'],
                                                config_parameters['Parameters']['Orderer'])
                configtx_param = config.get_config(ssh_client, config_parameters['ConfigPath']['Configtx'],
                                                   config_parameters['Parameters']['Configtx'])
                peer_param = config.get_config(ssh_client, config_parameters['ConfigPath']['Peer'],
                                               config_parameters['Parameters']['Peer'])

                # bft_configtx_param = config.get_config(ssh_client, configParameters['ConfigPath']['BftConfigtx'],
                #                                        configParameters['Parameters']['BftConfigtx'])
                # storage.insert_bft_config_to_database(mysql_connection, order_param, configtx_param, bft_configtx_param, peer_param,
                #                                       config_id)

                storage.insert_config_to_database(mysql_connection, order_param, configtx_param, peer_param,
                                                  config_id)
                storage.get_metrics(mysql_connection, ssh_client, config_id, performance_id)
                deploy_caliper.mv_report_and_log(ssh_client, config_id, performance_id)
    ssh_client.close()
