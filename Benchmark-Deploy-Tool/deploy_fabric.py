import datetime
import logging
import re
import uuid

import numpy as np
import paramiko
import main

from helper import convert_to_number
from deploy_caliper import run_caliper_and_log, mv_report_and_log
from config import modify_param_yaml, get_config, modify_connection_yaml
from initialize import read_yaml_config
from storage import insert_config_to_database, get_metrics


def run_command_and_log(ssh_client, commands, log_prefix):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"D:/log/{log_prefix}_{timestamp}.log"
    logger = logging.getLogger(f'{log_prefix}_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    logger.addHandler(file_handler)
    try:
        outputs = ssh_run_command(ssh_client, commands)
        for index, output in enumerate(outputs):
            logger.info(
                f"{timestamp} Command '{commands[index]}' output :\n{remove_ansi_escape_codes(output)}")
        print(f"Output logged to {log_file}")
    except Exception as e:
        logger.error(f"Error running command: {e}")
        print(f"Error running command: {e}")


def ssh_run_command(ssh_client, commands):
    try:
        outputs = []
        for command in commands:
            print("%s is executing" % command)
            stdin, stdout, stderr = ssh_client.exec_command(command)
            output = stdout.read().decode("utf-8")
            outputs.append(output)
        return outputs
    except paramiko.AuthenticationException as auth_exception:
        print(f"Authentication failed: {auth_exception}")
    except paramiko.SSHException as ssh_exception:
        print(f"SSH connection failed: {ssh_exception}")
    except Exception as e:
        print(f"An error occurred: {e}")


def deploy_fabric_and_log(ssh_client, cc_name):
    commands = ["cd /home/charles/Project/Blockchain/fabric-samples/test-network && ./network.sh down",
                "cd /home/charles/Project/Blockchain/fabric-samples/test-network && ./network.sh up createChannel",
                # "cd /home/charles/Project/Blockchain/fabric-samples/test-network && source /etc/profile && ./network.sh deployCC -ccn simple -ccp ../../caliper-benchmarks/src/fabric/scenario/simple/go -ccl go",
                "cd /home/charles/Project/Blockchain/fabric-samples/test-network && source /etc/profile && ./network.sh deployCC -ccn "
                + cc_name +
                " -ccp ../../caliper-benchmarks/src/fabric/scenario/"
                + cc_name +
                "/go -ccl go"]
    run_command_and_log(ssh_client, commands, "fabric_log")


def remove_ansi_escape_codes(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def run_fabric(ssh_client, mysql_connection, config_parameters, mode, param_type):
    if mode == 'default':
        config_id = str(uuid.uuid1())
        run_fabric_default(ssh_client, mysql_connection, config_parameters, config_id)
    elif mode == 'benchmark':
        if param_type == 'Orderer':
            run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, 'Orderer')
        elif param_type == 'Configtx':
            run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, 'Configtx')
        elif param_type == 'Peer':
            run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, 'Peer')
        elif param_type == 'BftConfigtx':
            run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, 'BftConfigtx')
        elif param_type == 'all':
            run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, 'Orderer')
            run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, 'Configtx')
            run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, 'Peer')
            run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, 'BftConfigtx')
        else:
            print('run_fabric incorrect param_type')


def run_fabric_default(ssh_client, mysql_connection, config_parameters, config_id):
    updates_connection_org1 = {
        'peers-peer0.org1.example.com-url': 'grpcs://192.168.3.39:7051',
        'certificateAuthorities-ca.org1.example.com-url': 'https://192.168.3.39:7054',
    }
    cc_names = ['simple', 'smallbank']
    for cc_name in cc_names:
        performance_id = str(uuid.uuid1())
        deploy_fabric_and_log(ssh_client, cc_name)
        modify_connection_yaml(ssh_client, config_parameters['ConfigPath']['ConnectionOrg1'],
                               updates_connection_org1)
        run_caliper_and_log(ssh_client, cc_name)
        order_param = get_config(ssh_client, config_parameters['ConfigPath']['Orderer'],
                                 config_parameters['Parameters']['Orderer'])
        configtx_param = get_config(ssh_client, config_parameters['ConfigPath']['Configtx'],
                                    config_parameters['Parameters']['Configtx'])
        peer_param = get_config(ssh_client, config_parameters['ConfigPath']['Peer'],
                                config_parameters['Parameters']['Peer'])
        insert_config_to_database(mysql_connection, order_param, configtx_param, peer_param,
                                  config_id)
        get_metrics(mysql_connection, ssh_client, config_id, performance_id)
        mv_report_and_log(ssh_client, config_id, f'{performance_id}_{cc_name}')
        break


def run_fabric_benchmark(ssh_client, mysql_connection, config_parameters, param_type):
    param_range = read_yaml_config('param_range.yaml')
    for k, v in param_range['Parameters'][param_type].items():
        lower = v['lower']
        upper = v['upper']
        step = v['step']
        lower_value, unit = convert_to_number(str(lower))
        upper_value, unit = convert_to_number(str(upper))
        for i in np.arange(lower_value, upper_value + 1, step):
            new_value = (str(i) + unit) if unit else str(i)
            updates = {k: new_value}
            config_id = str(uuid.uuid1())
            modify_param_yaml(ssh_client, config_parameters['ConfigPath'][param_type], updates)
            run_fabric_default(ssh_client, mysql_connection, config_parameters, config_id)