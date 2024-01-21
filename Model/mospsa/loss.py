import datetime
import logging
import random
import uuid

import joblib
import numpy as np

from Benchmark_Deploy_Tool.config import modify_connection_yaml, get_config, modify_param_yaml
from Benchmark_Deploy_Tool.deploy_fabric import deploy_fabric_and_log, run_caliper_and_log, mv_report_and_log
from Benchmark_Deploy_Tool.storage import insert_config_to_database, get_metrics, query_metric_by_performance_id


class L2Loss(object):
    def __call__(self, W, *args):
        # extract the parameters
        X = args[0][0]
        Y = args[0][1]

        # data loss
        pred = np.dot(W, X)
        squared_loss = np.sum((Y - pred) ** 2, axis=1).reshape((Y.shape[0], -1))
        average_squared_loss = squared_loss / X.shape[1]
        return average_squared_loss


def evacuate_fabric(input_param):
    input_param = np.array(input_param).reshape(1, len(input_param))
    # create & modify & query & open & query & transfer
    payload_function = 'transfer'
    # ['XGBoost', 'SVR', 'AdaBoost', 'KNeighbors']
    model_name = 'SVR'
    # 'throughput', 'avg_latency', 'disc_write'
    for target_col in ['throughput']:
        model = joblib.load(f'../traditional_model/{model_name}/{target_col}_{payload_function}_2metric_best_model.pkl')
        output = model.predict(input_param)
    return output


def evacuate_fabric_latency(input_param):
    input_param = np.array(input_param).reshape(1, len(input_param))
    # create & modify & query & open & query & transfer
    payload_function = 'transfer'
    # ['XGBoost', 'SVR', 'AdaBoost', 'KNeighbors']
    model_name = 'KNeighbors'
    # 'throughput', 'avg_latency', 'disc_write'
    for target_col in ['avg_latency']:
        model = joblib.load(f'../traditional_model/{model_name}/{target_col}_{payload_function}_2metric_best_model.pkl')
        output = model.predict(input_param)
    return output


def evacuate_fabric_disc_write(input_param):
    input_param = np.array(input_param).reshape(1, len(input_param))
    # create & modify & query & open & query & transfer
    payload_function = 'transfer'
    # ['XGBoost', 'SVR', 'AdaBoost', 'KNeighbors']
    model_name = 'KNeighbors'
    # 'throughput', 'avg_latency', 'disc_write'
    for target_col in ['disc_write']:
        model = joblib.load(f'../traditional_model/{model_name}/{target_col}_{payload_function}_2metric_best_model.pkl')
        output = model.predict(input_param)
    return output


def evacuate_fabric_metric_prediction_model(input_param, rho):
    # metric = 'throughput'
    # create & modify & query & open & query & transfer
    cc_name = 'transfer'
    input_param = np.array(input_param).reshape(1, len(input_param))
    # create & modify & query & open & query & transfer
    # payload_function = 'transfer'
    # ['XGBoost', 'SVR', 'AdaBoost', 'KNeighbors']
    model_name = random.choice(['SVR'])
    # 'throughput', 'avg_latency', 'disc_write'
    # for target_col in ['disc_write']:
    model_throughput = joblib.load(f'../traditional_model/spsa/{model_name}/{cc_name}_throughput_2metric_model.pkl')
    throughput = model_throughput.predict(input_param)
    # print(model_name)
    model_error_rate = joblib.load(f'../traditional_model/spsa/{model_name}/{cc_name}_error_rate_2metric_model.pkl')
    error_rate = model_error_rate.predict(input_param)
    # fail = 0
    model_disc_write = joblib.load(f'../traditional_model/spsa/{model_name}/{cc_name}_disc_write_2metric_model.pkl')
    disc_write = model_disc_write.predict(input_param)

    model_avg_latency = joblib.load(f'../traditional_model/spsa/{model_name}/{cc_name}_avg_latency_2metric_model.pkl')
    avg_latency = model_avg_latency.predict(input_param)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    log_prefix = 'spsa'
    log_file = f"F:/Project/PythonProject/Auto-Tuning-HLF/Model/log/spsa/{log_prefix}_{timestamp}.log"

    logger = logging.getLogger(f'{log_prefix}_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logger.info(f"input_param: {input_param} throughput: {throughput} error_rate: {error_rate} disc_write: {disc_write}")
    print(f"input_param: {input_param} throughput: {throughput} error_rate: {error_rate} disc_write: {disc_write}")
    return (throughput + rho * disc_write)


def evacuate_fabric_metric(ssh_client, mysql_connection, config_parameters, input_param, payload_function):
    config_id = str(uuid.uuid1())
    input_param = {'Orderer_BatchSize_PreferredMaxBytes': input_param[0],
                   'Orderer_BatchSize_MaxMessageCount': input_param[1]}
    return run_fabric_spsa(ssh_client, mysql_connection, config_parameters, config_id, input_param)


def run_fabric_spsa(ssh_client, mysql_connection, config_parameters, config_id, input_param):
    updates_connection_org1 = {
        'peers-peer0.org1.example.com-url': 'grpcs://192.168.3.39:7051',
        'certificateAuthorities-ca.org1.example.com-url': 'https://192.168.3.39:7054',
    }
    updates = {
        'Orderer_BatchSize_MaxMessageCount': int(input_param['Orderer_BatchSize_MaxMessageCount']),
        'Orderer_BatchSize_PreferredMaxBytes': str(int(input_param['Orderer_BatchSize_PreferredMaxBytes'])) + ' MB'
    }
    modify_param_yaml(ssh_client, config_parameters['ConfigPath']['Configtx'], updates)
    cc_names = ['simple', 'smallbank']
    for cc_name in cc_names:
        performance_id = str(uuid.uuid1())
        deploy_fabric_and_log(ssh_client, cc_name)
        modify_connection_yaml(ssh_client, config_parameters['ConfigPath']['ConnectionOrg1'],
                               updates_connection_org1)
        run_caliper_and_log(ssh_client, cc_name + '_spsa')
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
        throughput = query_metric_by_performance_id(mysql_connection, 'throughput', performance_id)
        fail = query_metric_by_performance_id(mysql_connection, 'fail', performance_id)
        return throughput - 0.1 * fail
