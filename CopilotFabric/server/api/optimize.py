from flask import jsonify, app, Blueprint

from Benchmark_Deploy_Tool.deploy_fabric import ssh_run_command, run_fabric
from CopilotFabric.server.api import ssh_client, mysql_connection, config_parameters
from CopilotFabric.server.service.optimize import aspsa_maximize, mopso_maximize
from CopilotFabric.server.service.prediction_model import bpnn_predict_model, baseline_predict_model

data = {
    'peer_nodes_count': 2,
    'organizations': 1,
    'orderer_nodes_count': 3,
    'ca_count': 2,
    'channel': 1,
    'block_height': 7312,
    'transaction_count': 5680,
    'chaincode_count': 3
}

optimize_api = Blueprint('optimize_api', __name__)


@optimize_api.route('/model/fabricBenchmark', methods=['POST'])
def fabric_benchmark():
    # run_fabric(ssh_client, mysql_connection, config_parameters, 'default', '')
    return jsonify({'data': {'status': 'success'}, 'code': 20000})


@optimize_api.route('/model/bpnnPrediction', methods=['POST'])
def bpnn_prediction():
    # bpnn_predict_model()
    return jsonify({'data': {'status': 'success'}, 'code': 20000})


@optimize_api.route('/model/baselinePrediction', methods=['POST'])
def baseline_prediction():
    # baseline_predict_model()
    return jsonify({'data': {'status': 'success'}, 'code': 20000})


@optimize_api.route('/model/aspsaOptimize', methods=['POST'])
def aspsa_optimize():
    aspsa_maximize()
    return jsonify({'data': {'status': 'success'}, 'code': 20000})


@optimize_api.route('/model/moaspsaOptimize', methods=['POST'])
def moaspsa_optimize():
    return jsonify({'data': {'status': 'success'}, 'code': 20000})


@optimize_api.route('/model/mopsoOptimize', methods=['POST'])
def mopso_optimize():
    mopso_output = mopso_maximize()
    return jsonify({'data': mopso_output, 'code': 20000})
