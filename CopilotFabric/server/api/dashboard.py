from flask import jsonify, app, Blueprint

from Benchmark_Deploy_Tool.deploy_fabric import ssh_run_command
from CopilotFabric.server.api import ssh_client

data = {
    'peer_nodes_count': 4,
    'organizations': 2,
    'orderer_nodes_count': 3,
    'ca_count': 2,
    'channel': 1,
    'block_height': 15517,
    'transaction_count': 1024122,
    'chaincode_count': 3
}

dashboard_api = Blueprint('dashboard_api', __name__)


@dashboard_api.route('/count/peer', methods=['GET'])
def get_peer_nodes_count():
    # commands = ["docker ps -a --filter 'name=^peer' --format '{{.Names}}' | wc -l"]
    # outputs = ssh_run_command(ssh_client, commands)
    # return jsonify({'data': {'count': outputs}, 'code': 20000})
    return jsonify({'data': {'count': data['peer_nodes_count']}, 'code': 20000})

@dashboard_api.route('/count/organization', methods=['GET'])
def get_organizations():
    # commands = ["docker ps --format '{{.Names}}' | grep '^peer' | grep -o 'org[0-9]*' | sort -u | wc -l"]
    # outputs = ssh_run_command(ssh_client, commands)
    # return jsonify({'data': {'count': outputs}, 'code': 20000})
    return jsonify({'data': {'count': data['organizations']}, 'code': 20000})

@dashboard_api.route('/count/orderer', methods=['GET'])
def get_orderer_nodes_count():
    # commands = ["docker ps -a --filter 'name=^orderer' --format '{{.Names}}' | wc -l"]
    # outputs = ssh_run_command(ssh_client, commands)
    # return jsonify({'data': {'count': outputs}, 'code': 20000})
    return jsonify({'data': {'count': data['orderer_nodes_count']}, 'code': 20000})


@dashboard_api.route('/count/ca', methods=['GET'])
def get_ca_count():
    # commands = ["docker ps -a --filter 'name=^ca' --format '{{.Names}}' | wc -l"]
    # outputs = ssh_run_command(ssh_client, commands)
    return jsonify({'data': {'count': data['ca_count']}, 'code': 20000})


@dashboard_api.route('/count/channel', methods=['GET'])
def get_channel():
    return jsonify({'data': {'count': data['channel']}, 'code': 20000})


@dashboard_api.route('/count/blockHeight', methods=['GET'])
def get_block_height():
    return jsonify({'data': {'count': data['block_height']}, 'code': 20000})


@dashboard_api.route('/count/transaction', methods=['GET'])
def get_transaction_count():
    return jsonify({'data': {'count': data['transaction_count']}, 'code': 20000})


@dashboard_api.route('/count/chaincode', methods=['GET'])
def get_chaincode_count():
    # commands = ["docker ps --format '{{.Names}}' | grep '^dev-peer' | awk -F- '{print $3}' | sort -u | wc -l"]
    # outputs = ssh_run_command(ssh_client, commands)
    # return jsonify({'data': {'count': outputs}, 'code': 20000})
    return jsonify({'data': {'count': data['chaincode_count']}, 'code': 20000})
