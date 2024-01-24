import logging

import nacos
from minio import Minio

from Model.initialize import read_yaml_config

logger = logging.getLogger("model_service")
file_handler = logging.FileHandler('D:/log/CopilotFabric.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

minio_client = Minio(
    "192.168.3.39:9000",
    access_key="admin",
    secret_key="12345678",
    secure=False
)

model_config = read_yaml_config('F:/Project/PythonProject/Auto-Tuning-HLF/CopilotFabric/server/model_config.yaml')

SERVER_ADDRESSES = "192.168.3.39:8848"
NAMESPACE = "copilotfabric"

# no auth mode
nacos_client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE)
# auth mode
# client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE, ak="{ak}", sk="{sk}")
