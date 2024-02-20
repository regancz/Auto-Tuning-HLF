import uuid

import nacos

SERVER_ADDRESSES = "192.168.3.39:8848"
NAMESPACE = "copilotfabric"

# no auth mode
nacos_client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE)


def main():

    # auth mode
    # client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE, ak="{ak}", sk="{sk}")
    yaml_file_path = '/Model/mopso/output.yaml'
    data_id = "test"
    group = 'test'

    upload_yaml_to_nacos(yaml_file_path, data_id, group)


def upload_yaml_to_nacos(yaml_file_path, data_id, group):
    # with open(yaml_file_path, 'r') as file:
    #     yaml_content = file.read()
    # result = nacos_client.publish_config(data_id=data_id, group=group, content=yaml_content)
    # print(result)
    config = nacos_client.get_configs(group=group)
    print(config)


if __name__ == '__main__':
    main()
