import yaml


def modify_param_yaml(ssh_client, yaml_file, updates):
    try:
        stdin, stdout, stderr = ssh_client.exec_command(f'cat {yaml_file}')
        yaml_content = stdout.read().decode()
        data = yaml.safe_load(yaml_content)
        for key, value in updates.items():
            nested_keys = key.split('_')
            temp_data = data
            for k in nested_keys[:-1]:
                temp_data = temp_data[k]
            temp_data[nested_keys[-1]] = value
        updated_yaml = yaml.dump(data, default_flow_style=False)
        stdin, stdout, stderr = ssh_client.exec_command(f'echo "{updated_yaml}" > {yaml_file}')
    finally:
        print(f"Already modify {yaml_file}")


def modify_connection_yaml(ssh_client, yaml_file, updates):
    try:
        stdin, stdout, stderr = ssh_client.exec_command(f'cat {yaml_file}')
        yaml_content = stdout.read().decode()
        data = yaml.safe_load(yaml_content)
        for key, value in updates.items():
            nested_keys = key.split('-')
            temp_data = data
            for k in nested_keys[:-1]:
                temp_data = temp_data[k]
            temp_data[nested_keys[-1]] = value
        updated_yaml = yaml.dump(data, default_flow_style=False)
        stdin, stdout, stderr = ssh_client.exec_command(f'echo "{updated_yaml}" > {yaml_file}')
    finally:
        print(f"Already modify {yaml_file}")


def get_config(ssh_client, yaml_file, para_name):
    try:
        stdin, stdout, stderr = ssh_client.exec_command(f'cat {yaml_file}')
        yaml_content = stdout.read().decode()
        data = yaml.safe_load(yaml_content)
        param = {}
        for name in para_name:
            nested_keys = name.split('_')
            temp_data = data
            for k in nested_keys[:-1]:
                temp_data = temp_data[k]
            param[name] = temp_data[nested_keys[-1]]
        return param
    finally:
        print(f"Read config: {yaml_file}")
