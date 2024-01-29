import io

from ruamel.yaml import YAML


def modify_yaml(ssh_client, yaml_file, updates, delimiter='_'):
    try:
        data = load_yaml(ssh_client, yaml_file)
        for key, value in updates.items():
            nested_keys = key.split(delimiter)
            temp_data = data
            for k in nested_keys[:-1]:
                temp_data = temp_data[k]
            temp_data[nested_keys[-1]] = value

        buf = io.BytesIO()
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(data, buf)

        ftp_client = ssh_client.open_sftp()
        with ftp_client.file(f'{yaml_file}', 'w') as remote_file:
            remote_file.write(buf.getvalue())

    finally:
        print(f"Already modify {yaml_file}")


def modify_param_yaml(ssh_client, yaml_file, updates):
    modify_yaml(ssh_client, yaml_file, updates, delimiter='_')


def modify_connection_yaml(ssh_client, yaml_file, updates):
    modify_yaml(ssh_client, yaml_file, updates, delimiter='-')


def load_yaml(ssh_client, yaml_file):
    stdin, stdout, stderr = ssh_client.exec_command(f'cat {yaml_file}')
    yaml_content = stdout.read().decode('utf-8')
    yaml = YAML(typ='rt')
    yaml.preserve_quotes = True
    yaml.preserve_quotes_roundtrip = True
    yaml.preserve_scalar_quotes = True
    yaml.preserve_indent = True
    yaml.preserve_block = True
    return yaml.load(yaml_content)


def get_config(ssh_client, yaml_file, para_name):
    try:
        data = load_yaml(ssh_client, yaml_file)
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
