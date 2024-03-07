import argparse

from CopilotFabric.server.service.prediction_model import bpnn_predict_model, baseline_predict_model
from Model.data.performance_analyze import get_dataset_lasso
from Model.initialize import read_yaml_config, mysql_connect


def main():
    configParameters = read_yaml_config('../Benchmark_Deploy_Tool/config.yaml')
    mysql_connection, engine = mysql_connect(configParameters['Database']['Mysql']['Host'],
                                             configParameters['Database']['Mysql']['Port'],
                                             configParameters['Database']['Mysql']['User'],
                                             configParameters['Database']['Mysql']['Password'],
                                             configParameters['Database']['Mysql']['Database'])
    parser = argparse.ArgumentParser(description='Process different task types.')
    parser.add_argument('--task',
                        help='Specify the task type (param_select, bpnn_predict_model, baseline_predict_model)',
                        required=True)
    args = parser.parse_args()

    task_type = args.param

    if task_type == "param_select":
        get_dataset_lasso(engine)
    elif task_type == 'bpnn_predict_model':
        bpnn_predict_model()
    elif task_type == 'baseline_predict_model':
        baseline_predict_model()
    else:
        print(f"Unsupported task_type: {task_type}")


if __name__ == "__main__":
    main()
