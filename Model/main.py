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
    task_type = "param_select"
    if task_type == "param_select":
        get_dataset_lasso(engine)
    elif task_type == 'bpnn_predict_model':
        bpnn_predict_model()
    elif task_type == 'baseline_predict_model':
        baseline_predict_model()


if __name__ == "__main__":
    main()
