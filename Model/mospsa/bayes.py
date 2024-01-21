import numpy as np
import matplotlib.pyplot as plt

from Model import initialize
from Model.mospsa.loss import evacuate_fabric_metric_prediction_model
from Model.mospsa.optimizer import get_hlf_boundary_2metirc


def main():
    config_parameters = initialize.read_yaml_config('../../Benchmark_Deploy_Tool/config.yaml')
    ssh_client = initialize.ssh_connect(config_parameters['SSH']['Host'], config_parameters['SSH']['Port'],
                                        config_parameters['SSH']['Username'], config_parameters['SSH']['Password'])
    mysql_connection = initialize.mysql_connect(config_parameters['Database']['Mysql']['Host'],
                                                config_parameters['Database']['Mysql']['Port'],
                                                config_parameters['Database']['Mysql']['User'],
                                                config_parameters['Database']['Mysql']['Password'],
                                                config_parameters['Database']['Mysql']['Database'])
    boundary = get_hlf_boundary_2metirc()
    initial_point = np.random.uniform(low=boundary['Lower'].values, high=boundary['Upper'].values)
    initial_point[0] = 10
    initial_point[1] = 10

    # 用于记录每次迭代后的目标函数值
    best_point, objective_values = gradient_descent(evacuate_fabric_metric_prediction_model, initial_point,
                                                    learning_rate=0.015, num_iterations=30)

    # 打印结果
    print("最佳参数解:", best_point)
    print("最大目标值:", objective_values[-1])

    # 绘制目标函数值的变化
    plt.plot(range(len(objective_values)), objective_values, color='green', label='Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.show()


def gradient_descent(objective_function, initial_point, learning_rate=0.01, num_iterations=100):
    current_point = initial_point
    objective_values = []
    param_bounds = [(1, 100), (1, 1000)]
    rho = 1

    for _ in range(num_iterations):
        gradient = calculate_gradient(objective_function, current_point, rho=rho)
        current_point -= learning_rate * gradient
        current_point = clip_params(current_point, param_bounds)
        objective_value = objective_function(current_point, rho=rho)
        objective_values.append(objective_value)

        # Adjust rho
        rho += 0.1
        rho = min(rho, 2)

    return current_point, objective_values


def clip_params(params, bounds):
    # 确保参数在指定范围内
    if bounds is not None:
        for i, (lower, upper) in enumerate(bounds):
            params[i] = max(lower, min(params[i], upper))
    return params


def calculate_gradient(objective_function, point, epsilon=1e-8, rho=1):
    gradient = np.zeros_like(point)

    for i in range(len(point)):
        perturbation = np.zeros_like(point)
        perturbation[i] = epsilon
        gradient[i] = (objective_function(point + perturbation, rho=rho) - objective_function(point - perturbation, rho=rho)) / (
                    2 * epsilon)

    return gradient


if __name__ == "__main__":
    main()
