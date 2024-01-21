import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from Benchmark_Deploy_Tool import initialize
from Model.mospsa.bayes import gradient_descent
from Model.mospsa.loss import evacuate_fabric_metric, evacuate_fabric, evacuate_fabric_latency, \
    evacuate_fabric_metric_prediction_model
from Model.mospsa.optimizer import spsa_maximize, adam_maximize, get_hlf_boundary_2metirc, spsa_maximize_optimize, \
    spsa_maximize2, spsa_maximize3
from Model.moopso.P_objective import get_hlf_boundary


# def main1():
#     boundary = get_hlf_boundary()
#     random_input = np.random.uniform(low=boundary['Lower'].values, high=boundary['Upper'].values)
#     adam_maximize(evacuate_fabric, random_input, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations=100, c=1.0, gamma=0.1, low=boundary['Lower'].values, high=boundary['Upper'].values)


# def main():
#     boundary = get_hlf_boundary_2metirc()
#     random_input = np.random.uniform(low=boundary['Lower'].values, high=boundary['Upper'].values)
#     # random_input = boundary['Lower'].values
#     input_dim = random_input.size
#     # create the optimizer class
#     max_iter = 100
#
#     # initial_params = [1, 2, 1000, 99, 40, 1, 12]  # 初始参数解
#     delta = np.ones(input_dim)
#     # 将奇数索引位置的元素置为 -1
#     delta[1::2] = -1  # 扰动大小
#     # num_iterations = 50  # 迭代次数
#
#     # Nostop: alpha=0.602, gamma=0.101
#     best_params, objective_values = spsa_maximize(evacuate_fabric, evacuate_fabric_latency, random_input,
#                                                   num_iterations=max_iter, rho=1,
#                                                   a=80, c=2, A=1, alpha=0.6, gamma=0.1,
#                                                   low=boundary['Lower'].values, high=boundary['Upper'].values)
#     print("最佳参数解:", best_params)
#     print("最大总延迟:", evacuate_fabric(best_params))
#     # style.use('ggplot')
#     # plt.rcParams['axes.facecolor'] = 'whitesmoke'
#     plt.plot(range(len(objective_values)), objective_values, color='#2c6fbb', label='SPSA')
#     plt.xlabel('Iteration')
#     plt.ylabel('Metric')
#     # plt.title('Change of Objective Function Value')
#     plt.show()


def main3():
    config_parameters = initialize.read_yaml_config('../../Benchmark_Deploy_Tool/config.yaml')
    ssh_client = initialize.ssh_connect(config_parameters['SSH']['Host'], config_parameters['SSH']['Port'],
                                        config_parameters['SSH']['Username'], config_parameters['SSH']['Password'])
    mysql_connection = initialize.mysql_connect(config_parameters['Database']['Mysql']['Host'],
                                                config_parameters['Database']['Mysql']['Port'],
                                                config_parameters['Database']['Mysql']['User'],
                                                config_parameters['Database']['Mysql']['Password'],
                                                config_parameters['Database']['Mysql']['Database'])
    boundary = get_hlf_boundary_2metirc()
    # np.random.seed(3)
    random_input = np.random.uniform(low=boundary['Lower'].values, high=boundary['Upper'].values)
    # random_input = [10, 10]
    # random_input = boundary['Lower'].values
    input_dim = random_input.size
    # create the optimizer class
    max_iter = 40

    # initial_params = [1, 2, 1000, 99, 40, 1, 12]  # 初始参数解
    delta = np.ones(input_dim)
    # 将奇数索引位置的元素置为 -1
    delta[1::2] = -1  # 扰动大小
    # num_iterations = 50  # 迭代次数

    # Nostop: alpha=0.602, gamma=0.101 c=9.661111877354848
    # Standard Deviation of (throughput - 0.1 * fail): 6.141553199477377
    # best_params, objective_values, best_objective = spsa_maximize(ssh_client, mysql_connection, config_parameters,
    #                                                               evacuate_fabric_metric, evacuate_fabric_latency,
    #                                                               random_input,
    #                                                               num_iterations=max_iter, rho=1,
    #                                                               a=10, c=2, A=1, alpha=0.602, gamma=0.101,
    #                                                               low=boundary['Lower'].values,
    #                                                               high=boundary['Upper'].values)
    # print("original spsa best_params:", best_params)
    # print("original spsa metric:", best_objective)
    # plt.plot(range(max_iter), objective_values[:max_iter], label='SPSA', marker='o', markersize=3)
    #
    # best_params2, objective_values2, best_objective2 = spsa_maximize2(ssh_client, mysql_connection, config_parameters,
    #                                                                   evacuate_fabric_metric, evacuate_fabric_latency,
    #                                                                   random_input,
    #                                                                   num_iterations=max_iter, rho=1,
    #                                                                   a=10, c=4, A=1, alpha=0.602, gamma=0.101,
    #                                                                   low=boundary['Lower'].values,
    #                                                                   high=boundary['Upper'].values)
    # print("original spsa2 best_params:", best_params2)
    # print("original spsa2 metric:", best_objective2)
    #
    # # plt.rcParams['axes.facecolor'] = 'whitesmoke'
    # # color='#2c6fbb',
    # plt.plot(range(len(objective_values2)), objective_values2, label='SSPSA', marker='o', markersize=3)
    #
    # best_params3, objective_values3, best_objective3 = spsa_maximize3(ssh_client, mysql_connection, config_parameters,
    #                                                                   evacuate_fabric_metric, evacuate_fabric_latency,
    #                                                                   random_input,
    #                                                                   num_iterations=max_iter, rho=1,
    #                                                                   a=10, c=4, A=1, alpha=0.602, gamma=0.101,
    #                                                                   low=boundary['Lower'].values,
    #                                                                   high=boundary['Upper'].values)
    # print("original spsa3 best_params:", best_params3)
    # print("original spsa3 metric:", best_objective3)
    #
    # # plt.rcParams['axes.facecolor'] = 'whitesmoke'
    # # color = '#2c6fbb',
    # plt.plot(range(len(objective_values3)), objective_values3, label='GSPSA', marker='o', markersize=3)

    best_params_optimize, objective_values_optimize, best_objective_optimize, param_ledger_preferred, param_ledger_count, metric_ledger = spsa_maximize_optimize(
        random_input,
        rho=1,
        num_iterations=max_iter,
        a=10, c=2, A=1,
        alpha=0.602,
        gamma=0.101,
        low=boundary[
            'Lower'].values,
        high=boundary[
            'Upper'].values)
    print("spsa_optimize best_params:", best_params_optimize)
    print("spsa_optimize metric:", best_objective_optimize)
    # style.use('ggplot')
    # plt.rcParams['axes.facecolor'] = 'whitesmoke'

    plt.plot(range(len(objective_values_optimize)), objective_values_optimize, label='ASPSA', marker='o', markersize=3)

    initial_point = np.random.uniform(low=boundary['Lower'].values, high=boundary['Upper'].values)
    initial_point[0] = 10
    initial_point[1] = 10

    best_point_bayesian, objective_values_bayesian = gradient_descent(evacuate_fabric_metric_prediction_model,
                                                                      initial_point,
                                                                      learning_rate=0.1, num_iterations=max_iter)
    print("bayesian best_params:", best_point_bayesian)
    print("bayesian metric:", objective_values_bayesian[-1])
    plt.plot(range(len(objective_values_bayesian)), objective_values_bayesian, label='Bayesian', marker='o', markersize=3)

    plt.xlabel('Iteration')
    plt.ylabel('Metric')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
    plt.show()


def main4():
    boundary = get_hlf_boundary_2metirc()
    np.random.seed(3)
    random_input = np.random.uniform(low=boundary['Lower'].values, high=boundary['Upper'].values)
    input_dim = random_input.size
    max_iter = 40
    delta = np.ones(input_dim)
    delta[1::2] = -1  # 扰动大小
    best_params_optimize, objective_values_optimize, best_objective_optimize, param_ledger_preferred, param_ledger_count, metric_ledger = spsa_maximize_optimize(
        random_input,
        rho=1,
        num_iterations=max_iter,
        a=10, c=2, A=1,
        alpha=0.602,
        gamma=0.101,
        low=boundary[
            'Lower'].values,
        high=boundary[
            'Upper'].values)
    print("spsa_optimize best_params:", best_params_optimize)
    print("spsa_optimize metric:", best_objective_optimize)
    plt.scatter(param_ledger_preferred, metric_ledger, label='ASPSA', marker='o', facecolors='none', edgecolors='blue',
                s=10)
    plt.xlabel('PreferredMaxBytes')
    plt.ylabel('Metric')
    plt.legend()
    plt.show()


def main5():
    boundary = get_hlf_boundary_2metirc()
    np.random.seed(3)
    random_input = np.random.uniform(low=boundary['Lower'].values, high=boundary['Upper'].values)
    input_dim = random_input.size
    max_iter = 40
    delta = np.ones(input_dim)
    delta[1::2] = -1  # 扰动大小
    best_params_optimize, objective_values_optimize, best_objective_optimize, param_ledger_preferred, param_ledger_count, metric_ledger = spsa_maximize_optimize(
        random_input,
        rho=1,
        num_iterations=max_iter,
        a=10, c=2, A=1,
        alpha=0.602,
        gamma=0.101,
        low=boundary[
            'Lower'].values,
        high=boundary[
            'Upper'].values)
    # 315.8 4.414 310.8 3.414
    default_obj = 310.8
    default_objs = []
    rho = 1
    for i in range(len(objective_values_optimize)):
        default_objs.append(rho * 3.414 + default_obj)
        rho += 0.1
        rho = min(rho, 2)
    sns.set(style='ticks')
    plt.plot(range(len(objective_values_optimize)), objective_values_optimize, label='ASPSA', marker='o', markersize=3)
    plt.plot(range(len(objective_values_optimize)), default_objs, label='Default', marker='o', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Metric')
    plt.legend()
    # plt.title('Change of Objective Function Value')
    plt.show()


if __name__ == "__main__":
    main3()
