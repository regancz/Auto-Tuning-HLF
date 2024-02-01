import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from Model import initialize
from Model.mopso.p_objective import convert_to_number
from Model.mospsa.loss import evacuate_fabric_metric, evacuate_fabric_metric_prediction_model


class SPSA:
    """
    An optimizer class that implements Simultaneous Perturbation Stochastic Approximation (SPSA)
    """

    def __init__(self, a, c, A, alpha, gamma, loss_function):
        # Initialize gain parameters and decay factors
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.loss_function = loss_function
        # self.evaluate_function = evaluate_function

        # counters
        self.t = 0

    def step(self, current_estimate, *args):
        """
        :param current_estimate: This is the current estimate of the parameter vector
        :return: returns the updated estimate of the vector
        """

        # get the current values for gain sequences
        a_t = self.a / (self.t + 1 + self.A) ** self.alpha
        c_t = self.c / (self.t + 1) ** self.gamma

        # get the random perturbation vector from bernoulli distribution
        # it has to be symmetric around zero
        # But normal distribution does not work which makes the perturbations close to zero
        # Also, uniform distribution should not be used since they are not around zero
        delta = np.random.randint(0, 2, current_estimate.shape) * 2 - 1

        # measure the loss function at perturbations
        loss_plus = self.loss_function(current_estimate + delta * c_t, args)
        loss_minus = self.loss_function(current_estimate - delta * c_t, args)

        # compute the estimate of the gradient
        g_t = (loss_plus - loss_minus) / (2.0 * delta * c_t)

        # update the estimate of the parameter
        current_estimate = current_estimate - a_t * g_t

        # increment the counter
        self.t += 1

        return current_estimate


def spsa_maximize_optimize(initial_params, rho, num_iterations, a, c, A, alpha, gamma, low, high):
    params = initial_params.tolist()
    low = low.tolist()
    high = high.tolist()
    params[0] = (params[0] - low[0]) * (20 - 1) / (high[0] - low[0]) + 1
    params[1] = (params[1] - low[1]) * (20 - 1) / (high[1] - low[1]) + 1
    params[0], params[1] = params[1], params[0]
    initial_params__normalized = params
    params[0] = 10
    params[1] = 10
    delta = np.random.randint(0, 2, 2) * 2 - 1
    best_objective = 0
    best_params = params
    objective_values = []
    archive_param = []
    param_ledger_count = []
    param_ledger_preferred = []
    metric_ledger = []
    # create & modify & query & open & query & transfer
    # payload_function = 'transfer'
    t = 0
    k = 0
    rho = 1
    g_t_history = []
    while True:
        c_t = c / (t + 1) ** gamma
        objective_original = evacuate_fabric_metric_prediction_model(params, rho=rho)
        if objective_original > best_objective:
            best_params = params
        param_plus = params + delta * c_t
        param_minus = params - delta * c_t
        for i in range(param_plus.size):
            if param_plus[i] < 1 or param_plus[i] > 20:
                param_plus[i] = best_params[i]
            if param_minus[i] < 1 or param_minus[i] > 20:
                param_minus[i] = best_params[i]
        param_plus_input = np.array(param_plus)
        param_minus_input = np.array(param_minus)

        # save param
        param_plus_input_s = np.array(param_plus)
        param_minus_input_s = np.array(param_minus)
        param_plus_input_s[0] = (param_plus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_minus_input_s[0] = (param_minus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_plus_input_s[1] = (param_plus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_minus_input_s[1] = (param_minus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_ledger_preferred.append(param_plus_input_s[0])
        param_ledger_preferred.append(param_minus_input_s[0])
        param_ledger_count.append(param_plus_input_s[1])
        param_ledger_count.append(param_minus_input_s[1])

        objective_plus = evacuate_fabric_metric_prediction_model(param_plus_input, rho=rho)
        objective_minus = evacuate_fabric_metric_prediction_model(param_minus_input, rho=rho)
        metric_ledger.append(objective_plus)
        metric_ledger.append(objective_minus)
        a_t = a / (t + 1 + A) ** alpha
        curr_objective = max(objective_minus, objective_plus)
        g_t = (objective_plus - objective_minus) / (2.0 * delta * c_t)
        g_t_mean = (np.array(g_t_history).sum() + g_t) / (len(g_t_history) + 1)
        g_t = g_t_mean
        g_t_history.append(g_t)
        print(f'Step {t}: curr_objective: {curr_objective} best_objective: {best_objective} params: {params}')
        params[1] = params[1] - a_t * g_t[1]
        params[0] = params[0] - a_t * g_t[0]
        objective_values.append(curr_objective)
        if curr_objective < best_objective:
            # continue
            # t = 0
            params = best_params
            a = 0.3 * a
            g_t_history = []
        else:
            if objective_plus > best_objective:
                best_objective = objective_plus
                best_params = param_plus
                # objective_values.append(objective_plus)
                archive_param.append(best_params)
            if objective_minus > best_objective:
                best_objective = objective_minus
                best_params = param_minus
                # objective_values.append(objective_minus)
                archive_param.append(best_params)
        t += 1
        k += 1
        rho += 0.1
        rho = min(rho, 2)
        if k > num_iterations:
            break
    return best_params, objective_values, best_objective, param_ledger_preferred, param_ledger_count, metric_ledger


def spsa_maximize(ssh_client, mysql_connection, config_parameters, objective, objective_2rd, initial_params,
                  rho, num_iterations, a, c, A, alpha, gamma, low, high,
                  ):
    params = initial_params.tolist()
    low = low.tolist()
    high = high.tolist()
    params[0] = (params[0] - low[0]) * (20 - 1) / (high[0] - low[0]) + 1
    params[1] = (params[1] - low[1]) * (20 - 1) / (high[1] - low[1]) + 1
    params[0], params[1] = params[1], params[0]
    params[0] = 10
    params[1] = 10
    delta = np.random.randint(0, 2, 2) * 2 - 1
    best_objective = 0
    best_params = params
    objective_values = []
    archive_param = []
    param_ledger_count = []
    param_ledger_preferred = []
    metric_ledger = []
    # create & modify & query & open & query & transfer
    # payload_function = 'transfer'
    t = 0
    k = 0
    rho = 1
    g_t_history = []
    while True:
        c_t = c / (t + 1) ** gamma
        objective_original = evacuate_fabric_metric_prediction_model(params, rho=rho)
        if objective_original > best_objective:
            best_params = params
        param_plus = params + delta * c_t
        param_minus = params - delta * c_t
        # for i in range(param_plus.size):
        #     if param_plus[i] < 1 or param_plus[i] > 20:
        #         param_plus[i] = best_params[i]
        #     if param_minus[i] < 1 or param_minus[i] > 20:
        #         param_minus[i] = best_params[i]
        param_plus = check_bound(param_plus)
        param_minus = check_bound(param_minus)
        param_plus_input = np.array(param_plus)
        param_minus_input = np.array(param_minus)

        # save param
        param_plus_input_s = np.array(param_plus)
        param_minus_input_s = np.array(param_minus)
        param_plus_input_s[0] = (param_plus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_minus_input_s[0] = (param_minus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_plus_input_s[1] = (param_plus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_minus_input_s[1] = (param_minus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_ledger_preferred.append(param_plus_input_s[0])
        param_ledger_preferred.append(param_minus_input_s[0])
        param_ledger_count.append(param_plus_input_s[1])
        param_ledger_count.append(param_minus_input_s[1])

        objective_plus = evacuate_fabric_metric_prediction_model(param_plus_input, rho=rho)
        objective_minus = evacuate_fabric_metric_prediction_model(param_minus_input, rho=rho)
        metric_ledger.append(objective_plus)
        metric_ledger.append(objective_minus)
        a_t = a / (t + 1 + A) ** alpha
        curr_objective = max(objective_minus, objective_plus)
        g_t = (objective_plus - objective_minus) / (2.0 * delta * c_t)
        g_t_mean = (np.array(g_t_history).sum() + g_t) / (len(g_t_history) + 1)
        # g_t = g_t_mean
        g_t_history.append(g_t)
        print(f'Step {t}: curr_objective: {curr_objective} best_objective: {best_objective} params: {params}')
        params[1] = params[1] - a_t * g_t[1]
        params[0] = params[0] - a_t * g_t[0]
        objective_values.append(curr_objective)
        best_objective = max(best_objective, curr_objective)
        if curr_objective < best_objective - 10:
            # continue
            # t = 0
            params = best_params
            g_t_history = []
            # a = 0.3 * a
        # else:
        #     if objective_plus > best_objective:
        #         best_objective = objective_plus
        #         best_params = param_plus
        #         # objective_values.append(objective_plus)
        #         archive_param.append(best_params)
        #     if objective_minus > best_objective:
        #         best_objective = objective_minus
        #         best_params = param_minus
        #         # objective_values.append(objective_minus)
        #         archive_param.append(best_params)
        t += 1
        k += 1
        rho += 0.1
        rho = min(rho, 2)
        if k > num_iterations:
            break
    return best_params, objective_values, best_objective


def spsa_maximize2(ssh_client, mysql_connection, config_parameters, objective, objective_2rd, initial_params,
                  rho, num_iterations, a, c, A, alpha, gamma, low, high,
                  ):
    params = initial_params.tolist()
    low = low.tolist()
    high = high.tolist()
    params[0] = (params[0] - low[0]) * (20 - 1) / (high[0] - low[0]) + 1
    params[1] = (params[1] - low[1]) * (20 - 1) / (high[1] - low[1]) + 1
    params[0], params[1] = params[1], params[0]
    params[0] = 10
    params[1] = 10
    delta = np.random.randint(0, 2, 2) * 2 - 1
    best_objective = 0
    best_params = params
    objective_values = []
    archive_param = []
    param_ledger_count = []
    param_ledger_preferred = []
    metric_ledger = []
    # create & modify & query & open & query & transfer
    # payload_function = 'transfer'
    t = 0
    k = 0
    rho = 1
    g_t_history = []
    while True:
        c_t = c / (t + 1) ** gamma
        objective_original = evacuate_fabric_metric_prediction_model(params, rho=rho)
        if objective_original > best_objective:
            best_params = params
        param_plus = params + delta * c_t
        param_minus = params - delta * c_t
        for i in range(param_plus.size):
            if param_plus[i] < 1 or param_plus[i] > 20:
                param_plus[i] = best_params[i]
            if param_minus[i] < 1 or param_minus[i] > 20:
                param_minus[i] = best_params[i]
        param_plus_input = np.array(param_plus)
        param_minus_input = np.array(param_minus)

        # save param
        param_plus_input_s = np.array(param_plus)
        param_minus_input_s = np.array(param_minus)
        param_plus_input_s[0] = (param_plus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_minus_input_s[0] = (param_minus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_plus_input_s[1] = (param_plus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_minus_input_s[1] = (param_minus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_ledger_preferred.append(param_plus_input_s[0])
        param_ledger_preferred.append(param_minus_input_s[0])
        param_ledger_count.append(param_plus_input_s[1])
        param_ledger_count.append(param_minus_input_s[1])

        objective_plus = evacuate_fabric_metric_prediction_model(param_plus_input, rho=rho)
        objective_minus = evacuate_fabric_metric_prediction_model(param_minus_input, rho=rho)
        metric_ledger.append(objective_plus)
        metric_ledger.append(objective_minus)
        a_t = a / (t + 1 + A) ** alpha
        curr_objective = max(objective_minus, objective_plus)
        g_t = (objective_plus - objective_minus) / (2.0 * delta * c_t)
        g_t_mean = (np.array(g_t_history).sum() + g_t) / (len(g_t_history) + 1)
        # g_t = g_t_mean
        g_t_history.append(g_t)
        print(f'Step {t}: curr_objective: {curr_objective} best_objective: {best_objective} params: {params}')
        params[1] = params[1] - a_t * g_t[1]
        params[0] = params[0] - a_t * g_t[0]
        objective_values.append(curr_objective)
        if curr_objective < best_objective:
            # continue
            # t = 0
            params = best_params
            a = 0.3 * a
        else:
            if objective_plus > best_objective:
                best_objective = objective_plus
                best_params = param_plus
                # objective_values.append(objective_plus)
                archive_param.append(best_params)
            if objective_minus > best_objective:
                best_objective = objective_minus
                best_params = param_minus
                # objective_values.append(objective_minus)
                archive_param.append(best_params)
        t += 1
        k += 1
        rho += 0.1
        rho = min(rho, 2)
        if k > num_iterations:
            break
    return best_params, objective_values, best_objective


def get_random_input():
    lower_self = [1, 1]
    upper_self = [20, 20]
    return np.random.uniform(lower_self, upper_self)


def check_bound(param_plus):
    if param_plus[0] < 1 or param_plus[1] > 20:
        return get_random_input()
    if param_plus[0] < 1:
        param_plus[0] = 10
    if param_plus[1] > 20:
        param_plus[1] = 10
    return param_plus


def spsa_maximize3(ssh_client, mysql_connection, config_parameters, objective, objective_2rd, initial_params,
                  rho, num_iterations, a, c, A, alpha, gamma, low, high,
                  ):
    params = initial_params.tolist()
    low = low.tolist()
    high = high.tolist()
    params[0] = (params[0] - low[0]) * (20 - 1) / (high[0] - low[0]) + 1
    params[1] = (params[1] - low[1]) * (20 - 1) / (high[1] - low[1]) + 1
    params[0], params[1] = params[1], params[0]
    params[0] = 10
    params[1] = 10
    delta = np.random.randint(0, 2, 2) * 2 - 1
    best_objective = 0
    best_params = params
    objective_values = []
    archive_param = []
    param_ledger_count = []
    param_ledger_preferred = []
    metric_ledger = []
    # create & modify & query & open & query & transfer
    # payload_function = 'transfer'
    t = 0
    k = 0
    rho = 1
    g_t_history = []
    while True:
        c_t = c / (t + 1) ** gamma
        objective_original = evacuate_fabric_metric_prediction_model(params, rho=rho)
        if objective_original > best_objective:
            best_params = params
        param_plus = params + delta * c_t
        param_minus = params - delta * c_t
        # for i in range(param_plus.size):
        #     if param_plus[i] < 1 or param_plus[i] > 20:
        #         param_plus[i] = best_params[i]
        #     if param_minus[i] < 1 or param_minus[i] > 20:
        #         param_minus[i] = best_params[i]
        param_plus = check_bound(param_plus)
        param_minus = check_bound(param_minus)
        param_plus_input = np.array(param_plus)
        param_minus_input = np.array(param_minus)

        # save param
        param_plus_input_s = np.array(param_plus)
        param_minus_input_s = np.array(param_minus)
        param_plus_input_s[0] = (param_plus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_minus_input_s[0] = (param_minus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_plus_input_s[1] = (param_plus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_minus_input_s[1] = (param_minus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_ledger_preferred.append(param_plus_input_s[0])
        param_ledger_preferred.append(param_minus_input_s[0])
        param_ledger_count.append(param_plus_input_s[1])
        param_ledger_count.append(param_minus_input_s[1])

        objective_plus = evacuate_fabric_metric_prediction_model(param_plus_input, rho=rho)
        objective_minus = evacuate_fabric_metric_prediction_model(param_minus_input, rho=rho)
        metric_ledger.append(objective_plus)
        metric_ledger.append(objective_minus)
        a_t = a / (t + 1 + A) ** alpha
        curr_objective = max(objective_minus, objective_plus)
        g_t = (objective_plus - objective_minus) / (2.0 * delta * c_t)
        g_t_mean = (np.array(g_t_history).sum() + g_t) / (len(g_t_history) + 1)
        g_t = g_t_mean
        g_t_history.append(g_t)
        print(f'Step {t}: curr_objective: {curr_objective} best_objective: {best_objective} params: {params}')
        params[1] = params[1] - a_t * g_t[1]
        params[0] = params[0] - a_t * g_t[0]
        objective_values.append(curr_objective)
        best_objective = max(best_objective, curr_objective)
        if curr_objective < best_objective - 10:
            # continue
            # t = 0
            params = best_params
            g_t_history = []
            # a = 0.3 * a
        # else:
        #     if objective_plus > best_objective:
        #         best_objective = objective_plus
        #         best_params = param_plus
        #         # objective_values.append(objective_plus)
        #         archive_param.append(best_params)
        #     if objective_minus > best_objective:
        #         best_objective = objective_minus
        #         best_params = param_minus
        #         # objective_values.append(objective_minus)
        #         archive_param.append(best_params)
        t += 1
        k += 1
        rho += 0.1
        rho = min(rho, 2)
        if k > num_iterations:
            break
    return best_params, objective_values, best_objective


def adam_maximize(objective, initial_params, low, high, learning_rate=0.1, beta1=0.9, beta2=0.999,
                  epsilon=1e-8, iterations=100, c=1.0, gamma=0.1):
    # 初始化参数和变量
    x = initial_params
    # x = np.random.uniform(-10, 10, size=(2,))  # 初始参数随机取值
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    delta = np.random.randint(0, 2, initial_params.shape) * 2 - 1

    for i in range(iterations):
        t += 1
        c_t = c / (t + 1) ** gamma
        x_normalized = min_max_scaling(x, low, high)
        param_plus = x_normalized + delta * c_t
        param_minus = x_normalized - delta * c_t
        param_plus = reverse_min_max_scaling(param_plus, low, high)
        param_minus = reverse_min_max_scaling(param_minus, low, high)
        objective_plus = objective(param_plus)
        objective_minus = objective(param_minus)
        grad = (objective_plus - objective_minus) / (2.0 * delta * c_t)
        # grad = np.gradient(objective, x)  # 计算梯度
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        print(f"Step {i + 1}: f(x) = {objective(x)}")
    return x


def min_max_scaling(x, low, high):
    x_normalized = (x - low) / (high - low)
    return x_normalized


def reverse_min_max_scaling(x_normalized, low, high):
    return x_normalized * (high - low) + low


def get_hlf_boundary_2metirc():
    param_range = initialize.read_yaml_config(
        '../../Benchmark_Deploy_Tool/param_range.yaml')
    contained_col = ['Orderer_BatchSize_PreferredMaxBytes', 'Orderer_BatchSize_MaxMessageCount']
    boundary = pd.DataFrame(columns=['Name', 'Lower', 'Upper'], index=range(len(contained_col)))
    boundary['Name'] = boundary['Name'].astype(str)
    boundary['Lower'] = boundary['Lower'].astype(float)
    boundary['Upper'] = boundary['Upper'].astype(float)
    idx = 0

    # Order:6, Configtx:4, Peer:48
    for param_type in ['Peer', 'Orderer', 'Configtx']:
        for k, v in param_range['Parameters'][param_type].items():
            if k in contained_col:
                lower = v['lower']
                upper = v['upper']
                lower_value, unit = convert_to_number(str(lower))
                upper_value, unit = convert_to_number(str(upper))
                boundary.iloc[idx, 0] = k
                boundary.iloc[idx, 1] = lower_value
                boundary.iloc[idx, 2] = upper_value
                idx += 1
    # mask = boundary[:, 0] == 'peer_keepalive_minInterval'
    # print(boundary['peer_keepalive_minInterval'][0])
    return boundary


def moaspsa_maximize_optimize(initial_params, rho, num_iterations, a, c, A, alpha, gamma, low, high):
    params = initial_params.tolist()
    low = low.tolist()
    high = high.tolist()
    params[0] = (params[0] - low[0]) * (20 - 1) / (high[0] - low[0]) + 1
    params[1] = (params[1] - low[1]) * (20 - 1) / (high[1] - low[1]) + 1
    params[0], params[1] = params[1], params[0]
    initial_params__normalized = params
    params[0] = 10
    params[1] = 10
    delta = np.random.randint(0, 2, 2) * 2 - 1
    best_objective = 0
    best_params = params
    objective_values = []
    archive_param = []
    param_ledger_count = []
    param_ledger_preferred = []
    metric_ledger = []
    # create & modify & query & open & query & transfer
    # payload_function = 'transfer'
    t = 0
    k = 0
    g_t_history = []
    while True:
        c_t = c / (t + 1) ** gamma
        objective_original = evacuate_fabric_metric_prediction_model(params, rho=rho)
        if objective_original > best_objective:
            best_params = params
        param_plus = params + delta * c_t
        param_minus = params - delta * c_t
        for i in range(param_plus.size):
            if param_plus[i] < 1 or param_plus[i] > 20:
                param_plus[i] = best_params[i]
            if param_minus[i] < 1 or param_minus[i] > 20:
                param_minus[i] = best_params[i]
        param_plus_input = np.array(param_plus)
        param_minus_input = np.array(param_minus)

        # save param
        param_plus_input_s = np.array(param_plus)
        param_minus_input_s = np.array(param_minus)
        param_plus_input_s[0] = (param_plus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_minus_input_s[0] = (param_minus_input_s[0] - 1) * (100 - 1) / 19 + 1
        param_plus_input_s[1] = (param_plus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_minus_input_s[1] = (param_minus_input_s[1] - 1) * (1000 - 1) / 19 + 1
        param_ledger_preferred.append(param_plus_input_s[0])
        param_ledger_preferred.append(param_minus_input_s[0])
        param_ledger_count.append(param_plus_input_s[1])
        param_ledger_count.append(param_minus_input_s[1])

        objective_plus = evacuate_fabric_metric_prediction_model(param_plus_input, rho=rho)
        objective_minus = evacuate_fabric_metric_prediction_model(param_minus_input, rho=rho)
        metric_ledger.append(objective_plus)
        metric_ledger.append(objective_minus)
        a_t = a / (t + 1 + A) ** alpha
        curr_objective = max(objective_minus, objective_plus)
        g_t = (objective_plus - objective_minus) / (2.0 * delta * c_t)
        g_t_mean = (np.array(g_t_history).sum() + g_t) / (len(g_t_history) + 1)
        g_t = g_t_mean
        g_t_history.append(g_t)
        print(f'Step {t}: curr_objective: {curr_objective} best_objective: {best_objective} params: {params}')
        params[1] = params[1] - a_t * g_t[1]
        params[0] = params[0] - a_t * g_t[0]
        objective_values.append(curr_objective)
        if curr_objective < best_objective:
            params = best_params
            a = 0.3 * a
            g_t_history = []
        else:
            if objective_plus > best_objective:
                best_objective = objective_plus
                best_params = param_plus
                archive_param.append(best_params)
            if objective_minus > best_objective:
                best_objective = objective_minus
                best_params = param_minus
                archive_param.append(best_params)
        t += 1
        k += 1
        rho += 0.1
        rho = min(rho, 2)
        if k > num_iterations:
            break
    return best_params, objective_values, best_objective, param_ledger_preferred, param_ledger_count, metric_ledger