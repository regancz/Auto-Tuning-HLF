import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [list(map(float, line.split())) for line in lines]
    return np.array(data)


def determine_nondominated_solutions(data):
    num_alternatives = data.shape[0]
    non_dominated_indices = []

    for i in range(num_alternatives):
        is_nondominated = True
        for j in range(num_alternatives):
            if i != j:
                # 根据正向指标（例如，tps）和负向指标（例如，latency）的性质判断非支配关系
                if (data[i, -2] <= data[j, -2] and data[i, -1] > data[j, -1]) or \
                        (data[i, -2] < data[j, -2] and data[i, -1] >= data[j, -1]):
                    is_nondominated = False
                    break
        if is_nondominated:
            non_dominated_indices.append(i)

    return non_dominated_indices


def topsis(data):
    # 提取参数和性能指标
    parameters = data[:, :17]
    performance = data[:, -2:]

    # 归一化性能指标
    scaler = MinMaxScaler()
    normalized_performance = scaler.fit_transform(performance)

    # 计算正理想解和负理想解
    ideal_best = np.max(normalized_performance, axis=0)
    ideal_worst = np.min(normalized_performance, axis=0)

    # 计算各属性的权值（使用香农熵）
    attribute_weights = calculate_attribute_weights(performance)
    print('attribute_weights: ', attribute_weights)

    # 计算综合得分
    topsis_scores = calculate_topsis_scores(normalized_performance, ideal_best, ideal_worst, attribute_weights)

    # 将原始数据和综合得分合并，并按照综合得分进行排序
    sorted_data = np.column_stack((data, topsis_scores))
    sorted_data = sorted_data[sorted_data[:, -1].argsort()[::-1]]

    return sorted_data


def calculate_attribute_weights(parameters):
    num_attributes = parameters.shape[1]
    attribute_weights = []

    for i in range(num_attributes):
        # 计算香农熵
        entropy_value = entropy(parameters[:, i])

        # 计算属性权值
        attribute_weight = 1 - entropy_value
        attribute_weights.append(attribute_weight)

    # 归一化属性权值
    attribute_weights = np.array(attribute_weights) / np.sum(attribute_weights)

    return attribute_weights


def calculate_topsis_scores(normalized_performance, ideal_best, ideal_worst, attribute_weights):
    num_alternatives = normalized_performance.shape[0]
    topsis_scores = []

    for i in range(num_alternatives):
        # 计算正负理想解的欧氏距离
        d_best = np.linalg.norm(normalized_performance[i] - ideal_best)
        d_worst = np.linalg.norm(normalized_performance[i] - ideal_worst)

        # 计算综合得分
        # 计算综合得分（根据最大TPS和最小延迟进行调整）
        tps_score = 1 / (1 + d_best)  # 越大越好
        latency_score = 1 / (1 + d_worst)  # 越小越好

        # 考虑属性权值
        topsis_score = (tps_score * attribute_weights[0]) + (latency_score * attribute_weights[1])

        topsis_scores.append(topsis_score)

    return topsis_scores


# 从txt文件读取数据
file_path = './pareto_res.txt'
data = read_data_from_file(file_path)

# 确定非支配解的索引
non_dominated_indices = determine_nondominated_solutions(data)

# 使用TOPSIS计算综合得分
sorted_data = topsis(data[non_dominated_indices])

# 打印结果
print("TOPSIS Scores for Non-Dominated Solutions:", sorted_data)
