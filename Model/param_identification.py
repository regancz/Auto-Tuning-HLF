from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def feature_selection(parameter_data, performance_data, alpha=0.1, method='lasso', sort_method='feature_importance'):
    # 数据归一化
    scaler = StandardScaler()
    parameter_data_normalized = scaler.fit_transform(parameter_data)

    # 应用 LASSO 模型进行特征选择
    if method == 'lasso':
        model = Lasso(alpha=alpha)
        model.fit(parameter_data_normalized, performance_data)

        # 获取选择后的特征索引
        selected_features_indices = np.where(model.coef_ != 0)[0]
        selected_features = [f"Parameter_{i + 1}" for i in selected_features_indices]

        # 获取选择后的特征数据
        selected_parameter_data = parameter_data[:, selected_features_indices]

        # 使用增量方法对参数进行排序
        if sort_method == 'feature_importance':
            # 示例中使用 LASSO 的系数作为特征重要性评估
            feature_importance = np.abs(model.coef_)
            sorted_indices = np.argsort(feature_importance)[::-1]
            # 返回排序后的特征重要性
            sorted_features = [selected_features[idx] for idx in sorted_indices]
            sorted_importance = [feature_importance[idx] for idx in sorted_indices]
            return sorted_features, sorted_importance

        # 其他排序方法的实现可以在这里添加

        return selected_parameter_data, selected_features

    else:
        # 其他特征选择方法的实现可以在这里添加
        pass


def lasso_test(parameter_rows, performance_rows):
    df1 = pd.DataFrame(parameter_rows)
    df2 = pd.DataFrame(performance_rows)
    frames = [df1, df2]
    df = pd.concat(frames, axis=1)
    scaler = StandardScaler()
    df_sc = scaler.fit_transform(df)
    df_sc = pd.DataFrame(df_sc, columns=df.columns)
    y = df_sc['throughput']
    X = df_sc.drop('throughput', axis=1)  # becareful inplace= False
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    alpha_lasso = 10 ** np.linspace(-3, 1, 100)
    lasso = Lasso()
    coefs_lasso = []

    for i in alpha_lasso:
        lasso.set_params(alpha=i)
        lasso.fit(X_train, y_train)
        coefs_lasso.append(lasso.coef_)

    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.plot(alpha_lasso, coefs_lasso)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('weights: scaled coefficients')
    plt.title('Lasso regression coefficients Vs. alpha')
    plt.legend(df.drop('throughput', axis=1, inplace=False).columns)
    plt.show()

    lasso = Lasso(alpha=10 ** (-2))
    model_lasso = lasso.fit(X_train, y_train)
    coef = pd.Series(model_lasso.coef_, index=X_train.columns)
    print(coef[coef != 0].abs().sort_values(ascending=False))
    fea = X_train.columns
    a = pd.DataFrame()
    a['feature'] = fea
    a['importance'] = coef.values

    a = a.sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 8))
    plt.barh(a['feature'], a['importance'])
    plt.title('the importance features')
    plt.show()