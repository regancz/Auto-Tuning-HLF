import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, hidden1, hidden2, hidden3):
        loss = torch.mean(hidden1 + hidden2 + hidden3)  # 自定义损失
        return loss


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Sequential(
            # nn.BatchNorm1d(12, affine=False),
            nn.Linear(17, 1),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )
        self.hidden2 = nn.Sequential(
            # nn.BatchNorm1d(6, affine=False),
            nn.Linear(11, 1),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )
        self.hidden3 = nn.Sequential(
            # nn.BatchNorm1d(14, affine=False),
            nn.Linear(19, 1),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )

        self.output = nn.Linear(12, 1)

    # dim: 12, 5, 4 + 4
    def forward(self, peer_config_df, orderer_config_df, metric, bench_config_df):
        extra_config_df = orderer_config_df[['Orderer_BatchSize_PreferredMaxBytes',
                                             'Orderer_BatchSize_MaxMessageCount',
                                             'Orderer_General_Authentication_TimeWindow']]
        # peer_config = torch.tensor(peer_config_df.values)
        # orderer_config = torch.tensor(orderer_config_df.values)
        # extra_config = torch.tensor(extra_config_df.values)
        peer_config_tensor = torch.tensor(peer_config_df.values).float()
        orderer_config_tensor = torch.tensor(orderer_config_df.values).float()
        extra_config_tensor = torch.tensor(extra_config_df.values).float()
        bench_config_tensor = torch.tensor(bench_config_df.values).float()

        metric_df = metric[['gossip_state_commit_duration',
                            'broadcast_validate_duration',
                            'blockcutter_block_fill_duration',
                            'broadcast_enqueue_duration']]
        minmax_scaler = MinMaxScaler()
        metric_scaled = minmax_scaler.fit_transform(metric_df)
        metric_tensor = torch.tensor(metric_scaled).float()

        combined_input1 = torch.cat((bench_config_tensor, peer_config_tensor), dim=1)
        out_hidden1 = self.hidden1(combined_input1)
        combined_input2 = torch.cat((bench_config_tensor, out_hidden1, orderer_config_tensor), dim=1)
        out_hidden2 = self.hidden2(combined_input2)
        combined_input3 = torch.cat((bench_config_tensor, out_hidden1, out_hidden2, peer_config_tensor), dim=1)
        out_hidden3 = self.hidden3(combined_input3)
        combined_input4 = torch.cat((bench_config_tensor, out_hidden1, out_hidden2, out_hidden3, metric_tensor), dim=1)
        output = self.output(combined_input4)

        return output


def train_and_predict_with_metrics(peer_config, orderer_config, metric):
    # 选择特征和目标列
    features = pd.concat([peer_config, orderer_config, metric[['gossip_state_commit_duration',
                                                               'broadcast_validate_duration',
                                                               'blockcutter_block_fill_duration',
                                                               'broadcast_enqueue_duration']]], axis=1)
    target = np.log1p(metric['throughput'])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 对数据进行标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    minmax_scaler = MinMaxScaler()
    y_train_scaled = minmax_scaler.fit_transform(y_train.values.reshape(-1, 1))
    minmax_scaler.fit_transform(y_test.values.reshape(-1, 1))
    # 使用 XGBoost 模型
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train_scaled, y_train_scaled)
    xgb_predictions = xgb_model.predict(X_test_scaled)

    # 使用 SVR 模型
    svr_model = SVR()
    svr_model.fit(X_train_scaled, y_train)
    svr_predictions = svr_model.predict(X_test_scaled)

    # 计算 MSE 和 R-squared
    mse_xgb = mean_squared_error(y_test, np.expm1(xgb_predictions.reshape(-1, 1)))
    r2_xgb = r2_score(y_test, np.expm1(xgb_predictions.reshape(-1, 1)))

    mse_svr = mean_squared_error(y_test, np.expm1(svr_predictions.reshape(-1, 1)))
    r2_svr = r2_score(y_test, (svr_predictions.reshape(-1, 1)))

    return mse_xgb, r2_xgb, mse_svr, r2_svr
