import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Peer param
        self.hidden1 = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU()
        )
        # Order param
        self.hidden2 = nn.Sequential(
            nn.Linear(30, 7),
            nn.ReLU()
        )
        # Peer param, blockcutter and broadcast about tx
        self.hidden3 = nn.Sequential(
            nn.Linear(47, 20),
            nn.ReLU()
        )
        self.output = nn.Linear(20, 3)

    def forward(self, x, extra_params1):
        out_hidden1 = self.hidden1(x)
        combined_input2 = torch.cat((out_hidden1, extra_params1), dim=1)
        out_hidden2 = self.hidden2(combined_input2)
        combined_input3 = torch.cat((out_hidden2, x), dim=1)
        out_hidden3 = self.hidden3(combined_input3)
        output = self.output(out_hidden3)
        return output


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted_output, true_targets, second_hidden_output):
        mse_loss = nn.MSELoss()
        num_target_parameters = 3
        loss = mse_loss(second_hidden_output[:, :num_target_parameters], true_targets)
        return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RegressionModel().to(device)
custom_loss = CustomLoss().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
input_data = torch.randn(1000, 40).to(device)
extra_parameters = torch.ones(1000, 10).to(device)
true_targets = torch.randn(1000, 3).to(device)  # 真实的目标值
output = model(input_data, extra_parameters)
loss = custom_loss(output, true_targets, output[:, -3:])  # 假设最后三个参数是与目标相关的参数
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(loss)
print(output)
