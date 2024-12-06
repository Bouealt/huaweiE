import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. 数据预处理
file_path = 'data8.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# Select relevant columns
data_selected = data[['Normal Lane Flow', 'Average Speed (km/h)', 'P']]

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Define sliding window
sequence_length = 10
forecast_horizon = 3


def create_sequences(data, sequence_length, forecast_horizon):
    X, y_forward, y_backward = [], [], []
    for i in range(sequence_length, len(data) - forecast_horizon):
        X.append(data[i - sequence_length:i])  # Input sequence
        y_forward.append(data[i:i + forecast_horizon])  # Future values
        y_backward.append(data[i - sequence_length:i - sequence_length + forecast_horizon])  # Past values
    return np.array(X), np.array(y_forward), np.array(y_backward)


X, y_forward, y_backward = create_sequences(data_scaled, sequence_length, forecast_horizon)

# Split data into training and test sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_forward_train, y_forward_test = y_forward[:split_index], y_forward[split_index:]
y_backward_train, y_backward_test = y_backward[:split_index], y_backward[split_index:]

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_forward_train_tensor = torch.tensor(y_forward_train, dtype=torch.float32)
y_backward_train_tensor = torch.tensor(y_backward_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_forward_test_tensor = torch.tensor(y_forward_test, dtype=torch.float32)
y_backward_test_tensor = torch.tensor(y_backward_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_forward_train_tensor, y_backward_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 2. 定义双向 TCN 模型
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(TemporalBlock, self).__init__()

        # 计算合适的 padding 来保持时间维度不变
        padding = (kernel_size - 1) * dilation

        # 卷积1：保持时间维度不变
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()

        # 卷积2：保持时间维度不变
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()

        # 1x1卷积，确保跳跃连接中的通道数一致
        self.net = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # 输入通过两次卷积操作
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))

        # 打印输入和输出的形状，确保时间维度匹配
        # print(f"x shape: {x.shape}, out shape: {out.shape}")

        # 添加跳跃连接，确保时间维度一致
        return out[:, :, :x.shape[2]] + self.net(x)  # 裁剪掉多余的时间步长


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 调用 TemporalBlock，不再传递 padding
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)




class BidirectionalTCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(BidirectionalTCN, self).__init__()
        self.forward_tcn = TCN(num_inputs, num_channels, kernel_size)
        self.backward_tcn = TCN(num_inputs, num_channels, kernel_size)
        self.final_layer = nn.Linear(num_channels[-1] * 2, num_inputs * forecast_horizon)

    def forward(self, x):
        forward_out = self.forward_tcn(x)
        backward_out = self.backward_tcn(torch.flip(x, [2]))  # Flip along time dimension
        out = torch.cat([forward_out[:, :, -1], backward_out[:, :, -1]], dim=1)
        return self.final_layer(out).view(out.size(0), forecast_horizon, -1)


# 3. 模型训练和保存
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
num_channels = [16, 32, 64]  # Number of filters in each layer
model = BidirectionalTCN(num_inputs=3, num_channels=num_channels).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lists to store losses and metrics
train_losses = []
test_losses_forward = []
test_losses_backward = []
rmse_forward_list = []
rmse_backward_list = []

# Define a function to compute RMSE
def compute_rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()

# Training loop
epochs = 50
model_save_path = "bidirectional_tcn_model_8.pth"
results_file_path = "training_results_8.csv"

with open(results_file_path, "w") as f:
    f.write("Epoch,Train Loss,Test Forward Loss,Test Backward Loss,Test Forward RMSE,Test Backward RMSE\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_forward_batch, y_backward_batch in train_loader:
            X_batch, y_forward_batch, y_backward_batch = X_batch.to(device), y_forward_batch.to(device), y_backward_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch.transpose(1, 2))  # Switch dimensions to (batch_size, num_features, sequence_length)
            loss_forward = criterion(output, y_forward_batch)
            loss_backward = criterion(output, y_backward_batch)
            loss = loss_forward + loss_backward
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            y_forward_test_tensor = y_forward_test_tensor.to(device)
            y_backward_test_tensor = y_backward_test_tensor.to(device)

            predictions = model(X_test_tensor.transpose(1, 2))
            test_loss_forward = criterion(predictions, y_forward_test_tensor)
            test_loss_backward = criterion(predictions, y_backward_test_tensor)

            rmse_forward = compute_rmse(predictions, y_forward_test_tensor)
            rmse_backward = compute_rmse(predictions, y_backward_test_tensor)

            test_losses_forward.append(test_loss_forward.item())
            test_losses_backward.append(test_loss_backward.item())
            rmse_forward_list.append(rmse_forward)
            rmse_backward_list.append(rmse_backward)

            # Save metrics to file
            f.write(f"{epoch+1},{train_loss / len(train_loader)},{test_loss_forward.item()},{test_loss_backward.item()},{rmse_forward},{rmse_backward}\n")

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, '
              f'Test Forward Loss: {test_loss_forward.item()}, Test Backward Loss: {test_loss_backward.item()}, '
              f'RMSE Forward: {rmse_forward}, RMSE Backward: {rmse_backward}')

# Save the model
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

print(f"Results saved to {results_file_path}")
