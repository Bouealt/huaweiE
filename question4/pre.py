# -*- coding: utf-8 -*-
# @Author  : Yang Chen
# @Time    : 25/9/2024 上午12:09
import torch
import pandas as pd
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os

# Define sliding window
sequence_length = 10
forecast_horizon = 3

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


# 检查是否有 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 定义模型架构 (与训练时保持一致)
num_channels = [16, 32, 64]  # 假设之前的模型通道数
model = BidirectionalTCN(num_inputs=3, num_channels=num_channels).to(device)  # 模型挂载到GPU

# 2. 加载已训练的模型权重
model.load_state_dict(torch.load('bidirectional_tcn_model_8.pth'))  # 加载模型
model.eval()  # 设置为评估模式

# 3. 从文件中加载测试数据
# 假设测试数据保存在 'test_data.csv' 文件中，并且包含与训练时相同的特征
test_data_path = 'data8.xlsx'  # 替换为你的测试数据文件路径
test_data = pd.read_excel(test_data_path)

# 假设测试数据的列为 ['Normal Lane Flow', 'Average Speed (km/h)', 'P']
test_data_selected = test_data[['Normal Lane Flow', 'Average Speed (km/h)', 'P']]

# 4. 数据标准化 (与训练时一致)
scaler = StandardScaler()
test_data_scaled = scaler.fit_transform(test_data_selected)

# 5. 将数据转换为 PyTorch 张量，并挂载到 GPU
test_data_tensor = torch.tensor(test_data_scaled, dtype=torch.float32).to(device)

# 6. 准备输入数据（使用测试数据的最后一个序列）
input_sequence = test_data_tensor[-1:, :]  # 获取最后一个时间序列
input_sequence = input_sequence.unsqueeze(0)  # 调整为 (1, 3, sequence_length) 格式
input_sequence = input_sequence.transpose(1, 2)  # 转置为 (1, sequence_length, 3)

# 7. 执行预测
forecast_horizon = 3  # 根据之前训练时设置的预测时间步数
with torch.no_grad():
    predicted_output = model(input_sequence)  # 预测未来的时间步
    predicted_output = predicted_output.cpu().numpy()  # 将结果从 GPU 移到 CPU 并转换为 NumPy 数组

# 8. 将预测结果保存为新的表格

# 假设要保存为 Excel 文件
output_path = 'predicted_output8.xlsx'  # 你可以修改保存路径

# 创建 DataFrame
predicted_df = pd.DataFrame(predicted_output[0], columns=['Normal Lane Flow', 'Average Speed (km/h)', 'P'])

# 保存到 Excel 文件
predicted_df.to_excel(output_path, index=False)  # 保存为 Excel 文件，不保存行索引

# 也可以选择保存为 CSV 文件
# predicted_df.to_csv('predicted_output.csv', index=False)

print(f"Predicted future {forecast_horizon} time steps have been saved to {output_path}")
