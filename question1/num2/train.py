import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import os


# 定义函数：数据准备
def prepare_data(file_path, window_size):
    data = pd.read_excel(file_path)

    # 准备特征和目标
    features = data[['Traffic Density (vehicles/m)', 'Normal Lane Flow', 'Average Speed (km/h)']].values
    target = data['Congestion_Index'].values

    # 使用滑动窗口创建时序特征和目标数据
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])  # 过去 window_size 个时间步的特征
        y.append(target[i + window_size])  # 预测下一个时间步的拥堵指数

    X = np.array(X)
    y = np.array(y)

    # 归一化特征
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler


# 定义 Transformer 模型
class TransformerTrafficModel(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, output_dim):
        super(TransformerTrafficModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)  # 将输入的维度映射到 d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, output_dim)  # 输出层，将 d_model 映射为 1

        # 添加一个全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # 将输入通过线性层，映射到 d_model 维度
        x = self.input_fc(x)
        # 传入 Transformer 编码器
        x = self.transformer_encoder(x)
        # 全局池化以得到 (batch_size, d_model) 形状的输出
        x = x.permute(0, 2, 1)  # 交换维度以适应池化层 (batch_size, d_model, seq_length)
        x = self.global_avg_pool(x).squeeze(-1)  # 通过全局平均池化，并去掉多余的维度
        # 输出预测值
        x = self.fc(x)
        return x

# Step 1: 训练模型的函数并保存模型
def train_model(X_train, y_train, X_test, y_test, model, log_file, model_file, learning_rate=0.001, epochs=50):
    # 转换数据为 tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建保存日志的文件
    if os.path.exists(log_file):
        os.remove(log_file)  # 如果文件已存在，先删除它

    # 训练循环
    with open(log_file, "a") as log:
        log.write("Epoch, Train Loss, Test Loss, Train RMSE, Test RMSE, Train MAE, Test MAE, Train R2, Test R2\n")  # 写入标题行

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # 前向传播
            predictions = model(X_train_tensor)

            # 计算损失
            train_loss = criterion(predictions, y_train_tensor)

            # 反向传播
            train_loss.backward()
            optimizer.step()

            # 每个 epoch 评估模型在测试集上的表现
            model.eval()
            with torch.no_grad():
                predictions_test = model(X_test_tensor)

                # 转换为 numpy 数组以计算其他性能指标
                predictions_train_numpy = predictions.detach().cpu().numpy()
                predictions_test_numpy = predictions_test.detach().cpu().numpy()

                # 计算测试损失
                test_loss = criterion(predictions_test, y_test_tensor)

                # 计算其他性能指标
                train_rmse = np.sqrt(mean_squared_error(y_train, predictions_train_numpy))
                test_rmse = np.sqrt(mean_squared_error(y_test, predictions_test_numpy))

                train_mae = mean_absolute_error(y_train, predictions_train_numpy)
                test_mae = mean_absolute_error(y_test, predictions_test_numpy)

                train_r2 = r2_score(y_train, predictions_train_numpy)
                test_r2 = r2_score(y_test, predictions_test_numpy)

            # 将结果写入日志文件
            log.write(f"{epoch + 1},{train_loss.item():.4f},{test_loss.item():.4f},"
                      f"{train_rmse:.4f},{test_rmse:.4f},{train_mae:.4f},{test_mae:.4f},"
                      f"{train_r2:.4f},{test_r2:.4f}\n")

            # 打印日志
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, "
                      f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, "
                      f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}, "
                      f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

    # 保存模型
    torch.save(model.state_dict(), model_file)
    print(f"模型已保存到: {model_file}")

# Step 2: 训练流程

# 第一次训练
# 读取并处理监控点1的数据
window_size = 3
X_train_1, X_test_1, y_train_1, y_test_1, scaler = prepare_data('107.xlsx', window_size)

# 初始化 Transformer 模型
input_dim = X_train_1.shape[2]  # 输入特征的维度
d_model = 64  # Transformer 模型的维度
n_heads = 4  # Multi-head Attention 头数
n_layers = 2  # Transformer 层数
output_dim = 1  # 输出预测值（拥堵指数）

# 初始化模型
model = TransformerTrafficModel(input_dim, d_model, n_heads, n_layers, output_dim)

# 训练模型，使用监控点1的数据（第一次训练）
train_model(X_train_1, y_train_1, X_test_1, y_test_1, model, 'training_log_point1.txt', 'model_point1.pth')

# 第二次训练（增量训练）
# 读取并处理监控点2的数据
X_train_2, X_test_2, y_train_2, y_test_2, scaler = prepare_data('105.xlsx', window_size)

# 加载第一次训练好的模型
model.load_state_dict(torch.load('model_point1.pth'))  # 加载第一次训练后的模型

# 使用监控点2的数据继续训练（增量训练）
train_model(X_train_2, y_train_2, X_test_2, y_test_2, model, 'training_log_point2.txt', 'model_point2.pth')
