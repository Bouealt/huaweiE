import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


# 加载模型的函数
def load_model(model_file, input_dim, d_model, n_heads, n_layers, output_dim):
    model = TransformerTrafficModel(input_dim, d_model, n_heads, n_layers, output_dim)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model


# 定义 Transformer 模型 (与训练时保持一致)
class TransformerTrafficModel(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, output_dim):
        super(TransformerTrafficModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


# 定义预测函数
def predict_with_model(model, input_data, scaler):
    # 预处理输入数据（归一化）
    input_data = scaler.transform(input_data)
    input_data_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    # 进行预测
    with torch.no_grad():
        prediction = model(input_data_tensor)

    return prediction.item()


# 进行文档预测
def predict_for_file(file_path, window_size, model_file, scaler, output_file):
    # 读取 Excel 文件中的数据
    data = pd.read_excel(file_path)

    # 提取特征数据
    features = data[['Traffic Density (vehicles/m)', 'Normal Lane Flow', 'Average Speed (km/h)']].values

    # 初始化模型参数（保持与训练一致）
    input_dim = features.shape[1]  # 输入特征维度 (3 个特征)
    d_model = 64  # Transformer 模型维度
    n_heads = 4  # 多头注意力头数
    n_layers = 2  # Transformer 层数
    output_dim = 1  # 输出预测值

    # 加载已经训练好的模型
    model = load_model(model_file, input_dim, d_model, n_heads, n_layers, output_dim)

    # 创建滑动窗口并进行预测
    predictions = []
    for i in range(len(features) - window_size):
        input_data = features[i:i + window_size]
        predicted_value = predict_with_model(model, input_data, scaler)
        predictions.append(predicted_value)

    # 将预测结果填充到原始数据的新列中
    # 在前 window_size 行中填充 NaN（因为无法预测前几个时间步）
    data['Predicted Congestion_Index'] = [np.nan] * window_size + predictions

    # 保存结果到新的 Excel 文件
    data.to_excel(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")


# 重新拟合 MinMaxScaler
def fit_scaler(training_file, window_size):
    data = pd.read_excel(training_file)
    features = data[['Traffic Density (vehicles/m)', 'Normal Lane Flow', 'Average Speed (km/h)']].values

    # 拟合 MinMaxScaler，确保与训练时的归一化方式一致
    scaler = MinMaxScaler()
    scaler.fit(features)  # 使用整个特征数据进行拟合
    return scaler


# 示例使用
file_path = '103.xlsx'  # 输入文件
model_file = 'model_point2.pth'  # 已保存的模型文件
output_file = 'predicted_103.xlsx'  # 输出文件
window_size = 3  # 滑动窗口大小

# 重新拟合 scaler（使用训练时的数据进行拟合）
scaler = fit_scaler('107.xlsx', window_size)

# 调用预测函数
predict_for_file(file_path, window_size, model_file, scaler, output_file)
