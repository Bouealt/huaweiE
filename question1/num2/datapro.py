# -*- coding: utf-8 -*-
# @Author  : Yang Chen
# @Time    : 23/9/2024 下午8:20

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据文件
file_path = '103.xlsx'
data = pd.read_excel(file_path)

# 设置时序窗口大小
window_size = 5  # 使用过去5个时间步的数据来预测下一个时间步的拥堵指数

# 准备特征和目标
features = data[['Traffic Density (vehicles/m)', 'Normal Lane Flow', 'Average Speed (km/h)']].values
target = data['Congestion_Index'].values

# 使用滑动窗口创建时序特征和目标数据
X, y = [], []
for i in range(len(features) - window_size):
    X.append(features[i:i+window_size])  # 过去 window_size 个时间步的特征
    y.append(target[i+window_size])      # 预测下一个时间步的拥堵指数

X = np.array(X)
y = np.array(y)

# 归一化特征
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
