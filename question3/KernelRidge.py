import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # 导入用于保存模型的库

# 读取Excel文件
file_path = 'map.xlsx'  # 修改为实际的路径
df = pd.read_excel(file_path)

# 特征是 Normal Lane Flow / 35
X = (df['Normal Lane Flow'] / 35).values.reshape(-1, 1)
y = df['Average Speed (km/h)'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用核岭回归
model = KernelRidge(kernel='rbf', alpha=1.0)
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 可视化结果
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.scatter(X_test, y_pred, color='red', label='Predicted data')
plt.xlabel('Normalized Normal Lane Flow')
plt.ylabel('Average Speed (km/h)')
plt.title('Kernel Ridge Regression')
plt.legend()
plt.show()

# 打印评估指标
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 保存模型
joblib.dump(model, 'kernel_ridge_regression_model.pkl')

# 保存结果到CSV
results_df = pd.DataFrame({
    'Normalized Lane Flow': X_test.flatten(),
    'Actual Speed': y_test,
    'Predicted Speed': y_pred
})
results_df.to_csv('model_predictions.csv', index=False)

# 保存指标到文本文件
with open('model_performance.txt', 'w') as f:
    f.write(f"Mean Squared Error: {mse:.2f}\n")
    f.write(f"R^2 Score: {r2:.2f}\n")
