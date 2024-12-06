import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# 数据准备
def prepare_data_ml(file_path, lag=1):
    data = pd.read_excel(file_path)

    # 创建滞后特征
    data['Lagged_Index'] = data['Congestion_Index'].shift(lag)

    # 删除缺失值
    data.dropna(inplace=True)

    # 目标变量和特征
    target = data['Congestion_Index']
    features = data[['Lagged_Index', 'Traffic Density (vehicles/m)', 'Normal Lane Flow', 'Average Speed (km/h)']]

    return features, target


# 训练机器学习模型
def train_ml_model(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


# 进行预测
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    return train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2


# 保存结果到txt文件
def save_results(log_file, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2, model_params):
    with open(log_file, 'w', encoding='utf-8') as log:  # 指定UTF-8编码
        log.write("Metric, Train, Test\n")
        log.write(f"RMSE, {train_rmse:.4f}, {test_rmse:.4f}\n")
        log.write(f"MAE, {train_mae:.4f}, {test_mae:.4f}\n")
        log.write(f"R², {train_r2:.4f}, {test_r2:.4f}\n")
        log.write("\nModel Parameters:\n")
        for param, value in model_params.items():
            log.write(f"{param}: {value}\n")


# 使用数据
features, target = prepare_data_ml('107.xlsx')

# 将目标变量分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 训练机器学习模型
n_estimators = 100
max_depth = None
rf_model = train_ml_model(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)

# 评估模型
train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2 = evaluate_model(rf_model, X_train, y_train, X_test,
                                                                               y_test)

# 保存结果
log_file = 'random_forest_training_log.txt'
model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
save_results(log_file, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2, model_params)

# 打印结果
print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')
print(f'Train MAE: {train_mae}, Test MAE: {test_mae}')
print(f'Train R²: {train_r2}, Test R²: {test_r2}')

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 数据准备
def prepare_data_arima(file_path, lag=1):
    data = pd.read_excel(file_path)

    # 创建滞后特征
    data['Lagged_Index'] = data['Congestion_Index'].shift(lag)

    # 删除缺失值
    data.dropna(inplace=True)

    # 目标变量和特征
    target = data['Congestion_Index']
    features = data[
        ['Lagged_Index', 'Traffic Density (vehicles/m)', 'Normal Lane Flow', 'Average Speed (km/h)']].dropna()

    return features, target


# 训练ARIMA模型
def train_arima_model(data, order=(1, 1, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit


# 进行预测
def evaluate_model(model, train_data, test_data):
    predictions = model.forecast(steps=len(test_data))
    train_predictions = model.predict(start=0, end=len(train_data) - 1, dynamic=False, typ='levels')

    train_rmse = mean_squared_error(train_data, train_predictions, squared=False)
    test_rmse = mean_squared_error(test_data, predictions, squared=False)
    train_mae = mean_absolute_error(train_data, train_predictions)
    test_mae = mean_absolute_error(test_data, predictions)
    train_r2 = r2_score(train_data, train_predictions)
    test_r2 = r2_score(test_data, predictions)

    return train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2


# 保存结果到txt文件
def save_results(log_file, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2):
    with open(log_file, 'w') as log:
        log.write("Metric, Train, Test\n")
        log.write(f"RMSE, {train_rmse:.4f}, {test_rmse:.4f}\n")
        log.write(f"MAE, {train_mae:.4f}, {test_mae:.4f}\n")
        log.write(f"R², {train_r2:.4f}, {test_r2:.4f}\n")


# 使用数据
features, target = prepare_data_arima('107.xlsx')

# 将目标变量分为训练集和测试集
train_size = int(len(target) * 0.8)
train, test = target[:train_size], target[train_size:]

# 训练ARIMA模型
arima_model = train_arima_model(train)

# 评估模型
train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2 = evaluate_model(arima_model, train, test)

# 保存结果
log_file = 'arima_training_log.txt'
save_results(log_file, train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2)

# 打印结果
print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')
print(f'Train MAE: {train_mae}, Test MAE: {test_mae}')
print(f'Train R²: {train_r2}, Test R²: {test_r2}')