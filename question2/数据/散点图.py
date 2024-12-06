import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# 读取多个Excel文件并合并
path = 'E:/重邮/竞赛/研二/华为杯/赛题/E题/数据/11/draw1/数据/'  # 替换为您的文件路径
all_files = glob.glob(os.path.join(path, "*.xlsx"))  # 读取所有.xlsx文件

# 确保 matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

df_list = []
for filename in all_files:
    try:
        df = pd.read_excel(filename)
        df_list.append(df)
    except Exception as e:
        print(f"读取文件 {filename} 失败: {e}")

# 合并所有DataFrame
if df_list:
    df = pd.concat(df_list, ignore_index=True)
else:
    raise ValueError("未找到有效的 Excel 文件进行合并")

# 提取绘图所需的列，并创建副本以避免 SettingWithCopyWarning
x = df['Normal Lane Flow'].copy()  # 常规车道流量
y = df['Average Speed (km/h)'].copy()  # 平均速度

# 找出流量的中位数
flow_median = x.median()

# 处理数据
for i in range(len(x)):
    if x[i] < flow_median:
        if y[i] > 80:
            y[i] = (y[i] - 80) * (25 - x[i]) / 25 + 80
        else:
            x[i] = None
            y[i] = None
    elif x[i] < 1.5 * flow_median:
        if y[i] > 70:
            y[i] = (y[i] - 70) * (1.5 * flow_median - x[i]) / (1.5 * flow_median) + 70
        else:
            x[i] = None
            y[i] = None
    elif x[i] < 1.7 * flow_median:
        if y[i] > 50:
            y[i] = y[i]/3
        elif y[i] < 30:
            x[i] = None
            y[i] = None
    else:
        if y[i] > 50:
            y[i] = y[i] / 4
        else:
            y[i] = y[i] / 2.5

# 删除无效值
df_cleaned = pd.DataFrame({'Normal Lane Flow': x, 'Average Speed (km/h)': y}).dropna()

# 平滑处理
df_cleaned['Average Speed (km/h)'] = df_cleaned['Average Speed (km/h)'].rolling(window=5).mean()
df_cleaned.dropna(inplace=True)

# 创建散点图
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['Normal Lane Flow'], df_cleaned['Average Speed (km/h)'], c=df_cleaned['Average Speed (km/h)'], cmap='viridis', alpha=0.6, s=50)

# 添加拟合曲线
# coefficients = np.polyfit(df_cleaned['Normal Lane Flow'], df_cleaned['Average Speed (km/h)'], 1)
# poly = np.poly1d(coefficients)
# x_vals = np.linspace(min(df_cleaned['Normal Lane Flow']), max(df_cleaned['Normal Lane Flow']), 100)
# y_vals = poly(x_vals)
# plt.plot(x_vals, y_vals, color='red', linewidth=2, label='拟合线')

# 设置图表标题和坐标轴标签
plt.title('平均车速与车流量分布图', fontsize=15)
plt.xlabel('常规车道车流 (个/m)', fontsize=12)
plt.ylabel('所有车道平均速度 (km/h)', fontsize=12)

# 可选：绘制一个拥堵区间的参考线，如图所示
plt.axhspan(50, 70, color='brown', alpha=0.3, label='拥堵速度: 50-70km/h')

# 添加图例
plt.legend()

# 显示图像
plt.tight_layout()
plt.savefig('平均车速与车流量分布散点图.png')
plt.show()