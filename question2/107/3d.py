import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import glob
import os
from sklearn.preprocessing import MinMaxScaler

# 读取多个Excel文件并合并
path = 'E:/重邮/竞赛/研二/华为杯/赛题/E题/数据/11/draw1/107/'  # 替换为您的文件路径
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

# 对 "Traffic Density (vehicles/m)" 列进行归一化
scaler = MinMaxScaler()
df['Traffic Density (vehicles/m)'] = scaler.fit_transform(df[['Traffic Density (vehicles/m)']])

# 将归一化后的数据保存到新的 Excel 文件中
output_path = 'E:/重邮/竞赛/研二/华为杯/赛题/E题/数据/11/draw1/107/归一化结果.xlsx'  # 保存路径
df.to_excel(output_path, index=False)
print(f"归一化后的数据已保存到 {output_path}")

# 计算时间t：t = frame / 33 * 10，单位是秒
df['Relative Time (seconds)'] = df['Frame'] / 33 * 10

# 视频开始时间
start_time = datetime.strptime('11:41:03', '%H:%M:%S')

# 函数：将秒数转换为分钟和秒，并加到初始时间
def calculate_absolute_time(relative_time_in_seconds):
    # 计算绝对时间
    return start_time + timedelta(seconds=relative_time_in_seconds)

# 计算绝对时间并将其存储为时间格式
df['Absolute Time'] = df['Relative Time (seconds)'].apply(calculate_absolute_time)

# 三维可视化：速度（Z轴），归一化密度（Y轴），绝对时间（X轴）
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')

# 使用 Absolute Time 作为 X 轴
x = df['Absolute Time']  # X轴: 绝对时间
y = df['Traffic Density (vehicles/m)']  # Y轴: 归一化后的车流密度
z = df['Average Speed (km/h)']  # Z轴: 车速

# 绘制三维图，使用不同的cmap配色
ax.plot_trisurf(mdates.date2num(x), y, z, cmap='magma', edgecolor='none')

# 设置轴标签
ax.set_xlabel('Time', fontsize=15)
ax.set_ylabel('Normalized Traffic Density (vehicles/m)', fontsize=15)
ax.set_zlabel('Average Speed (km/h)', fontsize=15)
ax.set_title('3D Visualization of Speed, Density, and Time（点位1）', fontsize=15)

# 设置X轴为时间格式，并自动定位时间刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置为小时和分钟显示
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 每小时显示一个刻度
plt.tight_layout(pad=1.0)  # 设置自动边距，pad 为间距
plt.savefig('点位1时间、速度、密度3D图_归一化.png')
plt.show()
