import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 读取指定的 Excel 文件
file_path = './103_1_with_speed_density.xlsx'
df = pd.read_excel(file_path)

# 确保 matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 假设 Excel 中有“Frame”和“P”（拥堵概率）的列
# 计算时间t：t = frame / 25 * 10，单位是秒
df['Relative Time (seconds)'] = df['Frame'] / 25 * 10

# 视频开始时间
start_time = datetime.strptime('12:56:47', '%H:%M:%S')

# 函数：将秒数转换为分钟和秒，并加到初始时间
def calculate_absolute_time(relative_time_in_seconds):
    return start_time + timedelta(seconds=relative_time_in_seconds)

# 计算绝对时间并将其存储为时间格式
df['Absolute Time'] = df['Relative Time (seconds)'].apply(calculate_absolute_time)

# 绘制拥堵概率随时间变化的折线图
plt.figure(figsize=(10, 6))

# 绘制折线图，横轴是时间，纵轴是拥堵概率 P
plt.plot(df['Absolute Time'], df['P'], color='#f81e3e', label='拥堵概率')

# 设置图表标题和坐标轴标签
plt.title('中断概率随时间变化图', fontsize=16)
plt.xlabel('时间', fontsize=14)
plt.ylabel('中断概率', fontsize=14)

# 格式化 X 轴时间显示
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

# 添加图例
plt.legend()

# 显示图像
plt.tight_layout()
plt.savefig('中断概率随时间变化图.png')
plt.show()
