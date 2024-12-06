import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates  # 新增导入日期格式化库
import glob
import os

# 读取多个Excel文件并合并
path = 'E:/重邮/竞赛/研二/华为杯/赛题/E题/数据/11/draw1/108/'  # 替换为您的文件路径
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

# 视频的开始时间
start_time = datetime.strptime('11:35:43', '%H:%M:%S')  # 替换为您的视频开始时间

# 计算时间t：t = frame / 33 * 10，单位是秒
df['Relative Time (seconds)'] = df['Frame'] / 33 * 10

# 函数：将相对时间（秒）转换为绝对时间（时间戳）
def calculate_absolute_time(relative_time_in_seconds):
    return start_time + timedelta(seconds=relative_time_in_seconds)

# 计算绝对时间并将其添加到 DataFrame 中
df['Time'] = df['Relative Time (seconds)'].apply(calculate_absolute_time)

# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制车速的线图（蓝色线，左侧纵轴）
ax1.plot(df['Time'], df['Average Speed (km/h)'], label='Speed (km/h)', color='#947ee3', linewidth=2)
ax1.set_xlabel('Time', fontsize=15)
ax1.set_ylabel('Speed (km/h)', fontsize=15, color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 创建第二个纵轴用于应急车道占有率
ax2 = ax1.twinx()
ax2.set_ylabel('Emergency Lane Occupancy Rate (%)', fontsize=15, color='#24b5db')
ax2.scatter(df['Time'], df['Emergency Lane Flow Rate'], s=df['Emergency Lane Flow Rate']*100,
            color='#ea152d', alpha=0.8, label='Emergency Lane Occupancy Rate (%)')
ax2.tick_params(axis='y', labelcolor='#24b5db')
ax2.set_ylim(0, 3)  # 设置应急车道占有率 Y 轴的上下限为 0 到 10

# 设置图例和标题
plt.title('Speed and Emergency Lane Occupancy Rate Over Time（点位3）', fontsize=18)

# 调整 X 轴显示时间
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 将X轴设置为小时和分钟格式
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 设置时间间隔为1小时
fig.autofmt_xdate()  # 自动调整X轴日期标签

# 显示图表
plt.tight_layout()
plt.savefig('点位3时间、应急车道、速度关系图.png')
plt.show()
