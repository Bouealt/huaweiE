import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import glob
import os
import matplotlib.dates as mdates

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

# 计算时间t：t = frame / 25 * 10，单位是秒
df['Relative Time (seconds)'] = df['Frame'] / 33 * 10

# 视频开始时间
start_time = datetime.strptime('11:35:43', '%H:%M:%S')

# 函数：将秒数转换为分钟和秒，并加到初始时间
def calculate_absolute_time(relative_time_in_seconds):
    # 将相对时间转换为秒
    return start_time + timedelta(seconds=relative_time_in_seconds)

# 计算绝对时间
df['Absolute Time'] = df['Relative Time (seconds)'].apply(calculate_absolute_time)

# 计算车流总量
df['Total Flow'] = df['Normal Lane Flow'] + df['Emergency Lane Flow']

# 作图 - 使用堆叠图
plt.figure(figsize=(10, 6))

# 修改配色方案
colors = ['#34a2cb', '#ff9103']  # 使用更青春的颜色

# 绘制堆叠图
plt.stackplot(df['Absolute Time'], df['Normal Lane Flow'], df['Emergency Lane Flow'],
              labels=['Normal Lane Flow', 'Emergency Lane Flow'], colors=colors, alpha=0.8)

# 设置图例和标题
plt.title('32.31.250.108点位3交通流变化情况', fontweight='bold',fontsize=18)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Traffic Flow (vehicles)',fontsize=15)

# 设置x轴时间格式为24小时制，并加上半小时的时间刻度
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # 使用24小时制显示时间
# plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=30))  # 每30分钟显示一个刻度
plt.gcf().autofmt_xdate()  # 自动调整日期格式以防止重叠
# 设置刻度标签的字体大小
plt.xticks(fontsize=15)  # X轴刻度标签字体大小
plt.yticks(fontsize=15)  # Y轴刻度标签字体大小
plt.legend(loc='upper right',fontsize=15)

# 显示图像
plt.tight_layout()
# 如果您需要保存图像，可以使用以下代码
plt.savefig('点位3交通流随时间变化情况.png')

plt.show()

