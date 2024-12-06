import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 用于绘制热力图
from datetime import datetime, timedelta
import glob
import os
import matplotlib.dates as mdates

# 读取多个Excel文件并合并
path = 'E:/重邮/竞赛/研二/华为杯/赛题/E题/数据/11/draw1/'  # 替换为您的文件路径
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

# 计算时间t：t = frame / 33 * 10，单位是秒
df['Relative Time (seconds)'] = df['Frame'] / 33 * 10

# 视频开始时间
start_time = datetime.strptime('11:35:43', '%H:%M:%S')

# 函数：将秒数转换为分钟和秒，并加到初始时间
def calculate_absolute_time(relative_time_in_seconds):
    return start_time + timedelta(seconds=relative_time_in_seconds)

# 计算绝对时间
df['Absolute Time'] = df['Relative Time (seconds)'].apply(calculate_absolute_time)

# 计算车流总量
df['Total Flow'] = df['Normal Lane Flow'] + df['Emergency Lane Flow']

# 相关性分析
# 选择数值型列进行相关性分析
df_numeric = df[['Normal Lane Flow', 'Emergency Lane Flow', 'Total Flow',
                 'Average Speed (km/h)', 'Traffic Density (vehicles/m)',
                 'Normal Lane Flow Rate', 'Emergency Lane Flow Rate']]

# 计算相关系数矩阵
correlation_matrix = df_numeric.corr()

# 打印相关性矩阵
# print(correlation_matrix)


# 绘制热力图来展示相关性
plt.figure(figsize=(20, 16))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 16})
plt.title('各因素相关性分析',fontsize=25)
# plt.legend(fontsize=16)
plt.xticks(fontsize=14, rotation=45)  # X轴刻度标签字体大小
plt.yticks(fontsize=14, rotation=45)  # Y轴刻度标签字体大小
plt.tight_layout(pad=1.0)  # 设置自动边距，pad 为间距
plt.savefig('总体热力图.png')
plt.show()
