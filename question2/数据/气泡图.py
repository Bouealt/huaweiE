import pandas as pd
import matplotlib.pyplot as plt

# 读取指定的Excel文件
file_path = 'E:\重邮\竞赛\研二\华为杯\赛题\E题\数据/11\draw1\数据/105_1_with_speed_density.xlsx'  # 替换为您的文件路径

# 读取Excel文件
df = pd.read_excel(file_path)

# 提取绘图所需的列
x = df['Normal Lane Flow']  # 常规车道流量
y = df['Average Speed (km/h)']  # 平均速度
p = df['P']  # 拥堵概率

# 创建气泡图
plt.figure(figsize=(10, 6))
bubble = plt.scatter(x, y, s=p*1000, c=p, cmap='viridis', alpha=0.5, edgecolor='k')  # 使用拥堵概率来调整气泡大小和颜色

# 设置图表标题和坐标轴标签
plt.title('Normal Lane Flow vs Average Speed with Congestion Probability', fontsize=15)
plt.xlabel('Normal Lane Flow (vehicles)', fontsize=12)
plt.ylabel('Average Speed (km/h)', fontsize=12)

# 添加颜色条表示拥堵概率的大小
cbar = plt.colorbar(bubble)
cbar.set_label('Congestion Probability (P)', fontsize=12)

# 显示图像
plt.tight_layout()
plt.savefig('气泡图.png')
plt.show()
