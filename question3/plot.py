import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
file_path = 'compare.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 处理无穷大的情况：将 `inf` 替换为 `NaN`，避免在计算和绘图时出错
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# 计算每列的平均值，跳过NaN
avg_p_speed = data['P_Speed (km/h)'].mean()
avg_a_speed = data['A_Speed (km/h)'].mean()
avg_p_density = data['P_Density'].mean(skipna=True)
avg_a_density = data['A_Density'].mean(skipna=True)
avg_p_cost = data['P_Cost'].mean(skipna=True)
avg_a_cost = data['A_Cost'].mean(skipna=True)

# 输出平均值，检查是否合理
print(f"P_Speed Avg: {avg_p_speed}, A_Speed Avg: {avg_a_speed}")
print(f"P_Density Avg: {avg_p_density}, A_Density Avg: {avg_a_density}")
print(f"P_Cost Avg: {avg_p_cost}, A_Cost Avg: {avg_a_cost}")

# 设置图表大小和子图
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 获取X轴的最右边的值，用于标注位置
x_min, x_max = data['Frame'].min(), data['Frame'].max()

# 速度对比
axs[0].plot(data['Frame'], data['P_Speed (km/h)'], label='P_Speed (km/h)', color='#37B3C8')
axs[0].plot(data['Frame'], data['A_Speed (km/h)'], label='A_Speed (km/h)', color='#C84C37')
# 绘制虚线表示平均值
axs[0].axhline(avg_p_speed, color='#37B3C8', linestyle='--', label='P_Speed Avg')
axs[0].axhline(avg_a_speed, color='#C84C37', linestyle='--', label='A_Speed Avg')
# 在虚线旁边标注平均值，避免重叠
axs[0].text(x_max, avg_p_speed, f'{avg_p_speed:.2f}', color='#37B3C8', va='top')
axs[0].text(x_max, avg_a_speed, f'{avg_a_speed:.2f}', color='#C84C37', va='bottom')
axs[0].set_title('Speed Comparison')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Speed (km/h)')
axs[0].legend()

# 密度对比
axs[1].plot(data['Frame'], data['P_Density'], label='P_Density', color='#39C64F')
axs[1].plot(data['Frame'], data['A_Density'], label='A_Density', color='#C639B0')
# 绘制虚线表示平均值
axs[1].axhline(avg_p_density, color='#39C64F', linestyle='--', label='P_Density Avg')
axs[1].axhline(avg_a_density, color='#C639B0', linestyle='--', label='A_Density Avg')
# 标注平均值
axs[1].text(x_max, avg_p_density, f'{avg_p_density:.2f}', color='#39C64F', va='bottom')
axs[1].text(x_max, avg_a_density, f'{avg_a_density:.2f}', color='#C639B0', va='top')
axs[1].set_title('Density Comparison')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Density')
axs[1].set_ylim(0, max(data['P_Density'].max(), data['A_Density'].max()) * 1.2)  # 调整 y 轴范围
axs[1].legend()

# 代价函数对比
axs[2].plot(data['Frame'], data['P_Cost'], label='P_Cost', color='#7843BC')
axs[2].plot(data['Frame'], data['A_Cost'], label='A_Cost', color='#87BC43')
# 绘制虚线表示平均值
axs[2].axhline(avg_p_cost, color='#7843BC', linestyle='--', label='P_Cost Avg')
axs[2].axhline(avg_a_cost, color='#87BC43', linestyle='--', label='A_Cost Avg')
# 标注平均值
axs[2].text(x_max, avg_p_cost, f'{avg_p_cost:.2f}', color='#7843BC', va='bottom')
axs[2].text(x_max, avg_a_cost, f'{avg_a_cost:.2f}', color='#87BC43', va='top')
axs[2].set_title('Cost Comparison')
axs[2].set_xlabel('Frame')
axs[2].set_ylabel('Cost')
axs[2].set_ylim(0, max(data['P_Cost'].max(), data['A_Cost'].max()) * 1.2)  # 调整 y 轴范围
axs[2].legend()

# 调整布局并显示图像
plt.tight_layout()
plt.show()
