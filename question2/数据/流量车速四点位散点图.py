import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 创建一个映射，定义每个点位与对应的Excel文件
point_mapping = {
    "点位 1": ["107_1_with_speed_density.xlsx", "107_2_with_speed_density.xlsx"],
    "点位 2": ["105_1_with_speed_density.xlsx", "105_2_with_speed_density.xlsx", "105_3_with_speed_density.xlsx"],
    "点位 3": ["108_1_with_speed_density.xlsx", "108_2_with_speed_density.xlsx"],
    "点位 4": ["103_1_with_speed_density.xlsx", "103_2_with_speed_density.xlsx"]
}
# 确保 matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 设置文件路径
path = 'E:\重邮\竞赛\研二\华为杯\赛题\E题\数据/11\draw1\数据'

# 创建一个2x2的子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 遍历每个点位
for idx, (point, files) in enumerate(point_mapping.items()):
    # 获取子图的行和列索引
    row = idx // 2
    col = idx % 2

    # 合并点位的所有文件
    df_list = []
    for file in files:
        try:
            df = pd.read_excel(os.path.join(path, file))
            df_list.append(df)
        except Exception as e:
            print(f"读取文件 {file} 失败: {e}")

    # 合并所有数据
    if df_list:
        df = pd.concat(df_list, ignore_index=True)

    # 提取流量和车速
    flow = df['Traffic Density (vehicles/m)']  # 假设流量在这一列
    speed = df['Average Speed (km/h)']  # 假设车速在这一列

    # 绘制散点图
    axs[row, col].scatter(flow, speed, alpha=0.5, label=f'{point}')
    axs[row, col].set_title(f'{point} 散点图')
    axs[row, col].set_xlabel('交通车辆密度 (veh/m)')
    axs[row, col].set_ylabel('速度 (km/h)')
    axs[row, col].legend()

# 保存和显示图像
plt.savefig('四个点位散点图.png')
plt.show()
