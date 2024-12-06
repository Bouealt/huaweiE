import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: 读取数据
map_data_file = 'test.xlsx'  # 假设文件路径
map_data = pd.read_excel(map_data_file)

# Step 2: 初始化存储密度和代价函数的列表
densities = []
costs = []

# Step 3: 逐行计算密度和代价函数
for index, row in map_data.iterrows():
    flow = row['Normal Lane Flow']
    speed = row['Average Speed (km/h)']

    # 计算密度，密度 = 流量 / 速度 (注意速度不能为零)
    if speed > 0:
        density = flow / speed
    else:
        density = float('inf')  # 速度为0时，密度无限大

    # 计算代价函数，Cost = Weight × Flow^2
    weight = 1 / speed if speed > 0 else float('inf')  # 速度越高，权重越低
    cost = weight * flow ** 2

    # 存储结果
    densities.append(density)
    costs.append(cost)

# 将计算出的密度和代价函数添加到数据框中
map_data['Density'] = densities
map_data['Cost'] = costs

# 打印拟合出的全局参数
avg_normal_flow = map_data['Normal Lane Flow'].mean()  # 普通车道流量的平均值
avg_speed = map_data['Average Speed (km/h)'].mean()  # 车速的平均值

print(f"拟合参数:\n平均普通车道流量 = {avg_normal_flow}, 平均速度 = {avg_speed}")

# Step 4: 构建网络模型并设置参数
G = nx.DiGraph()

# 假设网络节点A（入口）到节点B（出口）
G.add_nodes_from(['A', 'B'])

# 边的容量为最大车道流量，权重为全局平均速度
capacity = 35  # 将最大流量作为容量
weight = 1 / avg_speed if avg_speed > 0 else float('inf')  # 速度越高，权重越低

# 在网络中添加边
G.add_edge('A', 'B', capacity=capacity, weight=weight)

# 可视化网络及其拟合后的参数
pos = {'A': (0, 0), 'B': (1, 0)}  # 手动设置节点位置，使其水平排列
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
labels = nx.get_edge_attributes(G, 'capacity')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# 可选：保存计算结果
map_data.to_excel('output_with_density_and_cost.xlsx', index=False)
