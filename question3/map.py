import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: 读取map_data和data_latest数据
map_data_file = 'map.xlsx'  # 假设文件路径

# 读取文件中的数据
map_data = pd.read_excel(map_data_file)

# Step 2: 计算全局特性
avg_normal_flow = map_data['Normal Lane Flow'].mean()  # 普通车道流量的平均值
max_normal_flow = map_data['Normal Lane Flow'].max()   # 最大普通车道流量

avg_speed = map_data['Average Speed (km/h)'].mean()  # 车速的平均值
min_speed = map_data['Average Speed (km/h)'].min()   # 最小车速

# 打印拟合出的全局参数
print(f"拟合参数:\n平均普通车道流量 = {avg_normal_flow}, 最大普通车道流量 = {max_normal_flow}")
print(f"平均速度 = {avg_speed}, 最小速度 = {min_speed}")

# Step 3: 构建网络模型并设置参数
G = nx.DiGraph()

# 假设网络节点A（入口）到节点B（出口）
G.add_nodes_from(['A', 'B'])

# 边的容量为拟合的最大车道流量，权重为速度
capacity = max_normal_flow  # 将最大流量作为容量
weight = 1 / avg_speed if avg_speed > 0 else float('inf')  # 速度越高，权重越低

# 在网络中添加边
G.add_edge('A', 'B', capacity=capacity, weight=weight)

# 可视化网络及其拟合后的参数
pos = {'A': (0, 0), 'B': (1, 0)}  # 手动设置节点位置，使其水平排列
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
labels = nx.get_edge_attributes(G, 'capacity')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()
