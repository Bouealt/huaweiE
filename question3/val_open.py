import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import joblib  # 导入用于加载已保存的模型

# Step 1: 读取数据
map_data_file = 'test.xlsx'  # 假设文件路径
map_data = pd.read_excel(map_data_file)

# 加载预训练的核岭回归模型
model = joblib.load('kernel_ridge_regression_model.pkl')

# Step 2: 初始化存储密度和代价函数的列表
densities = []
costs = []
adjusted_speeds = []  # 存储调整后的速度

# 设置初始的最大容量
initial_capacity = 35

# Step 3: 逐行计算密度和代价函数，并根据条件启用应急车道
for index, row in map_data.iterrows():
    flow = row['Normal Lane Flow']
    speed = row['Average Speed (km/h)']

    # 判断是否启用应急车道（流量大于25且车速小于40时启用）
    if flow >= 25 and speed <= 40:
        # 启用应急车道，最大容量变为1.5倍
        adjusted_capacity = initial_capacity * 1.5

        # 计算新的车速：使用模型，根据新的流量/容量比例来预测速度
        normalized_flow = flow / adjusted_capacity  # 标准化车流量
        predicted_speed = model.predict([[normalized_flow]])[0]  # 预测速度
        adjusted_speeds.append(predicted_speed)

        # 使用新的速度来计算密度和代价函数
        if predicted_speed > 0:
            density = flow / predicted_speed
            weight = 1 / predicted_speed
        else:
            density = float('inf')  # 速度为0时，密度无限大
            weight = float('inf')  # 速度为0时，权重无限大

    else:
        # 未启用应急车道，使用原始速度和初始容量
        adjusted_capacity = initial_capacity
        predicted_speed = speed  # 保持原始速度
        adjusted_speeds.append(predicted_speed)

        # 使用原始速度来计算密度和代价函数
        if speed > 0:
            density = flow / speed
            weight = 1 / speed
        else:
            density = float('inf')  # 速度为0时，密度无限大
            weight = float('inf')  # 速度为0时，权重无限大

    # 计算代价函数，Cost = Weight × Flow^2
    cost = weight * flow ** 2

    # 存储结果
    densities.append(density)
    costs.append(cost)

# 将计算出的密度、代价函数和调整后的速度添加到数据框中
map_data['Density'] = densities
map_data['Cost'] = costs
map_data['Adjusted Speed'] = adjusted_speeds

# 打印拟合出的全局参数
avg_normal_flow = map_data['Normal Lane Flow'].mean()  # 普通车道流量的平均值
avg_speed = map_data['Average Speed (km/h)'].mean()  # 车速的平均值

print(f"拟合参数:\n平均普通车道流量 = {avg_normal_flow}, 平均速度 = {avg_speed}")

# Step 4: 构建网络模型并设置参数
G = nx.DiGraph()

# 假设网络节点A（入口）到节点B（出口）
G.add_nodes_from(['A', 'B'])

# 边的容量为最大车道流量，权重为全局平均速度
capacity = initial_capacity  # 原始最大容量
weight = 1 / avg_speed if avg_speed > 0 else float('inf')  # 速度越高，权重越低

# 在网络中添加边
G.add_edge('A', 'B', capacity=capacity, weight=weight)

# 可视化网络及其拟合后的参数
pos = {'A': (0, 0), 'B': (1, 0)}  # 手动设置节点位置，使其水平排列
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
labels = nx.get_edge_attributes(G, 'capacity')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# Step 5: 保存计算结果
map_data.to_excel('output_with_emergency_lane.xlsx', index=False)
