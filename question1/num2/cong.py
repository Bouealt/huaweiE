import pandas as pd

# 读取数据文件 (替换为你本地的文件路径)
file_path = '../data/108.xlsx'
data = pd.read_excel(file_path)

# 拥堵指数的计算公式:
# γ^p(t) = (occ^p(t)^2 * 1000) / (Q^p(t) * v^p(t) + 0.00001)
# 假设 'Traffic Density (vehicles/m)' 为 occ^p(t) (占有率),
# 'Normal Lane Flow' 为 Q^p(t) (交通流量),
# 'Average Speed (km/h)' 为 v^p(t) (速度).

# 计算拥堵指数
data['Congestion_Index'] = (data['Traffic Density (vehicles/m)']**2 * 10) / (data['Normal Lane Flow'] * data['Average Speed (km/h)'] + 0.00001)

# 保存结果到新文件
output_file_path = '108.xlsx'
data.to_excel(output_file_path, index=False)

print(f"计算完成，结果已保存到 {output_file_path}")
