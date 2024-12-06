import pandas as pd
import numpy as np

# 读取数据文件 (替换为你本地的文件路径)
file_path = '108.xlsx'
data = pd.read_excel(file_path)

# 计算四分位数来检测异常值
Q1 = data['Congestion_Index'].quantile(0.25)
Q3 = data['Congestion_Index'].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值的上下界
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 检测异常值，生成布尔值列
data['Is_Outlier'] = ~((data['Congestion_Index'] >= lower_bound) & (data['Congestion_Index'] <= upper_bound))

# 处理异常值，赋值为其附近非异常值的平均值
# 对异常值进行前向和后向填充以填补异常值
data['Congestion_Index_Filled'] = data['Congestion_Index']
data.loc[data['Is_Outlier'], 'Congestion_Index_Filled'] = np.nan  # 异常值设为NaN
data['Congestion_Index_Filled'].fillna(method='ffill', inplace=True)  # 前向填充
data['Congestion_Index_Filled'].fillna(method='bfill', inplace=True)  # 后向填充

# 获取过滤后（非异常）的最小值和最大值
min_congestion_filtered = data.loc[~data['Is_Outlier'], 'Congestion_Index'].min()
max_congestion_filtered = data.loc[~data['Is_Outlier'], 'Congestion_Index'].max()

# 将非异常值缩放到 1 到 10 之间
data['Scaled_Congestion_Index'] = data['Congestion_Index_Filled'].apply(
    lambda x: 1 + (x - min_congestion_filtered) * (9 / (max_congestion_filtered - min_congestion_filtered))
    if lower_bound <= x <= upper_bound else x)

# 保存结果到新文件
output_file_path = '108_1.xlsx'
data.to_excel(output_file_path, index=False)

print(f"处理完成，结果已保存到 {output_file_path}")
