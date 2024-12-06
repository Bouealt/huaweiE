import cv2
import numpy as np

# 定义图像中的 4 个参考点 (x, y) 对应的像素坐标
# 定义检测区域的四个参考点 (x, y) 对应的像素坐标，表示检测区域的四边形
detection_area = np.array([
    [9, 256],  # 第一个点的像素坐标
    [329, 352],  # 第二个点的像素坐标
    [551, 216],  # 第三个点的像素坐标
    [345, 166]  # 第四个点的像素坐标
], dtype="float32")

# 对应的实际物理坐标（单位：米，假设这些点之间的距离是已知的）
real_world_points = np.array([
    [0, 0],  # 第一个点在现实世界中的坐标
    [7.5, 0],  # 第二个点在现实世界中的坐标
    [7.5, 45],  # 第三个点在现实世界中的坐标
    [0, 45]  # 第四个点在现实世界中的坐标
], dtype="float32")

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(detection_area, real_world_points)

# 应用透视变换，将图像坐标转为实际物理坐标
def convert_to_real_world(x, y):
    point = np.array([[x, y]], dtype="float32")
    point = np.array([point])
    real_world_point = cv2.perspectiveTransform(point, matrix)
    return real_world_point[0][0]

# 计算每个像素在现实世界中对应的距离变化
def calculate_pixel_to_meter_step(step=1):
    # 假设在 image_points 范围内，以左上角作为起点 (381, 395)
    start_pixel = (381, 395)
    # 假设移动 step 个像素，比如水平或垂直方向上的移动
    moved_pixel_x = (start_pixel[0] + step, start_pixel[1])  # 向右移动 step 个像素
    moved_pixel_y = (start_pixel[0], start_pixel[1] + step)  # 向下移动 step 个像素

    # 将这些像素转换为现实世界坐标
    start_real = convert_to_real_world(start_pixel[0], start_pixel[1])
    moved_real_x = convert_to_real_world(moved_pixel_x[0], moved_pixel_x[1])
    moved_real_y = convert_to_real_world(moved_pixel_y[0], moved_pixel_y[1])

    # 计算移动 step 个像素在现实世界中的对应距离
    distance_per_pixel_x = np.sqrt((moved_real_x[0] - start_real[0]) ** 2 + (moved_real_x[1] - start_real[1]) ** 2)
    distance_per_pixel_y = np.sqrt((moved_real_y[0] - start_real[0]) ** 2 + (moved_real_y[1] - start_real[1]) ** 2)

    print(f"每水平移动 {step} 像素，现实世界中移动的距离: {distance_per_pixel_x:.4f} 米")
    print(f"每垂直移动 {step} 像素，现实世界中移动的距离: {distance_per_pixel_y:.4f} 米")

# 调用函数计算每移动 1 像素的物理距离
calculate_pixel_to_meter_step(step=1)
