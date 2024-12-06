# -*- coding: utf-8 -*-
# @Author  : Yang Chen
# @Time    : 21/9/2024 下午2:13

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker  # 导入 ByteTrack 跟踪器

#车流参数介绍
# 创建一个模拟的 args 对象，用来传递给 BYTETracker
class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5  # 跟踪置信度阈值
        self.track_buffer = 15  # 缩短缓冲时间，确保快速处理
        self.match_thresh = 0.9  # 增加匹配阈值以适应快速移动目标
        self.mot20 = False  # 是否使用 MOT20 进行优化


# 初始化 YOLOv8 模型
model = YOLO('yolov8s.pt')

# 创建 args 对象并初始化 ByteTrack 跟踪器
args = TrackerArgs()
tracker = BYTETracker(args)

# 打开视频文件
cap = cv2.VideoCapture('E:\\grade2\\huaweiE\\data\\32.31.250.103\\move1.mp4')

frame_count = 0

# 定义普通车道和应急车道的矩形检测区域 (x_min, y_min, x_max, y_max)
normal_lane_box = (410, 97, 549, 206)  # 普通车道矩形区域 (x_min, y_min, x_max, y_max)
emergency_lane_box = (474, 233, 636, 370)  # 应急车道矩形区域 (x_min, y_min, x_max, y_max)

# 定义检测区域的四个参考点 (x, y) 对应的像素坐标，表示检测区域的四边形
detection_area = np.array([
    [9, 256],  # 第一个点的像素坐标
    [329, 352],  # 第二个点的像素坐标
    [551, 216],  # 第三个点的像素坐标
    [345, 166]  # 第四个点的像素坐标
], dtype="float32")

# 获取四边形的最小和最大 y 值，用于范围判断
min_y_in_detection_area = np.min(detection_area[:, 1])  # y轴的最小值
max_y_in_detection_area = np.max(detection_area[:, 1])  # y轴的最大值

# 定义每像素的实际物理距离 (假设为水平和垂直方向)
horizontal_pixel_distance = 0.1371   # 每水平像素代表的实际距离（米）
vertical_pixel_distance = 0.2128  # 每垂直像素代表的实际距离（米）

# 假设检测区域的实际物理长度，单位为米
road_length_meters = 45  # 该区域的实际道路长度（米）

# 记录普通车道和应急车道的车辆数量
normal_lane_count = 0
emergency_lane_count = 0
current_frame_vehicle_count = 0  # 记录当前帧车辆数量

# 记录已经通过的车辆的 track_id
normal_lane_history = set()  # 记录普通车道已经计数的车辆ID
emergency_lane_history = set()  # 记录应急车道已经计数的车辆ID

# 记录每辆车的进入和离开位置信息，用于整体车流速度计算
vehicle_entry_positions = {}  # 记录车辆进入区域时的位置
vehicle_exit_positions = {}  # 记录车辆离开区域时的位置

# 定义普通车道和应急车道的进入和离开矩形框
normal_lane_entry_box = (48, 133, 338, 343)  # 普通车道进入矩形框 (x_min, y_min, x_max, y_max)
normal_lane_exit_box = (647, 66, 709, 131)   # 普通车道离开矩形框 (x_min, y_min, x_max, y_max)
emergency_lane_entry_box = (355, 401, 618, 607)  # 应急车道进入矩形框
emergency_lane_exit_box = (728, 105, 783, 154)   # 应急车道离开矩形框

# 记录车辆的进入和离开时间（帧号）
normal_lane_entry_times = {}  # 记录每辆车进入普通车道的时间（帧号）
normal_lane_exit_times = {}   # 记录每辆车离开普通车道的时间（帧号）
emergency_lane_entry_times = {}  # 记录每辆车进入应急车道的时间（帧号）
emergency_lane_exit_times = {}   # 记录每辆车离开应急车道的时间（帧号）

total_observation_time = 0  # 总观察时间，以秒为单位
fps = 33  # 假设视频帧率为30帧/秒，可以根据实际帧率调整

# 数据保存，新增车流速度和车流密度列
df = pd.DataFrame(columns=["Frame", "Normal Lane Flow", "Emergency Lane Flow", "Average Speed (km/h)",
                           "Traffic Density (vehicles/m)"])

# 车辆类别列表（可以扩展，如果你的模型支持更多类型）
vehicle_classes = [2, 5, 7]  # 假设 2=car, 5=bus, 7=truck，YOLOv8 的类别ID，具体类别可以根据模型定义进行调整
# 定义一个字典来记录车辆在上一帧的中心点位置
previous_positions = {}
# 记录累计100帧的总速度和车辆数量
cumulative_speed = 0  # 用于累计每帧的平均速度
total_vehicle_count = 0  # 用于累计100帧中有效车辆的数量

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 获取当前帧的图像尺寸，并确保传递的是元组 (height, width)
    img_h, img_w = frame.shape[:2]
    img_size = (img_h, img_w)  # 确保传递的是 (height, width) 的元组

    # 传递 img_info 作为一个包含 height 和 width 的列表
    img_info = [img_h, img_w]  # 使用列表代替字典，符合 ByteTrack 期望的格式

    # 使用 YOLOv8 进行车辆检测
    results = model(frame)

    # YOLOv8 检测结果转为 ByteTrack 输入格式，确保格式为 [x1, y1, x2, y2, score]
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf[0].item()
        cls = int(box.cls[0].item())  # 获取类别ID

        # 仅统计车辆类的目标（如car、bus、truck等）
        if cls in vehicle_classes:
            detections.append([x1, y1, x2, y2, confidence])

    # 将检测结果转为 numpy 数组
    detections = np.array(detections)

    # 如果没有检测到目标，继续下一帧
    if detections.shape[0] == 0:
        continue

    # 使用 ByteTrack 进行跟踪
    online_targets = tracker.update(detections, img_info, img_size)  # 传递 img_size 和 img_info 参数

    # 记录当前帧检测到的车辆数量和总速度
    valid_vehicle_count = 0
    total_speed = 0

    # 遍历 ByteTrack 跟踪的对象，获取 track_id 并记录车辆位置
    for target in online_targets:
        tlbr = target.tlbr  # 获取跟踪的边界框 [x1, y1, x2, y2]
        track_id = target.track_id  # 获取跟踪的唯一 ID
        x1, y1, x2, y2 = map(int, tlbr)  # 解包边界框的坐标，并确保是整数

        # 计算车辆中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 绘制车辆的中心点
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # 绿色圆圈标记车辆中心点

        # 在车辆上方显示车辆的ID号
        cv2.putText(frame, f"ID: {track_id}", (center_x - 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)

        # 普通车道的判断（车辆中心点进入普通车道的矩形框区域）
        normal_x_min, normal_y_min, normal_x_max, normal_y_max = normal_lane_box
        if normal_x_min <= center_x <= normal_x_max and normal_y_min <= center_y <= normal_y_max and track_id not in normal_lane_history:
            normal_lane_count += 1
            normal_lane_history.add(track_id)
            print(f"Normal Lane: Vehicle {track_id} passed")

        # 应急车道的判断（车辆中心点进入应急车道的矩形框区域）
        emergency_x_min, emergency_y_min, emergency_x_max, emergency_y_max = emergency_lane_box
        if emergency_x_min <= center_x <= emergency_x_max and emergency_y_min <= center_y <= emergency_y_max and track_id not in emergency_lane_history:
            emergency_lane_count += 1
            emergency_lane_history.add(track_id)
            print(f"Emergency Lane: Vehicle {track_id} passed")

        # 普通车道的进入和离开检测
        normal_x_min_e, normal_y_min_e, normal_x_max_e, normal_y_max_e = normal_lane_entry_box
        normal_x_min_l, normal_y_min_l, normal_x_max_l, normal_y_max_l = normal_lane_exit_box

        # 检测普通车道的进入
        if normal_x_min_e <= center_x <= normal_x_max_e and normal_y_min_e <= center_y <= normal_y_max_e:
            if track_id not in normal_lane_entry_times:
                normal_lane_entry_times[track_id] = frame_count  # 记录车辆进入的帧号

        # 检测普通车道的离开
        if normal_x_min_l <= center_x <= normal_x_max_l and normal_y_min_l <= center_y <= normal_y_max_l:
            normal_lane_exit_times[track_id] = frame_count  # 记录车辆离开的帧号

        # 应急车道的进入和离开检测
        emergency_x_min_e, emergency_y_min_e, emergency_x_max_e, emergency_y_max_e = emergency_lane_entry_box
        emergency_x_min_l, emergency_y_min_l, emergency_x_max_l, emergency_y_max_l = emergency_lane_exit_box

        # 检测应急车道的进入
        if emergency_x_min_e <= center_x <= emergency_x_max_e and emergency_y_min_e <= center_y <= emergency_y_max_e:
            if track_id not in emergency_lane_entry_times:
                emergency_lane_entry_times[track_id] = frame_count  # 记录车辆进入的帧号

        # 检测应急车道的离开
        if emergency_x_min_l <= center_x <= emergency_x_max_l and emergency_y_min_l <= center_y <= emergency_y_max_l:
            emergency_lane_exit_times[track_id] = frame_count  # 记录车辆离开的帧号

        # 检查车辆是否在检测区域的 y 轴范围内（即 min_y_in_detection_area <= center_y <= max_y_in_detection_area）
        if min_y_in_detection_area <= center_y <= max_y_in_detection_area:
            current_frame_vehicle_count += 1  # 增加车辆计数

        if cv2.pointPolygonTest(detection_area, (center_x, center_y), False) >= 0:
            # 检查车辆是否已经在 previous_positions 中
            if track_id in previous_positions:
                # 如果车辆已经在 previous_positions 中，计算它在当前帧与上一帧之间的速度
                prev_x, prev_y = previous_positions[track_id]

                # 计算水平和垂直位移
                horizontal_distance = abs(center_x - prev_x) * horizontal_pixel_distance
                vertical_distance = abs(center_y - prev_y) * vertical_pixel_distance

                # 计算总位移（米）
                total_distance = np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)

                # 假设每帧之间的时间间隔是 1/帧率 (例如 1/3.3 秒)
                time_interval = 1 / 3.3  # 帧率可以根据实际情况调整
                speed_mps = total_distance / time_interval  # 米/秒
                speed_kmph = speed_mps * 3.6  # 转换为 km/h

                # 输出当前车辆的速度信息
                print(f"Vehicle {track_id}: Speed = {speed_kmph:.2f} km/h")

                # 累加车辆速度
                total_speed += speed_kmph
                valid_vehicle_count += 1  # 计数

            # 更新该车辆的上一帧位置，无论是否是首次出现或已经存在
            previous_positions[track_id] = (center_x, center_y)

        # 计算平均速度
    if valid_vehicle_count > 0:
        average_speed = total_speed / valid_vehicle_count
        print(f"Average Speed: {average_speed:.2f} km/h")
    else:
        average_speed = 0
        print("No valid vehicles to calculate speed.")
    # 累计当前帧的平均速度
    cumulative_speed += average_speed
    total_vehicle_count += valid_vehicle_count

    # 计算车流密度 = 车辆数量 / 道路长度
    traffic_density = current_frame_vehicle_count / road_length_meters

    # # 绘制检测区域边界框（绿色线条）
    # cv2.polylines(frame, [detection_area.astype(int)], True, (0, 255, 0), 2)  # 绿色多边形检测区域
    # #
    # # 绘制普通车道和应急车道的矩形框
    # cv2.rectangle(frame, (normal_x_min, normal_y_min), (normal_x_max, normal_y_max), (0, 0, 255), 2)  # 红色矩形
    # cv2.rectangle(frame, (emergency_x_min, emergency_y_min), (emergency_x_max, emergency_y_max), (255, 0, 0), 2)  # 蓝色矩形

    # 绘制普通车道和应急车道的进入和离开矩形框
    cv2.rectangle(frame, (normal_x_min_e, normal_y_min_e), (normal_x_max_e, normal_y_max_e), (0, 255, 0), 2)  # 绿色进入矩形
    cv2.rectangle(frame, (normal_x_min_l, normal_y_min_l), (normal_x_max_l, normal_y_max_l), (0, 0, 255), 2)  # 红色离开矩形

    cv2.rectangle(frame, (emergency_x_min_e, emergency_y_min_e), (emergency_x_max_e, emergency_y_max_e), (0, 255, 0),
                  2)  # 绿色进入矩形
    cv2.rectangle(frame, (emergency_x_min_l, emergency_y_min_l), (emergency_x_max_l, emergency_y_max_l), (0, 0, 255),
                  2)  # 红色离开矩形

    # 显示普通车道和应急车道的当前车辆通过数量
    cv2.putText(frame, f"Normal Lane Flow: {normal_lane_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                2)
    cv2.putText(frame, f"Emergency Lane Flow: {emergency_lane_count}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    # 显示平均车流速度
    cv2.putText(frame, f"Avg Speed: {average_speed:.2f} km/h", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                2)

    # 显示当前帧
    cv2.imshow("Frame", frame)

    # 每 100 帧保存一次结果
    if frame_count % 100 == 0:
        if total_vehicle_count > 0:
            cumulative_average_speed = cumulative_speed / total_vehicle_count
        else:
            cumulative_average_speed = 0

        # 确保总观察时间不为零
        if total_observation_time > 0:
            # 计算普通车道的占有时间
            normal_lane_occupancy_time = sum(
                (normal_lane_exit_times[track_id] - normal_lane_entry_times[track_id]) / fps
                for track_id in normal_lane_entry_times if track_id in normal_lane_exit_times
            )

            # 计算应急车道的占有时间
            emergency_lane_occupancy_time = sum(
                (emergency_lane_exit_times[track_id] - emergency_lane_entry_times[track_id]) / fps
                for track_id in emergency_lane_entry_times if track_id in emergency_lane_exit_times
            )

            # 计算车道占有率（百分比）
            normal_lane_occupancy_rate = (normal_lane_occupancy_time / total_observation_time) * 100
            emergency_lane_occupancy_rate = (emergency_lane_occupancy_time / total_observation_time) * 100
        else:
            # 如果 observation_time 为 0，跳过占有率计算
            normal_lane_occupancy_rate = 0
            emergency_lane_occupancy_rate = 0

        row_data = {
            "Frame": frame_count,
            "Normal Lane Flow": normal_lane_count,
            "Emergency Lane Flow": emergency_lane_count,
            "Average Speed (km/h)": average_speed,
            "Traffic Density (vehicles/m)": traffic_density,
            "Normal Lane Occupancy (%)": normal_lane_occupancy_rate,
            "Emergency Lane Occupancy (%)": emergency_lane_occupancy_rate
        }
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        print(f"Frame {frame_count}: {row_data}")

        # 重置车辆计数和历史记录
        normal_lane_count = 0  # 每 100 帧重置普通车道的计数
        emergency_lane_count = 0  # 重置应急车道的计数
        current_frame_vehicle_count = 0  # 重置当前帧车辆计数
        normal_lane_history = set()  # 重置普通车道的历史记录
        emergency_lane_history = set()  # 重置应急车道的历史记录
        previous_positions.clear()  # 重置车辆位置信息
        vehicle_entry_positions.clear()  # 重置进入区域的历史位置
        vehicle_exit_positions.clear()  # 重置离开区域的历史位置
        # 重置进入和离开时间
        normal_lane_entry_times.clear()
        normal_lane_exit_times.clear()
        emergency_lane_entry_times.clear()
        emergency_lane_exit_times.clear()

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 保存统计结果到 Excel 文件
df.to_excel("103_1_with_speed_cong.xlsx", index=False)

# 释放资源
cap.release()
cv2.destroyAllWindows()
