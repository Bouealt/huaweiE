# -*- coding: utf-8 -*-
# @Author  : Yang Chen
# @Time    : 21/9/2024 下午2:43


import cv2

# 初始化变量以存储参考点
ref_points = []

# 定义鼠标点击事件的回调函数
def click_event(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append((x, y))  # 保存点击的点
        print(f"参考点坐标: {x}, {y}")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 在图像上画一个圆表示参考点
        cv2.putText(frame, f'{len(ref_points)}', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

# 打开视频文件
cap = cv2.VideoCapture('E:\\grade2\\huaweiE\\ultralytics\\data\\32.31.250.103\\move1.mp4')

# 读取第一帧并显示
ret, frame = cap.read()
if not ret:
    print("无法读取视频")
    exit()

# 显示当前帧并设置鼠标回调函数
cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", click_event)

# 提示用户如何操作
print("请在图像上点击车道的参考点，按 'n' 切换下一帧，按 'q' 退出并保存参考点")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按 'q' 键退出
        break
    elif key == ord('n'):  # 按 'n' 键切换到下一帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取下一帧，视频结束")
            break
        cv2.imshow("Frame", frame)

# 打印保存的参考点
print(f"车道参考点坐标: {ref_points}")

# 可选：保存参考点到文本文件
with open('reference_points.txt', 'w') as f:
    for point in ref_points:
        f.write(f"{point[0]}, {point[1]}\n")
    print("参考点已保存到 reference_points.txt")

# 释放视频资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
