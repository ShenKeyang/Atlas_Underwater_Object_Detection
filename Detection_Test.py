import cv2
from ultralytics import YOLO
import os

# 设置中文显示
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def detect_objects_in_video(video_path, model_path, output_path=None, conf_threshold=0.6):
    # 加载模型
    model = YOLO(model_path)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取原始视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 如果指定了输出路径，创建视频写入对象
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 使用原始宽度和高度创建VideoWriter对象
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 定义类别名称和对应的颜色（BGR格式）
    class_names = ['holothurian', 'echinus', 'scallop', 'starfish']
    class_colors = {
        0: (0, 165, 255),    # 海参：橙色
        1: (0, 255, 255),    # 海胆：黄色
        2: (255, 255, 255),  # 扇贝：白色
        3: (160, 120, 190),  # 海星：粉色
    }
    
    frame_count = 0
    
    while cap.isOpened():
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 目标检测
        results = model(frame, conf=conf_threshold)
        
        # 获取检测结果并手动绘制，应用置信度阈值
        annotated_frame = frame.copy()  # 复制原始帧
        
        # 手动过滤和绘制边界框
        boxes = results[0].boxes  # 获取边界框信息
        for box in boxes:
            conf = float(box.conf)  # 获取置信度
            if conf >= conf_threshold:  # 确保置信度高于阈值
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 获取类别ID和名称
                cls = int(box.cls)
                label = f"{class_names[cls]} {conf:.2f}"
                
                # 获取类别对应的颜色
                color = class_colors[cls]
                
                # 绘制边界框和标签
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 8) #矩形框颜色、粗细
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 5)
        
        # 对绘制后的帧进行缩放（如果需要）
        resized_frame = cv2.resize(annotated_frame, (width, height))
        
        # 显示帧率
        frame_count += 1
        cv2.putText(resized_frame, f'Frame: {frame_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('YOLOv11 水下目标检测', resized_frame)
        
        # 保存结果
        if output_path:
            out.write(resized_frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

# 使用示例  
if __name__ == "__main__":
    # 训练好的模型路径
    model_path = "AUOD.pt" # 模型路径
    
    # 输入视频路径（替换为你的视频路径）
    video_path = "Dataset/test/YN050013.MP4"
    
    # 输出视频路径（可选）
    output_path = "Detection_Test_Output/YN050013_Detection_result.mp4"
    
    # 执行检测
    detect_objects_in_video(
        video_path=video_path,
        model_path=model_path,
        output_path=output_path,
        conf_threshold=0.6  # 可调整置信度阈值
    )