from ultralytics import YOLO
import cv2

# 加载最佳模型进行推理
model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict('underwater_dataset/source_data/data_from_zm/train-image/u002214.jpg', imgsz=640)

print(results[0].boxes)

# 可视化结果
# for r in results:
#     cv2.imshow('result', r.plot())
#     cv2.waitKey(0)
