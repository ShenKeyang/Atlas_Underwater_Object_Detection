from ultralytics import YOLO
import time
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

def train_yolov11(data_yaml, model_path="yolov11s", epochs=50, batch_size=8):
    """训练YOLOv11模型并记录Loss和评估指标"""
    
    model = YOLO(model_path)

    # 设置环境变量以解决OpenMP冲突
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 训练模型
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        device=0,
        project="YOLOv11_train_results",
        name="train_results",           
        workers=4,                   # 数据加载线程数    
        optimizer='Adam',            # 优化器类型：'Adam'（收敛快）或 'SGD'（泛化性好）
        lr0=0.001,                   # 初始学习率（Adam常用0.001，SGD常用0.01）
        lrf=0.01,                    # 最终学习率与初始学习率的比例（表示降至1%）
        augment=True,                # 启用Mosaic、MixUp 等增强（关键参数，提升泛化能力）
        hsv_h=0.03,                  # 色调增强系数
        hsv_s=0.8,                   # 饱和度增强系数
        hsv_v=0.6,                   # 亮度增强系数
        degrees=15.0,                # 随机旋转角度范围
        scale=0.5,                   # 图像缩放范围
        patience=20,                 # 早停机制：若20轮mAP无提升则停止训练
        save_period=10,              # 每10轮保存一次检查点
        amp=True                     # 启用自动混合精度训练（加速2倍，适用于RTX系列GPU）
    )
    
    # 假设 results 包含最后一个 epoch 的评估指标
    final_result = results[0] if isinstance(results, list) else results
    metrics_data = []
    
    # 提取并保存Loss和评估指标数据
    metrics_data = []
    for r in results:
        metrics_data.append({
            # Loss指标
            "epoch": r.epoch,
            "box_loss": r.box,
            "cls_loss": r.cls,
            "obj_loss": r.obj,
            "total_loss": r.box + r.cls + r.obj,
            # 评估指标（新增）
            "precision": r.precision.mean(),  # 平均精确率
            "recall": r.recall.mean(),        # 平均召回率
            "map50": r.map50,                 # mAP@0.5
            "map5095": r.map5095,             # mAP@0.5:0.95
            "f1": 2 * r.precision.mean() * r.recall.mean() / (r.precision.mean() + r.recall.mean() + 1e-10)  # F1分数
        })
    
    # 保存为CSV
    df = pd.DataFrame(metrics_data)
    df.to_csv("train_metrics.csv", index=False)
    print("训练指标已保存至 train_metrics.csv")
    
    # 可视化Loss曲线
    plt.figure(figsize=(12, 5))
    
    # 左侧：Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['total_loss'], label='Total Loss', color='blue')
    plt.plot(df['epoch'], df['box_loss'], label='Box Loss', color='orange')
    plt.plot(df['epoch'], df['cls_loss'], label='Class Loss', color='green')
    plt.plot(df['epoch'], df['obj_loss'], label='Object Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.grid(True, alpha=0.3)
    
    # 右侧：评估指标曲线
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['map50'], label='mAP@0.5', color='blue')
    plt.plot(df['epoch'], df['map5095'], label='mAP@0.5:0.95', color='orange')
    plt.plot(df['epoch'], df['precision'], label='Precision', color='green')
    plt.plot(df['epoch'], df['recall'], label='Recall', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Evaluation Metrics')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300)
    plt.show()
    
    # 打印最佳指标
    best_epoch = df['map5095'].idxmax()
    print(f"\n最佳训练结果（Epoch {df['epoch'][best_epoch]}）:")
    print(f"mAP@0.5: {df['map50'][best_epoch]:.4f}")
    print(f"mAP@0.5:0.95: {df['map5095'][best_epoch]:.4f}")
    print(f"Precision: {df['precision'][best_epoch]:.4f}")
    print(f"Recall: {df['recall'][best_epoch]:.4f}")
    print(f"F1 Score: {df['f1'][best_epoch]:.4f}")

if __name__ == '__main__':

    # 设置工作目录为项目根目录
    project_root = "E:/A大三小/专业方向课程设计/Atlas_Underwater_Object_Detection"
    os.chdir(project_root)

    # 验证路径
    data_yaml = "YOLOv11/Custom_Data.yaml"
    with open(data_yaml, 'r') as f:
        lines = f.readlines()
    train_path = lines[0].split(': ')[1].strip()
    val_path = lines[1].split(': ')[1].strip()

    # 构建绝对路径（基于项目根目录）
    abs_train_path = os.path.join(project_root, "YOLOv11/txt_&_img", train_path)
    abs_val_path = os.path.join(project_root, "YOLOv11/txt_&_img", val_path)

    print(f"训练集路径: {abs_train_path} - 存在: {os.path.exists(abs_train_path)}")
    print(f"验证集路径: {abs_val_path} - 存在: {os.path.exists(abs_val_path)}")

    train_yolov11(
        data_yaml="YOLOv11/Custom_Data.yaml",
        model_path="YOLOv11/yolov11s.pt",   # 使用yolov11s模型（可选n/s/m/l/x）
        epochs=100,
        batch_size=8
    )