import os
# 强制设置环境变量（必须在任何库导入前执行）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
# pylint: disable=wrong-import-position
from ultralytics import YOLO
# import torch


def train_model():
    # 加载预训练模型
    model = YOLO('yolo11s.pt')

    results = model.train(
        data='data.yaml',          # 数据配置文件
        epochs=300,                # 训练轮次
        imgsz=640,                 # 固定输入尺寸
        batch=8,                  # 批量大小
        pretrained=True,           # 使用预训练权重
        device=0,                  # 指定GPU
        workers=8,                 # 数据加载线程数

        # 优化参数
        int8=True,
        lr0=0.001,                 # 初始学习率
        lrf=0.01,                  # 学习率衰减系数
        momentum=0.937,            # 动量
        weight_decay=0.0005,       # 权重衰减
        box=7.5,                   # 边界框损失权重
        cls=0.5,                   # 分类损失权重
        dfl=1.5,                   # 分布聚焦损失权重

        # 数据增强
        fliplr=0.5,                # 水平翻转
        flipud=0.2,                # 垂直翻转
        mosaic=1.0,                # 马赛克增强
        mixup=0.1,                 # 混合增强
        copy_paste=0.1,            # 复制粘贴增强
    )

    return results


if __name__ == '__main__':
    # Windows系统需要添加此句（用于可执行文件打包）
    from multiprocessing import freeze_support
    freeze_support()

    # 调用训练函数
    train_model()
