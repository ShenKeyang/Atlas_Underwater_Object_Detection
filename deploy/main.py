# coding=utf-8
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

import time, os

import cv2  # 图片处理三方库，用于对图片进行前后处理
import numpy as np  # 用于对多维数组进行计算
import torch  # 深度学习运算框架，此处主要用来处理数据

from mindx.sdk import Tensor  # mxVision 中的 Tensor 数据结构
from mindx.sdk import base  # mxVision 推理接口
from mindx.sdk import transpose

from det_utils import get_labels, letterbox, scale_coords, nms, draw_bbox  # 模型前后处理相关函数

from tqdm import tqdm


def init_model():
    # 初始化资源和变量并载入模型
    base.mx_init()  # 初始化 mxVision 资源
    DEVICE_ID = 0  # 设备id
    model_path = 'models/best.om'  # 模型路径
    model = base.model(modelPath=model_path, deviceId=DEVICE_ID)  # 初始化 base.model 类
    return model


def process_yolov11_output(output):
    # 转换 [1, nc+4, num] 维度的 ndarray 转化为 [1, num, nc+5] 的形状
    transposed_output = np.transpose(output, (0, 2, 1))
    boxes = transposed_output[..., :4]
    classes = transposed_output[..., 4:]
    max_class_prob = np.max(classes, axis=-1, keepdims=True)
    processed_output = np.concatenate([boxes, max_class_prob, classes], axis=-1)
    return processed_output


def image_predict(img_bgr, model):

    start_time = time.time()

    # 前处理
    img, scale_ratio, pad_size = letterbox(img_bgr, new_shape=[640, 640])  # 对图像进行缩放与填充，保持长宽比
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.expand_dims(img, 0).astype(np.float32)  # 将形状转换为 channel first (1, 3, 640, 640)，即扩展第一维为 batchsize
    img = np.ascontiguousarray(img) / 255.0  # 转换为内存连续存储的数组
    img = [Tensor(img)] # 将numpy转为转为Tensor类

    # 执行推理
    # print(f'model input shape = {img[0].shape}')
    output = model.infer(img)[0]  # 执行推理。输入数据类型：List[base.Tensor]， 返回模型推理输出的 List[base.Tensor]
    # print(output.shape)

    # 后处理
    output.to_host()  # 将 Tensor 数据转移到内存
    output = np.array(output)  # 将数据转为 numpy array 类型
    output = process_yolov11_output(output)
    boxout = nms(torch.tensor(output), conf_thres=0.4, iou_thres=0.5)  # 利用非极大值抑制处理模型输出，conf_thres 为置信度阈值，iou_thres 为iou阈值
    pred_all = boxout[0].numpy()  # 转换为numpy数组
    scale_coords([640, 640], pred_all[:, :4], img_bgr.shape, ratio_pad=(scale_ratio, pad_size))  # 将推理结果缩放到原始图片大小
    labels_dict = get_labels()  # 得到类别信息，返回序号与类别对应的字典
    img_dw = draw_bbox(pred_all, img_bgr, (0, 255, 0), 8, labels_dict)  # 画出检测框、类别、概率

    end_time = time.time()
    # print(f"image_predict: {(end_time - start_time) * 1000:.2f}ms")

    return img_dw


def image_process(input_path='underwater.jpg'):
    # 载入模型
    start_time = time.time()
    model = init_model()
    end_time = time.time()
    print(f"模型载入：{(end_time - start_time) * 1000:.2f}ms")

    # 读入图片
    start_time = time.time()
    if not os.path.exists(input_path):
        raise ValueError(f'路径不存在: {input_path}')
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    end_time = time.time()
    print(f"图片载入：{(end_time - start_time) * 1000:.2f}ms")

    # 得到画上框的图片
    start_time = time.time()
    img_dw = image_predict(img_bgr, model)
    end_time = time.time()
    print(f"处理预测：{(end_time - start_time) * 1000:.2f}ms")

    # 保存图片到文件
    start_time = time.time()
    output_path = list(os.path.splitext(input_path))
    output_path.append('_processed')
    output_path[1], output_path[2] = output_path[2], output_path[1]
    output_path = ''.join(output_path)
    cv2.imwrite(output_path, img_dw)
    # print('save infer result success')
    end_time = time.time()
    print(f"图片保存：{(end_time - start_time) * 1000:.2f}ms")


def video_process(input_path='YN050013.MP4'):
    # 载入模型
    start_time = time.time()
    model = init_model()
    end_time = time.time()
    print(f"模型载入：{(end_time - start_time) * 1000:.2f}ms")

    # 读入视频
    if not os.path.exists(input_path):
        raise ValueError(f'路径不存在: {input_path}')
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")

    # 得到宽高帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取总帧数用于进度显示
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频写入器
    output_path = list(os.path.splitext(input_path))
    output_path.append('_processed')
    output_path[1], output_path[2] = output_path[2], output_path[1]
    output_path = ''.join(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


    # 逐帧处理视频
    with tqdm(total=frame_count, desc="读入视频", unit="帧") as pbar:
        while True:

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            end_time = time.time()
            # print(f"frame read: {(end_time - start_time) * 1000:.2f}ms")

            # 应用图像处理函数
            processed_frame = image_predict(frame, model)

            start_time = time.time()

            # 写入输出视频
            out.write(processed_frame)

            end_time = time.time()
            # print(f"frame write: {(end_time - start_time) * 1000:.2f}ms")

            # 更新进度
            pbar.update(1)



    cap.release()
    out.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    image_process()
    # video_process('test_1s.MP4')
    # video_process()
