# coding=utf-8
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

import time, os

import cv2  # 图片处理三方库，用于对图片进行前后处理
import numpy as np  # 用于对多维数组进行计算
import torch  # 深度学习运算框架，此处主要用来处理数据
from tqdm import tqdm

from mindx.sdk import Tensor  # mxVision 中的 Tensor 数据结构
from mindx.sdk import base  # mxVision 推理接口
from mindx.sdk import transpose
from mindx.sdk.base import ImageProcessor, Size, Image

from test_post import output_standardization, scale_coords, nms, draw_bbox  # 模型前后处理相关函数

from contextlib import contextmanager


@contextmanager
def count_time(desc='时间', show=False):
    """
    计算代码块执行时间的上下文管理器，自动根据执行时间选择合适的单位显示。

    :param desc: 时间描述信息，用于标识当前计时的代码块
    :type desc: str
    """

    # 0: not show
    # 1: follow show
    # 2: must show
    mode = 0

    if mode == 0:
        show = False
    elif mode == 2:
        show = True

    if not show:
        yield
    else:
        start_time = time.time()
        yield
        end_time = time.time()
        execution_time = end_time - start_time
        if execution_time < 1:
            print(f'{desc}: {execution_time * 1000:.2f} ms')
        else:
            print(f'{desc}: {execution_time :.2f} s')


def pre_progress(img_ori):
    """
    图片前处理（缩放、转RGB、补边）返回处理后图像、缩放因子
    
    :param img_ori: 解码得到的原始图片，NV12 编码格式
    :type img_ori: mindx.sdk.base.Image
    :return: 包含两个元素的元组，依次为：处理后图像、缩放因子
    :rtype: tuple(numpy.ndarray[(640, 640, 3), int8], float)
    """
    
    # 初始化一个ImageProcessor对象，相当于静态成员，之后可以直接调用
    if not hasattr(image_process, 'imageProcessor'):
        image_process.imageProcessor = ImageProcessor(0)
    imageProcessor = image_process.imageProcessor
    
    with count_time('图片缩放'):
        width, height = img_ori.width, img_ori.height
        scale_ratio = 640 / max(height, width)
        nw = int(width * scale_ratio)
        nh = int(height * scale_ratio)
        resize_para = Size(nw, nh)
        img_yuv = imageProcessor.resize(img_ori, resize_para)

    with count_time('编码转RGB'):
        img_yuv = img_yuv.to_tensor() # shape = (1, 540, 640, 1)
        img_yuv.to_host()
        img_yuv = np.array(img_yuv)[0]
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB_NV12) # shape = (360, 640, 3)
        
    with count_time('归一化'):
        img_rgb = img_rgb.astype(np.float32) / 255.0

    with count_time('尺寸调整/补边'):
        pad_h = 640 - img_rgb.shape[0]
        pad_w = 640 - img_rgb.shape[1]
        img_rgb = np.pad(img_rgb, ((pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2), (0, 0)), mode='constant', constant_values=114.0/255.0)        
        
    return (img_rgb, scale_ratio)


def image_predict(img_rgb):
    """
    运行检测任务，返回推理结果
    
    :param img_rgb: 经过前处理的图像
    :type img_rgb: numpy.ndarray[(640, 640, 3), float32]
    :return: 推理结果
    :rtype: 形状为 [1, 8, 8400] 的 tensor
    """

    # 初始化一个 model 对象，相当于静态成员，之后可以直接调用
    if not hasattr(image_predict, 'model'):
        with count_time('初始化model'):
            image_predict.model = base.model('models/best.om')
    model = image_predict.model
    
    with count_time('图片转契合tensor'):
        img = np.expand_dims(img_rgb, 0)
        img = img.transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)
        img = Tensor(img)

    with count_time('执行推理'):
        # 输入数据类型：List[base.Tensor]，返回模型推理输出的 List[base.Tensor]
        #print(f'len(model.infer([img])) = {len(model.infer([img]))}')
        #print(f'model.infer([img])[0].shape = {model.infer([img])[0].shape}')
        output = model.infer([img])[0]
    
    return output


def post_progress(output, scale_ratio, img_ori):
    output.to_host()
    output = np.array(output)
    ori_shape = (img_ori.height, img_ori.width, 3)

    with count_time('输出标准化'):
        output = output_standardization(output)

    with count_time('非极大值抑制'):
        # 利用非极大值抑制处理模型输出，conf_thres 为置信度阈值，iou_thres 为iou阈值
        boxout = nms(torch.tensor(output), conf_thres=0.4, iou_thres=0.5)  
        pred_all = boxout[0].numpy()

    with count_time('推理结果缩放'):
        scale_coords([640, 640], pred_all[:, :4], ori_shape)
    
    labels_dict = {0: 'holothurian', 1: 'echinus', 2: 'scallop', 3: 'starfish'}
    img_dw = img_ori.to_tensor() 
    img_dw.to_host()
    img_dw = np.array(img_dw)[0]
    img_dw = draw_bbox(pred_all, img_dw, (255), 8, labels_dict)  # 画出检测框、类别、概率

    return img_dw


def image_process(input_path='underwater.jpg'):
    # 初始化一个ImageProcessor对象，相当于静态成员，之后可以直接调用
    if not hasattr(image_process, 'imageProcessor'):
        image_process.imageProcessor = ImageProcessor(0)
    imageProcessor = image_process.imageProcessor
        
    with count_time('读入图片', show=True):
        if not os.path.exists(input_path):
            raise ValueError(f'路径不存在: {input_path}')
        img_ori = imageProcessor.decode(input_path, base.nv12)

    with count_time('前处理', show=True):
        img_rgb, scale_ratio = pre_progress(img_ori)
        
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('underwater_processed.jpg', img_bgr)
    
    with count_time('模型预测', show=True):
        output = image_predict(img_rgb)
        
    with count_time('后处理', show=True):
        img_dw = post_progress(output, scale_ratio, img_ori)
        
    return

    with count_time('保存图片', show=True):
        name, ext = os.path.splitext(input_path)
        output_path = name + '_processed' + ext
        cv2.imwrite(output_path, img_dw)

    return


def video_process(video_path='YN050013.MP4'):
    # 载入模型
    start_time = time.time()
    model = init()
    end_time = time.time()
    print(f"模型载入：{(end_time - start_time) * 1000:.2f}ms")

    # 读入视频
    if not os.path.exists(video_path):
        raise ValueError(f'路径不存在: {video_path}')
    img_bgr = cv2.imread(video_path, cv2.IMREAD_COLOR)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    # 得到宽高帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取总帧数用于进度显示
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频写入器
    output_path = list(os.path.splitext(video_path))
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
    base.mx_init()
    for i in tqdm(range(1000)):
        image_process()
    #video_process('test_1s.MP4')
    #video_process()
