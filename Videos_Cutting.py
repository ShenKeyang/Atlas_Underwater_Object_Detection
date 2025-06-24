import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim  # 用于计算帧间相似度

def extract_frames(video_path, output_dir, fps_interval=1, resize_dim=(1280, 720), 
                   similarity_threshold=0.85, save_all_if_no_motion=False):
    """
    从视频中抽取帧图像，支持关键帧筛选和抽帧时缩放
    
    参数:
    - video_path: 视频文件路径
    - output_dir: 输出图像目录
    - fps_interval: 抽帧间隔（每隔多少帧抽取一帧）
    - resize_dim: 缩放尺寸 (宽度, 高度)，设为None则不缩放
    - similarity_threshold: 关键帧筛选的相似度阈值 (0-1)，值越小保留的帧越多
    - save_all_if_no_motion: 若检测到无显著运动，是否保存所有抽取的帧
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    duration = total_frames / fps  # 视频时长（秒）
    
    print(f"视频信息: {fps:.2f} FPS, 共 {total_frames} 帧, 时长 {duration:.2f} 秒")
    print(f"抽帧参数: 每隔 {fps_interval} 帧抽取一帧")
    print(f"缩放参数: {resize_dim if resize_dim else '不缩放'}")
    print(f"关键帧筛选: 相似度阈值 {similarity_threshold}")
    
    # 初始化变量
    frame_count = 0
    save_count = 0
    prev_gray = None  # 用于存储上一帧的灰度图
    motion_detected = False  # 标记是否检测到运动
    
    # 开始抽帧
    for _ in tqdm(range(total_frames), desc="抽帧进度"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 按间隔抽帧
        if frame_count % fps_interval == 0:
            # 缩放帧（如果需要）
            if resize_dim:
                frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
            
            # 转换为灰度图用于相似度计算
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 关键帧筛选逻辑
            save_frame = False
            if prev_gray is not None:
                # 计算当前帧与前一帧的结构相似性指数(SSIM)
                sim = ssim(prev_gray, gray)
                if sim < similarity_threshold:
                    save_frame = True
                    motion_detected = True
            else:
                # 第一帧总是保存
                save_frame = True
            
            # 保存帧
            if save_frame:
                # 生成保存文件名（包含帧号）
                frame_num = frame_count // fps_interval
                save_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
                
                # 保存图像（使用高质量压缩）
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                save_count += 1
            
            # 更新上一帧的灰度图
            prev_gray = gray.copy()
        
        frame_count += 1
    
    # 如果没有检测到运动，决定是否保存所有抽取的帧
    if not motion_detected and save_all_if_no_motion:
        print("警告: 未检测到显著运动，正在重新保存所有抽取的帧...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频指针到开头
        frame_count = 0
        save_count = 0
        
        for _ in tqdm(range(total_frames), desc="重新抽帧"):
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % fps_interval == 0:
                if resize_dim:
                    frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
                
                frame_num = frame_count // fps_interval
                save_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                save_count += 1
                
            frame_count += 1
    
    cap.release()
    print(f"抽帧完成! 共抽取 {save_count} 帧图像, 保存至 {output_dir}")

# 使用示例（调整参数以满足您的需求）
extract_frames(
    video_path="Underwater_Dataset/train/videos/YN090013.MP4",
    output_dir="Underwater_Dataset/train/images",
    fps_interval=120,          # 每4秒抽一帧（假设30FPS）
    resize_dim=(1280, 720),    # 缩放到720p，降低后续处理压力
    similarity_threshold=0.85, # 相似度低于此值的帧会被保留
    save_all_if_no_motion=True # 如果没有检测到运动，仍然保存所有帧
)