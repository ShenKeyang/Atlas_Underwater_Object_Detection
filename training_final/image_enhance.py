import cv2


def clahe_enhance(image, clip_limit=3.0, grid_size=(8, 8)):
    """
    Lab空间处理亮度通道，自适应增强对比度，避免颜色失真
    :param image: 输入图像（BGR格式，uint8）
    :param clip_limit: 对比度限制（默认3.0，推荐1-5）
    :param grid_size: 分块大小（默认(8,8)，推荐4-16）
    :return: 增强后图像（BGR格式，uint8）
    """

    # 转Lab空间并分离亮度(L)和颜色(a/b)通道
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 仅增强亮度通道后合并
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l)
    lab = cv2.merge([l_enhanced, a, b])

    # 返回转化为BGR空间的图像
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
