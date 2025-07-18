import os
import sys
import json
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import cv2
# import numpy as np
from image_enhance import clahe_enhance


CURRENT_DIR = os.path.split(__file__)[0]
DATASET_DIR = os.path.join(CURRENT_DIR, 'underwater_dataset')

class_mapping = {
    'holothurian': 0,
    'echinus': 1,
    'scallop': 2,
    'starfish': 3
}

class_names = [
    'holothurian',
    'echinus',
    'scallop',
    'starfish'
]


# pylint: disable=too-many-locals
def xml2yolotxt(xml_path, txt_path):
    """
    将xml格式的标注框数据转化为yolo可用的txt标注文件

    :param xml_path: xml文件的路径
    :param txt_path: 输出的txt文件的路径
    """

    root = ET.parse(xml_path).getroot()

    yolo_lines = []

    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    for obj in root.findall('object'):

        class_name = obj.find('name').text
        class_idx = class_mapping.get(class_name, -1)
        if class_idx == -1:
            # 比如海草，不在 class_mapping 里面，不需要解析
            continue

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines))


def load_data_from_zm():
    """
    将 DATASET_DIR/source_data/data_from_zm/ 中的文件
    全部处理并转移到 DATASET_DIR/labels/ 和 DATASET_DIR/images/ 中
    """

    label_dir = os.path.join(DATASET_DIR, 'labels')
    image_dir = os.path.join(DATASET_DIR, 'images')
    ZM_DATA_DIR = os.path.join(DATASET_DIR, 'source_data/data_from_zm')

    source_xml_dir = os.path.join(ZM_DATA_DIR, 'test-A-box')
    source_image_dir = os.path.join(ZM_DATA_DIR, 'test-A-image')
    filename_prefix = 'zmtestA'
    for file in tqdm(os.listdir(source_xml_dir), desc='ZM test A 文件整理'):
        filename = os.path.splitext(file)[0]
        new_filename = filename_prefix + filename
        xml2yolotxt(os.path.join(source_xml_dir, file), os.path.join(label_dir, new_filename + '.txt'))
        shutil.copyfile(os.path.join(source_image_dir, filename + '.jpg'), os.path.join(image_dir, new_filename + '.jpg'))

    source_xml_dir = os.path.join(ZM_DATA_DIR, 'test-B-box')
    source_image_dir = os.path.join(ZM_DATA_DIR, 'test-B-image')
    filename_prefix = 'zmtestB'
    for file in tqdm(os.listdir(source_xml_dir), desc='ZM test B 文件整理'):
        filename = os.path.splitext(file)[0]
        new_filename = filename_prefix + filename
        xml2yolotxt(os.path.join(source_xml_dir, file), os.path.join(label_dir, new_filename + '.txt'))
        shutil.copyfile(os.path.join(source_image_dir, filename + '.jpg'), os.path.join(image_dir, new_filename + '.jpg'))

    source_xml_dir = os.path.join(ZM_DATA_DIR, 'train-box')
    source_image_dir = os.path.join(ZM_DATA_DIR, 'train-image')
    filename_prefix = 'zmtrain'
    all_c_files = []
    for file in os.listdir(source_xml_dir):
        if file[0] == 'c':
            all_c_files.append(file)
    for file in tqdm(all_c_files, desc='ZM train 文件整理'):
        filename = os.path.splitext(file)[0]
        new_filename = filename_prefix + filename
        xml2yolotxt(os.path.join(source_xml_dir, file), os.path.join(label_dir, new_filename + '.txt'))
        shutil.copyfile(os.path.join(source_image_dir, filename + '.jpg'), os.path.join(image_dir, new_filename + '.jpg'))


def label_coco2yolo(json_path, yolo_labels_dir, prefix=''):
    """
    将COCO格式的单个json标注文件转化为一系列yolo标注文件，并在文件名前附加前缀

    :param json_path: json标注文件的路径
    :param yolo_labels_dir: 输出的一系列标注文件的文件夹
    :param prefix: 文件名前缀，默认为空串
    """

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 生成映射关系：类别 ID -> index，图片 ID -> 图片信息，图片 ID -> 标注
    category_id_to_index = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    # 逐个处理所有图片
    for img_id, img_info in image_id_to_info.items():
        img_width = img_info['width']
        img_height = img_info['height']

        img_file_name = img_info['file_name']
        img_name_without_ext = os.path.splitext(img_file_name)[0]
        yolo_file_name = f"{prefix}{img_name_without_ext}.txt"
        yolo_file_path = os.path.join(yolo_labels_dir, yolo_file_name)

        with open(yolo_file_path, 'w', encoding='utf-8') as f:
            if img_id in image_id_to_annotations:
                for ann in image_id_to_annotations[img_id]:
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height

                    category_index = category_id_to_index[ann['category_id']]
                    f.write(f"{category_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def load_data_from_duo():
    """
    将 DATASET_DIR/source_data/DUO/ 中的文件
    全部处理并转移到 DATASET_DIR/labels/ 和 DATASET_DIR/images/ 中
    """

    DUO_DIR = os.path.join(DATASET_DIR, 'source_data/DUO/')

    json_path = os.path.join(DUO_DIR, 'annotations/instances_test.json')
    yolo_labels_dir = os.path.join(DATASET_DIR, 'labels')
    prefix = 'duotest'
    label_coco2yolo(json_path, yolo_labels_dir, prefix)
    duo_image_dir = os.path.join(DUO_DIR, 'images/test')
    image_dir = os.path.join(DATASET_DIR, 'images')
    for filename in tqdm(os.listdir(duo_image_dir), desc='复制 DUO test 图片'):
        new_filename = prefix + filename
        shutil.copyfile(os.path.join(duo_image_dir, filename), os.path.join(image_dir, new_filename))

    json_path = os.path.join(DUO_DIR, 'annotations/instances_train.json')
    yolo_labels_dir = os.path.join(DATASET_DIR, 'labels')
    prefix = 'duotrain'
    label_coco2yolo(json_path, yolo_labels_dir, prefix)
    duo_image_dir = os.path.join(DUO_DIR, 'images/train')
    image_dir = os.path.join(DATASET_DIR, 'images')
    for filename in tqdm(os.listdir(duo_image_dir), desc='复制 DUO train 图片'):
        new_filename = prefix + filename
        shutil.copyfile(os.path.join(duo_image_dir, filename), os.path.join(image_dir, new_filename))


# pylint: disable=too-many-locals
def labeling_image(image_path, label_path, preview_path, scaling_ratio=0.5):
    """
    将图片与yolo的txt格式标注文件结合绘制标注预览图

    :param image_path: 图片的路径
    :param label_path: 标注数据路径，应为yolo支持的txt格式
    :param preview_path: 预览文件存储路径
    :param scaling_ratio: 改变预览文件尺寸的缩放系数，默认为0.5
    """

    colors = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0),
        (200, 200, 200)
    ]

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}，程序退出")
        sys.exit(0)
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"标注文件 {label_path} 不存在，程序退出")
        sys.exit(0)

    height, width = image.shape[:2]
    height = int(height * scaling_ratio)
    width = int(width * scaling_ratio)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    for line in lines:
        line = line.strip().split()
        if not line:
            continue
        class_idx = int(line[0])
        x_center = float(line[1])
        y_center = float(line[2])
        box_width = float(line[3])
        box_height = float(line[4])
        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else f"Class_{class_idx}"
        color = colors[class_idx % len(colors)]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, max(1, int(width / 450)))
        label = f"{class_name}"
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, width / 900, color, max(1, int(width / 450)))

    cv2.imwrite(preview_path, image)


def labeling_all_image():
    """
    将 DATASET_DIR/images/ 和 DATASET_DIR/labels/ 结合，
    创建所有图片的标注预览并存储到 DATASET_DIR/labeled_image/ 中
    """

    images_dir = os.path.join(DATASET_DIR, 'images')
    labels_dir = os.path.join(DATASET_DIR, 'labels')
    preview_dir = os.path.join(DATASET_DIR, 'labeled_image')
    for file in tqdm(os.listdir(images_dir), desc='绘制图片标注预览'):
        filename = os.path.splitext(file)[0]
        image_path = os.path.join(images_dir, file)
        label_path = os.path.join(labels_dir, filename + '.txt')
        preview_path = os.path.join(preview_dir, file)
        if os.path.exists(preview_path):
            continue
        labeling_image(image_path, label_path, preview_path)


def enhance_all_image():
    images_dir = os.path.join(DATASET_DIR, 'images')
    output_dir = os.path.join(CURRENT_DIR, 'training_dataset/images')
    for file in tqdm(os.listdir(images_dir), '增强图像（提高对比度）'):
        input_path = os.path.join(images_dir, file)
        image = cv2.imread(input_path)
        image = clahe_enhance(image, grid_size=(4,4))
        output_path = os.path.join(output_dir, file)
        cv2.imwrite(output_path, image)


load_data_from_zm()
load_data_from_duo()
# labeling_all_image()
