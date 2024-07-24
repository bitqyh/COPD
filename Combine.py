# -*- coding: gbk -*-
import os
import cv2
import itertools
import numpy as np
import pandas as pd
from PIL import Image
import shutil
from datetime import datetime

# 文件夹路径
input_path = r"E:\桌面\文件\慢阻肺\CT\Train\Cut_Lung"
output_folder = r"E:\桌面\文件\慢阻肺\CT\Combin\Lung"
df = pd.read_excel('慢阻肺标注样例.xlsx', engine='openpyxl')
max_multiple = {"Label 0": 5, "Label 1": 5, "Label 2": 5, "Label 3": 5}
def imread_chinese_path(chinese_path):
    # 使用PIL库读取图像
    img_pil = Image.open(chinese_path)
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    # 将PIL图像转换为OpenCV图像
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    return img_cv
def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    # 使用PIL库读取图片
    image = Image.open(path)

    # # 将图片转换为numpy数组
    # image_array = np.array(image)
    #
    # # 如果需要，将RGB图片转换为BGR格式
    # if flags == cv2.IMREAD_COLOR and image_array.ndim == 3:
    #     image_array = image_array[:, :, ::-1]
    #
    # # 如果需要，将图片转换为灰度格式
    if flags == cv2.IMREAD_GRAYSCALE:
        image = image.convert('L')

    image_array = np.array(image)
    return image_array
def imwrite_unicode(path, img):
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 使用PIL库保存图片
    img_pil.save(path)

def has_subdirectory(directory_path):
    for name in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, name)):
            return True
    return False

# output_folder = os.path.join(output_folder, os.path.basename(input_path))

for root, dirs, _ in os.walk(input_path):
    for name in dirs:
        dir_path = os.path.join(root, name)
        if len(os.listdir(dir_path)) < 3:
            # dir_path = os.fsencode(dir_path)
            shutil.rmtree(dir_path)
            print(f"已删除空文件夹：{dir_path}")
        # if not os.listdir(dir_path):  # 检查文件夹是否为空
        #     os.rmdir(dir_path)  # 删除空文件夹
        #     print(f"已删除空文件夹：{dir_path}")

for index, row in df.iterrows():
    # image_name = row['B'] # 文件夹名Ct0000000
    # label = row['C'] #类别
    selected_columns = df[['img-ID', 'Label']]
    # 转换为二维数组
    label = np.array(selected_columns)

for root, _, files in os.walk(input_path):  # 遍历输入文件夹中的所有文件
    folder_name = os.path.basename(root)  # 获取文件夹名
    if has_subdirectory(root):
        continue

    indices = np.where(label[:, 0] == folder_name)
    if len(indices[0]) > 0:  # 如果找到了匹配的元素
        index = indices[0][0]  # 获取第一个匹配元素的索引

        cls_id = label[index, 1]

    folder_path = os.path.join(input_path, folder_name)
    # 获取文件夹下所有的图片文件
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # 计算最大组合数量
    max_combinations = len(image_files) * max_multiple[f"Label {cls_id}"]

    # 生成所有可能的三张图片的组合
    combinations = list(itertools.combinations(image_files, 3))

    # 如果组合数量大于最大组合数量，只取前max_combinations个组合
    if len(combinations) > max_combinations:
        combinations = combinations[:max_combinations]

    # 对于每个组合，读取三张图片并合并为一张三通道的图片
    for i, combination in enumerate(combinations):
        images = [imread_unicode(os.path.join(folder_path, image_file), cv2.IMREAD_GRAYSCALE) for image_file in combination]
        merged_image = cv2.merge(images)  # 合并图片

        # 保存合并后的图片
        # os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, folder_name)
        os.makedirs(output_path, exist_ok=True)
        imwrite_unicode(os.path.join(output_path, f'{folder_name}_{i}.jpg'), merged_image)


print("文件转换完成。")
print("当前时间：", datetime.now())
# # 文件夹路径
# folder_path = 'path_to_your_folder'
#
# # 获取文件夹下所有的图片文件
# image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#
# # 读取第一张图片来获取图片的尺寸
# first_image = cv2.imread(os.path.join(folder_path, image_files[0]), cv2.IMREAD_GRAYSCALE)
# image_shape = first_image.shape
#
# # 创建一个空的多通道图像
# multi_channel_image = np.zeros((*image_shape, len(image_files)))
#
# # 将每张图片添加到多通道图像中
# for i, image_file in enumerate(image_files):
#     gray = cv2.imread(os.path.join(folder_path, image_file), cv2.IMREAD_GRAYSCALE)
#     assert gray.shape == image_shape, "All images must have the same size"
#     multi_channel_image[:, :, i] = gray
#
# # 现在，multi_channel_image是一个多通道的图像，可以作为网络的输入