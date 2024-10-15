# -*- coding: utf-8 -*-
import os
import cv2
from multiprocessing import Pool

import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import shutil

# 文件夹路径
input_path = r"D:\文件\慢阻肺\CT\Train\Cut_Lung"
output_folder = r"F:\Combin_test\Lung"
df = pd.read_excel('慢阻肺标注样例.xlsx', engine='openpyxl')
# max_multiple = {"Label 0": 5, "Label 1": 10, "Label 2": 5, "Label 3": 50}

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    image = Image.open(path)
    if flags == cv2.IMREAD_GRAYSCALE:
        image = image.convert('L')
    image_array = np.array(image)
    return image_array

def imwrite_unicode(path, img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil.save(path)

def has_subdirectory(directory_path):
    for name in os.listdir(directory_path):
        if os.path.isdir(os.path.join(directory_path, name)):
            return True
    return False

# 图片堆叠
def process_combination(args):
    folder_path, part1, part2, part3, i, j, k, output_path = args
    images = [
        imread_unicode(os.path.join(folder_path, part1[i]), cv2.IMREAD_GRAYSCALE),
        imread_unicode(os.path.join(folder_path, part2[j]), cv2.IMREAD_GRAYSCALE),
        imread_unicode(os.path.join(folder_path, part3[k]), cv2.IMREAD_GRAYSCALE)
    ]
    merged_image = cv2.merge(images)
    imwrite_unicode(os.path.join(output_path, f'{i}_{j}_{k}.jpg'), merged_image)

# 删除空文件夹
for root, dirs, _ in os.walk(input_path):
    for name in dirs:
        dir_path = os.path.join(root, name)
        if len(os.listdir(dir_path)) < 3:
            shutil.rmtree(dir_path)
            print(f"已删除空文件夹：{dir_path}")

selected_columns = df[['img-ID', 'Label']]
label = np.array(selected_columns)
if __name__ == '__main__':
    for root, _, files in os.walk(input_path):
        folder_name = os.path.basename(root)
        if has_subdirectory(root):
            continue

        indices = np.where(label[:, 0] == folder_name)
        if len(indices[0]) > 0:
            index = indices[0][0]
            cls_id = label[index, 1]

        folder_path = os.path.join(input_path, folder_name)
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        # max_combinations = len(image_files) * max_multiple[f"Label {cls_id}"]

        # 将文件夹中的所有文件分为三份
        split_size = len(image_files) // 3
        part1 = image_files[:split_size]
        part2 = image_files[split_size:2*split_size]
        part3 = image_files[2*split_size:]

        # 确保每一部分都有足够的图片
        if len(part1) == 0 or len(part2) == 0 or len(part3) == 0:
            continue

        output_path = os.path.join(output_folder, folder_name)
        os.makedirs(output_path, exist_ok=True)



        args = ((folder_path, part1, part2, part3, i, j, k, output_path) for i in range(len(part1)) for j in range(len(part2)) for k in range(len(part3)))

        with Pool() as pool:
            pool.map(process_combination, args)
        # # 从每一份中各取一张图片来堆叠为三通道图片
        # for i in range(len(part1)):
        #     for j in range(len(part2)):
        #         for k in range(len(part3)):
        #             images = [
        #                 imread_unicode(os.path.join(folder_path, part1[i]), cv2.IMREAD_GRAYSCALE),
        #                 imread_unicode(os.path.join(folder_path, part2[j]), cv2.IMREAD_GRAYSCALE),
        #                 imread_unicode(os.path.join(folder_path, part3[k]), cv2.IMREAD_GRAYSCALE)
        #             ]
        #             merged_image = cv2.merge(images)
        #             imwrite_unicode(os.path.join(output_path, f'{folder_name}_{i}{j}{k}.jpg'), merged_image)

    print("文件转换完成。")
    print("当前时间：", datetime.now())