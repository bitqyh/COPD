import os
from datetime import datetime

import cv2
from PIL import Image
import numpy as np
from collections import deque


def imread_chinese_path(chinese_path):
    # 使用PIL库读取图像
    img_pil = Image.open(chinese_path)
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    # 将PIL图像转换为OpenCV图像
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    return img_cv


def region_growing(img):
    # img = imread_chinese_path(image_path)
    height, width = img.shape
    region = np.zeros((height, width), np.uint8)

    pixel_list = deque(SearchSeed(img))
    length = len(pixel_list)
    while pixel_list:
        current_pixel = pixel_list.popleft()

        region[current_pixel] = 255

        neighbors = [(current_pixel[0] - 1, current_pixel[1]), (current_pixel[0] + 1, current_pixel[1]),
                     (current_pixel[0], current_pixel[1] - 1), (current_pixel[0], current_pixel[1] + 1)]

        for neighbor in neighbors:
            if neighbor[0] >= 0 and neighbor[0] < height and neighbor[1] >= 0 and neighbor[1] < width:
                if img[neighbor] >= 1 and region[neighbor] == 0:  # 如果像素是白色且不在区域内
                    pixel_list.append(neighbor)
                    region[neighbor] = 255  # Mark the pixel as part of the region as soon as it's added to the list

    mask = region == 255
    img[mask] = 0

    return img


def SearchSeed(img):
    pixel_list = []
    scope = 150
    seed = None
    height, width = img.shape
    start_index = width // 2
    for i in range(0, width):  # 从中间列开始搜索
        if img[0, i] == 255:
            seed = (0, i)
            pixel_list.append(seed)
        if img[511, i] == 255:
            seed = (511, i)
            pixel_list.append(seed)
        if i <= 150 & i >= 400: # 限制搜索范围
            if img[i, 0] == 255:
                seed = (i, 0)
                pixel_list.append(seed)
            if img[i, 511] == 255:
                seed = (i, 511)
                pixel_list.append(seed)

    return pixel_list


def crop_fixed_size(img):
    # img = imread_chinese_path(image_path)
    left_x, right_x = 0, 512
    top_y, bottom_y = 50, 512
    crop_img = img[top_y:bottom_y, left_x:right_x]
    return crop_img


def process_image(img):
    # 读取图像并转换为灰度图像
    # img = imread_chinese_path(image_path)

    # 膨胀和腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # # 中值滤波去噪
    # img = cv2.medianBlur(img, 3)

    # 高斯去噪
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


input_folder = r"E:\桌面\文件\慢阻肺\test\Cut_Lung\CT5533484"  # 输入文件夹
threshold = 4000  # 设置阈值，根据实际情况调整这个值

# non_zero_count = cv2.countNonZero(img)
# print(non_zero_count)
for root, _, files in os.walk(input_folder):
    print(root)
    for file in files:
        if file.endswith('.jpg'):  # 只处理 jpg 文件
            file_path = os.path.join(root, file)
            img = imread_chinese_path(file_path)
            img = region_growing(img)  # 区域生长算法
            crop_img = crop_fixed_size(img)

            white_count = np.sum(crop_img == 255)
            # print(file + f"{white_count}")

            if white_count < threshold:  # 如果非零像素的数量低于阈值
                os.remove(file_path)  # 删除这个图像文件

            else:
                img = process_image(img)  # 去噪
                Image.fromarray(img).save(file_path, 'JPEG')

for root, dirs, _ in os.walk(input_folder, topdown=False):  # topdown=False，这样可以先遍历子目录
    for name in dirs:
        dir_path = os.path.join(root, name)
        if not os.listdir(dir_path):  # 检查文件夹是否为空
            os.rmdir(dir_path)  # 删除空文件夹
            print(f"已删除空文件夹：{dir_path}")

print("当前时间：", datetime.now())
