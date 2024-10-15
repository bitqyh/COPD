import os
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# 检查文件路径是否存在
# lung_path = r"E:\桌面\文件\慢阻肺\CT\Health\CT5480753\CT5480753_.12422.jpg"
# mediastinal_path = r"E:\桌面\文件\慢阻肺\CT\Med_Health\CT5480753\CT5480753_.12422.jpg"

input_lung_folder = r"E:\桌面\文件\慢阻肺\CT\Lung"
input_mediastinal_folder = r"E:\桌面\文件\慢阻肺\CT\Med_Lung"

output_folder = r"E:\桌面\文件\慢阻肺\test"


def crop_by_ratio(img, size):
    ratio = size / 512
    height, width = img.shape[:2]
    new_width = int(width * ratio)
    new_height = int(height * (ratio - 0.1))

    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2

    return img[start_y:start_y+new_height, start_x:start_x+new_width]
def cutoff_lung_mediastinal(lung_path, mediastinal_path, output_path):  #  lung_path:肺部图像路径 mediastinal_path:纵隔图像路径
    CT_path = os.path.dirname(lung_path)
    CT_name = os.path.basename(CT_path)
    name = os.path.basename(lung_path)
    if not os.path.isfile(lung_path) or not os.path.isfile(mediastinal_path):
        print("One or both image files do not exist.")
    else:

        lung_img = plt.imread(lung_path)
        lung_img = cv2.cvtColor(lung_img, cv2.COLOR_RGB2GRAY)
        lung_img = (lung_img * 255).astype(np.uint8)

        mediastinal_img = plt.imread(mediastinal_path)
        mediastinal_img = cv2.cvtColor(mediastinal_img, cv2.COLOR_RGB2GRAY)
        mediastinal_img = (mediastinal_img * 255).astype(np.uint8)
        if lung_img is None or mediastinal_img is None:
            print("One or both images could not be read.")
        else:
            # 使用形态学阈值方法分割出肺部的图像
            # _, binary_septal_mask = cv2.threshold(mediastinal_img, 64, 255, cv2.THRESH_BINARY_INV) # 二值化
            _, binary_septal_mask = cv2.threshold(mediastinal_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            masked_lung_image = cv2.bitwise_and(lung_img, lung_img, mask=binary_septal_mask)

            jpg_filename = name  # 创建 JPEG 文件名
            jpg_filepath = os.path.join(output_path, CT_name, jpg_filename)

            masked_lung_image = crop_by_ratio(masked_lung_image, 450)  # 裁剪图像
            _, masked_lung_image = cv2.threshold(masked_lung_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masked_lung_image = cv2.resize(masked_lung_image, (512, 512))
            try:
                Image.fromarray(masked_lung_image).save(jpg_filepath, 'JPEG')
            except Exception as e:
                print(f"保存图片时出现错误：{e}")

for root, _, files in os.walk(input_lung_folder):  # 遍历输入文件夹中的所有文件
    folder_name = os.path.basename(root)  # 获取文件夹名
    lung_folder = os.path.join(output_folder, "Cut_Lung", folder_name)
    output_path = os.path.join(output_folder, "Cut_Lung")
    os.makedirs(lung_folder, exist_ok=True)
    for file in files:
        lung_path = os.path.join(root, file) # 获取 DICOM 文件的完整路径
        mediastinal_path = lung_path.replace("Lung", "Med_Lung")
        try:
            cutoff_lung_mediastinal(lung_path, mediastinal_path, output_path)
        except Exception as e:
                 print(f"转换 {file} 时出现错误：{e}")


for root, dirs, _ in os.walk(output_folder, topdown=False):  # topdown=False，这样可以先遍历子目录
    for name in dirs:
        dir_path = os.path.join(root, name)
        if not os.listdir(dir_path):  # 检查文件夹是否为空
            os.rmdir(dir_path)  # 删除空文件夹
            print(f"已删除空文件夹：{dir_path}")

print("文件转换完成。")
print("当前时间：", datetime.now())
# for root, _, files in os.walk(input_folder):  # 遍历输入文件夹中的所有文件
#     folder_name = os.path.basename(root) # 获取文件夹名
#     lung_folder = os.path.join(output_folder, "Med_Lung", folder_name)  # 创建一个名为 LUNG/CT* 的子文件夹
#     os.makedirs(lung_folder, exist_ok=True)  # 如果文件夹不存在，则创建它
#     if os.listdir(lung_folder):  # 如果文件夹不为空
#         print(f"文件夹 {lung_folder} 不为空，跳过。")
#         continue
#     for file in files:
#         dicom_filepath = os.path.join(root, file) # 获取 DICOM 文件的完整路径
#         try:
#             dicom_file = pydicom.dcmread(dicom_filepath)  # 读取 DICOM 文件
#             if 'PixelData' in dicom_file:
#                 series_description = dicom_file.SeriesDescription.lower()  # 转换为小写字母
#                 if "lung" in series_description and "5.0" in series_description: # 确保是肺部的 CT 扫描图像
#                     pixel_data = dicom_file.pixel_array.astype(np.int16)  # 提取像素数据
#                     winwidth = dicom_file.WindowWidth
#                     wincenter = dicom_file.WindowCenter
#                     pixel_data = extract_mediastinal_window(pixel_data)  # 提取肺窗
#                     image = Image.fromarray(pixel_data)  # 将处理后的图像数据转换为 PIL 图像对象
#                     jpg_filename = f"{folder_name}_{os.path.splitext(file)[1]}.jpg" # 创建 JPEG 文件名
#                     jpg_filepath = os.path.join(lung_folder, jpg_filename)
#                     image.save(jpg_filepath, 'JPEG') # 保存为 JPEG 格式的文件
#         except Exception as e:
#             print(f"转换 {file} 时出现错误：{e}")
#
# for root, dirs, _ in os.walk(output_folder, topdown=False):  # topdown=False，这样可以先遍历子目录
#     for name in dirs:
#         dir_path = os.path.join(root, name)
#         if not os.listdir(dir_path):  # 检查文件夹是否为空
#             os.rmdir(dir_path)  # 删除空文件夹
#             print(f"已删除空文件夹：{dir_path}")
#
# print("文件转换完成。")