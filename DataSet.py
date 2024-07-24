import os
import cv2
import pydicom
from PIL import Image
import numpy as np

input_folder = r"E:\桌面\文件\慢阻肺\health\CT5522152"
output_folder = r"E:\桌面\文件\慢阻肺\test"
#  直方图均衡化
def histogram_equalization(img):
    img_equalized = cv2.equalizeHist(img)
    return img_equalized

# 伽马校正
def gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_corrected = cv2.LUT(img, table)
    return img_corrected
def extract_mediastinal_window(pixel_data):
    mediastinal_win_width = 350
    mediastinal_win_center = 40
    img_temp = pixel_data.copy()  # 复制像素数据，以免修改原始数据
    min_val = mediastinal_win_center - mediastinal_win_width / 2
    max_val = mediastinal_win_center + mediastinal_win_width / 2
    img_temp = np.clip(img_temp, min_val, max_val)
    actual_min_val = np.min(img_temp)
    actual_max_val = np.max(img_temp)
    img_temp = ((img_temp - actual_min_val) / (actual_max_val - actual_min_val) * 255.0).astype(np.uint8)

    return img_temp

def extract_lung_window(pixel_data):
    lung_win_width = 1500
    lung_win_center = -400

    img_temp = pixel_data.copy()  # 复制像素数据，以免修改原始数据
    min_val = lung_win_center - lung_win_width / 2
    max_val = lung_win_center + lung_win_width / 2
    img_temp = np.clip(img_temp, min_val, max_val)
    actual_min_val = np.min(img_temp)
    actual_max_val = np.max(img_temp)
    img_temp = ((img_temp - actual_min_val) / (actual_max_val - actual_min_val) * 255.0).astype(np.uint8)
    return img_temp

# def extract_lung_from_mediastinal(lung_img, mediastinal_img):
#     lung_img = lung_img.astype(np.int16)
#     mediastinal_img = mediastinal_img.astype(np.int16)
#     lung_only_img = lung_img - mediastinal_img
#     lung_only_img = np.clip(lung_only_img, 0, 255).astype(np.uint8)
#     return lung_only_img

for root, _, files in os.walk(input_folder):  # 遍历输入文件夹中的所有文件
    folder_name = os.path.basename(root) # 获取文件夹名
    lung_folder = os.path.join(output_folder, "Health", folder_name)  # 创建一个名为 LUNG/CT* 的子文件夹
    os.makedirs(lung_folder, exist_ok=True)  # 如果文件夹不存在，则创建它
    # if os.listdir(lung_folder):  # 如果文件夹不为空
    #     print(f"文件夹 {lung_folder} 不为空，跳过。")
    #     continue
    for file in files:
        dicom_filepath = os.path.join(root, file) # 获取 DICOM 文件的完整路径
        try:
            dicom_file = pydicom.dcmread(dicom_filepath)  # 读取 DICOM 文件
            if 'PixelData' in dicom_file:
                series_description = dicom_file.SeriesDescription.lower()  # 转换为小写字母
                if "lung" in series_description and "5.0" in series_description: # 确保是肺部的 CT 扫描图像
                    pixel_data = dicom_file.pixel_array.astype(np.int16)  # 提取像素数据
                    # max_value = np.max(pixel_data)
                    # min_value = np.min(pixel_data)
                    #
                    # print(f"最大值是：{max_value}")
                    # print(f"最小值是：{min_value}")
                    winwidth = dicom_file.WindowWidth
                    wincenter = dicom_file.WindowCenter
                    pixel_data = extract_lung_window(pixel_data)  # 提取肺窗
                    max_med_value = np.max(pixel_data)
                    min_med_value = np.min(pixel_data)

                    # img_equalized = histogram_equalization(pixel_data) # 直方图均衡化
                    # #img_corrected = gamma_correction(img_equalized, gamma=1.2)
                    # gamma_corrected = np.power(img_equalized / 255.0, 2.2) * 255.0 # 伽马校正


                    image = Image.fromarray(pixel_data)  # 将处理后的图像数据转换为 PIL 图像对象
                    image = image.convert('RGB')  # 转换为 RGB 格式
                    jpg_filename = f"{folder_name}_{os.path.splitext(file)[1]}.jpg" # 创建 JPEG 文件名
                    jpg_filepath = os.path.join(lung_folder, jpg_filename)
                    image.save(jpg_filepath, 'JPEG') # 保存为 JPEG 格式的文件
        except Exception as e:
            print(f"转换 {file} 时出现错误：{e}")

for root, dirs, _ in os.walk(output_folder, topdown=False):  # topdown=False，这样可以先遍历子目录
    for name in dirs:
        dir_path = os.path.join(root, name)
        if not os.listdir(dir_path):  # 检查文件夹是否为空
            os.rmdir(dir_path)  # 删除空文件夹
            print(f"已删除空文件夹：{dir_path}")

print("文件转换完成。")
"""
 'CT.1.2.840.113619.2.334.3.17433297.80.1641175861.742.11'
首先，导入了所需的库，包括 os、pydicom、PIL 的 Image 和 numpy。  
然后，定义了输入和输出文件夹的路径。  
创建了一个名为 lung_folder 的输出子文件夹，如果该文件夹不存在，则创建它。  
定义了一个名为 extract_lung_window 的函数，该函数接受一个参数 pixel_data，并返回一个处理过的图像数据。这个函数的主要目的是将输入的图像数据调整到肺窗的范围内，肺窗的宽度和中心值分别设定为 1500 和 -400。这个函数首先将图像数据剪切到肺窗的范围内，然后将其归一化到 0-255 的范围，并转换为 8 位无符号整数格式。  
在主程序部分，代码遍历了输入文件夹中的所有文件。对于每个文件，它首先尝试读取 DICOM 文件。如果文件中包含 'PixelData'，则提取出该数据，并将其转换为 16 位整数格式。  
然后，检查 DICOM 文件的 SeriesDescription 字段是否包含 "lung" 和 "5.0"。如果包含，那么就认为这个文件是肺部的 CT 扫描图像。  
对于这样的文件，代码会调用 extract_lung_window 函数处理图像数据，然后将处理后的数据转换为 PIL 图像对象。  
最后，将这个图像对象保存为 JPEG 格式的文件。文件名由 DICOM 文件所在的文件夹名和 DICOM 文件的文件名组成。  
如果在处理某个文件时出现错误，代码会捕获异常并打印出错误信息。  
当所有文件都处理完毕后，打印出 "文件转换完成。"。


from concurrent.futures import ThreadPoolExecutor

def process_file(filepath):
    # 您的文件处理逻辑在这里
    pass

with ThreadPoolExecutor(max_workers=4) as executor:
    for root, _, files in os.walk(input_folder):
        for file in files:
            dicom_filepath = os.path.join(root, file)
            executor.submit(process_file, dicom_filepath)
"""
