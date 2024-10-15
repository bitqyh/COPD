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

                    winwidth = dicom_file.WindowWidth
                    wincenter = dicom_file.WindowCenter
                    pixel_data = extract_lung_window(pixel_data)  # 提取肺窗
                    max_med_value = np.max(pixel_data)
                    min_med_value = np.min(pixel_data)



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
