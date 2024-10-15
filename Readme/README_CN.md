# COPD Detection Project

## 项目简介
本项目旨在通过深度学习模型检测和分类慢性阻塞性肺疾病（COPD）。使用了ResNet18模型进行图像分类，并结合了数据增强和预处理技术。

## 安装说明
1. 克隆项目仓库：
    ```bash
    git clone https://github.com/bitqyh/copd-detection.git
    cd copd-detection
    ```

2. 创建并激活虚拟环境：
    ```bash
    conda create -n copd-env python=3.8
    conda activate copd-env
    ```

3. 安装依赖项：
    ```bash
    pip install -r requirements.txt
    ```

## 使用说明
1. 预处理数据：
    ```bash
    python DataSet.py
    ```

2. 训练模型：
    ```bash
    python main.py
    ```

3. 评估模型：
    ```bash
    python evaluate.py
    ```

## 文件说明
- `net.py`：定义了ResNet18模型的结构。
- `label.py`：处理标签数据。
- `DataSet.py`：进行数据预处理，包括直方图均衡化和伽马校正。
- `cutoff.py`：裁剪图像并进行形态学处理。
- `DeletePhoto.py`：删除不符合条件的图像。
- `Divide.py`：将数据集划分为训练集、验证集和测试集。
- `Combine.py`：合并图像以生成多通道输入。
- `main.py`：主训练脚本，包含模型训练和验证逻辑。

## 依赖项
- Python 3.8
- PyTorch
- torchvision
- numpy
- pandas
- opencv-python
- pydicom
- Pillow
- matplotlib
- seaborn
- openpyxl

## 贡献指南
欢迎贡献代码和报告问题。请提交Pull Request或在Issues中报告问题。