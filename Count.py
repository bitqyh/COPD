import numpy as np

# 打开文件并读取内容 统计各标签的数量
with open('ClsTxt/cls_num_train.txt', 'r') as file:
    lines = file.readlines()

# 初始化一个字典来存储标签及其出现次数
label_counts = {}

# 遍历每一行，统计标签数量
for line in lines:
    label = line.split(';')[0]  # 提取标签
    if label in label_counts:
        label_counts[label] += 1
    else:
        label_counts[label] = 1
# 打印每个标签及其数量
for label, count in label_counts.items():
    print(f"Label {label}: {count}")

