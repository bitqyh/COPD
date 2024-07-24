from torch.utils.data import DataLoader, WeightedRandomSampler

def make_weights_for_balanced_classes(labels, nclasses):
    count = [0] * nclasses
    for label in labels:
        count[label] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(labels)
    for idx, label in enumerate(labels):
        weight[idx] = weight_per_class[label]
    return weight

# 假设dataset是你的数据集对象，它需要有一个能返回类别信息的方式，例如dataset.targets或者自定义的方法
# 假设nclasses是类别的总数
# weights = make_weights_for_balanced_classes(dataset.targets, nclasses)
# weights = torch.DoubleTensor(weights)
# sampler = WeightedRandomSampler(weights, len(weights))
#
# train_loader = DataLoader(dataset, batch_size=10, sampler=sampler)


# # 打开文件并读取内容 统计各标签的数量
# with open('cls_num_train.txt', 'r') as file:
#     lines = file.readlines()
#
# # 初始化一个字典来存储标签及其出现次数
# label_counts = {}
#
# # 遍历每一行，统计标签数量
# for line in lines:
#     label = line.split(';')[0]  # 提取标签
#     if label in label_counts:
#         label_counts[label] += 1
#     else:
#         label_counts[label] = 1
#
# # 打印每个标签及其数量
# for label, count in label_counts.items():
#     print(f"Label {label}: {count}")