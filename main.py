import torch
from torch import nn

from Loss.FocalLoss import FocalLoss
from net import resnet18
# from torch.utils.data import DataLoader  # 工具取黑盒子，用函数来提取数据集中的数据（小批次）
from DataEnhance.Data import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pkgutil
import os

enhance = 'Data'
lossfunction = 'FocalLoss'

writer = SummaryWriter('trainlog/logs')


with open('ClsTxt/cls_num_test.txt', 'r') as f:
    lines_test = f.readlines()
with open('ClsTxt/cls_num_val.txt', 'r') as f:
    lines_val = f.readlines()
with open('ClsTxt/cls_num_train.txt', 'r') as f:
    lines_train = f.readlines()

np.random.seed(10101)  # 生成随机数
np.random.shuffle(lines_test)
np.random.shuffle(lines_val)
np.random.shuffle(lines_train)
np.random.seed(None)

# '''取部分数据'''
# rate = 4
# lines_test = lines_test[:len(lines_test) // rate]
# lines_val = lines_val[:len(lines_val) // rate]
# lines_train = lines_train[:len(lines_train) // rate]



input_shape = [224, 224]  # 导入图像大小
train_data = DataGenerator(lines_train, input_shape, True)  # 训练集
val_data = DataGenerator(lines_val, input_shape, False)  # 验证集
test_data = DataGenerator(lines_test, input_shape, False)  # 测试集
val_len = len(val_data)
test_len = len(test_data)
print(val_len)  # 返回测试集长度
print(test_len)  # 返回验证集长度
# 取黑盒子工具

"""加载数据"""
batch_size: int = 256

'''读取数据'''
gen_train = DataLoader(train_data, batch_size)  # 训练集batch_size读取小样本，规定每次取多少样本
gen_val = DataLoader(val_data, batch_size)  # 测试集读取小样本
gen_test = DataLoader(test_data, batch_size)

'''构建网络'''
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")  # 电脑主机的选择
net = resnet18(True, progress=True, num_classes=4)  # 定于分类的类别
net.to(device)

'''选择优化器和学习率的调整方法'''
lr = 0.0001  # 定义学习率
weight_decay = 0.01
# 创建优化器，并添加权重衰减参数
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
sculer = torch.optim.lr_scheduler.StepLR(optim, step_size=1)

'''训练'''
epochs = 50  # 读取数据次数，每次读取顺序方式不同

inputs = torch.randn(1, 3, 224, 224).to(device)
writer.add_graph(net, inputs)  # 画图
best_val_loss = float('inf')

'''定义早停参数'''
best_accuracy = 0.65
patience = 10  # 定义耐心参数
patience_counter = 0  # 初始化耐心计数器

'''所有标签和预测结果'''
all_labels = []  # 存放所有标签
all_preds = []  # 存放所有预测结果
all_indices = []  # 存放所有索引
all_paths = []  # 存放所有路径


'''模型方法'''

'''定义损失函数'''
criterion = FocalLoss(alpha= 1, gamma= 2, reduction='mean')
# criterion = AdaptiveFocalLoss(net, alpha=1, gamma=2, weight_decay=0.01)
def get_folder_path_and_file_count(file_path):
    # Extract the folder path from the file path
    folder_path = os.path.dirname(file_path)

    # Count the number of files in the folder
    # Exclude '.' and '..' entries and directories
    files_in_folder = [f for f in os.listdir(folder_path) if not f.startswith('.') and not f == '..']

    return len(files_in_folder)




for epoch in range(0, epochs):
    total_train = 0  # 定义总损失



    for data in gen_train:
        img, label, paths = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
        optim.zero_grad()
        output = net(img)
        # train_loss = nn.CrossEntropyLoss()(output, label).to(device)
        train_loss = criterion(output, label).to(device)
        train_loss.backward()  # 反向传播
        optim.step()  # 优化器更新
        total_train += train_loss  # 损失相加



        # all_indices.extend([idx] * len(label))
        # all_paths.extend(paths)

    sculer.step()
    total_val = 0  # 总损失
    total_accuracy = 0  # 总精度
    for data in gen_val:
        img, label, paths = data  # 图片转数据
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            optim.zero_grad()  # 梯度清零
            out = net(img)  # 投入网络
            # test_loss = nn.CrossEntropyLoss()(out, label).to(device)
            test_loss = criterion(out, label).to(device)
            total_val += test_loss  # 测试损失，无反向传播
            preds = out.argmax(1)
            accuracy = ((preds == label).sum()).clone().detach().cpu().numpy()  # 正确预测的总和比测试集的长度，即预测正确的精度
            total_accuracy += accuracy

            # Store true labels and predictions
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    validation_accuracy = total_accuracy / val_len
    writer.add_scalars('Loss', {'Training Loss': total_train, 'Validation Loss': total_val}, epoch)
    writer.add_scalar('Validation Accuracy', total_accuracy / val_len, epoch)
    print("训练集上的损失：{}".format(total_train))
    print("验证集上的损失：{}".format(total_val))
    print("验证集上的精度：{:.1%}".format(total_accuracy / val_len))  # 百分数精度，正确预测的总和比测试集的长度

    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        patience_counter = 0  # 重置耐心计数器
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optim.state_dict(),
            #设置保存路径

        },

            f'PTH/checkpoint_radom_{batch_size}_{enhance}_{lossfunction}.pth')
    else:
        patience_counter += 1  # 增加耐心计数器

    if patience_counter >= patience:
        print("早停：在第{}轮停止训练".format(epoch))
        break  # 达到耐心阈值，停止训练


    # 计算每个类的精确率召回率和F1分数
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # 显示混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

    # 打印每个类的精确率、召回率和F1分数
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i} - Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f:.4f}")

    writer.close()
# torch.save({
#     'epoch': epoch,
#     'state_dict': net.state_dict(),
#     'optimizer': optim.state_dict(),
# }, f'checkpoint_radom_{batch_size}.pth')

print("模型已保存")

# torch.save(net.state_dict(),"DogandCat{}.pth".format(epoch+1))
# 在训练结束后，计算测试集上的损失
total_test_loss = 0
total_test_accuracy = 0
for data in gen_test:
    img, label = data
    with torch.no_grad():
        img = img.to(device)
        label = label.to(device)
        out = net(img)
        # test_loss = nn.CrossEntropyLoss()(out, label).to(device)
        test_loss = criterion(out, label).to(device)
        total_test_loss += test_loss
        accuracy = ((out.argmax(1) == label).sum()).clone().detach().cpu().numpy()
        total_test_accuracy += accuracy

print("测试集上的损失：{}".format(total_test_loss))
print("测试集上的精度：{:.1%}".format(total_test_accuracy / test_len))

'''

/***
 *                    _ooOoo_
 *                   o8888888o
 *                   88" . "88
 *                   (| -_- |)
 *                    O\ = /O
 *                ____/`---'\____
 
 *              .   ' \\| |// `.
 *               / \\||| : |||// \
 *             / _||||| -:- |||||- \
 *               | | \\\ - /// | |
 *             | \_| ''\---/'' | |
 *              \ .-\__ `-` ___/-. /
 *           ___`. .' /--.--\ `. . __
 *        ."" '< `.___\_<|>_/___.' >'"".
 *       | | : `- \`.;`\ _ /`;.`/ - ` : | |
 *         \ \ `-. \_ __\ /__ _/ .-` / /
 * ======`-.____`-.___\_____/___.-`____.-'======
 *                    `=---='
 *
 * .............................................
 *          佛祖保佑             永无BUG
 */
'''
