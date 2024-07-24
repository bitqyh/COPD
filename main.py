import torch
import torch.nn as nn
from net import resnet18
from torch.utils.data import DataLoader  # 工具取黑盒子，用函数来提取数据集中的数据（小批次）
from data import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from oversampling import make_weights_for_balanced_classes
import os

writer = SummaryWriter('trainlog/logs')

with open('cls_num_test.txt', 'r') as f:
    lines_test = f.readlines()
with open('cls_num_val.txt', 'r') as f:
    lines_val = f.readlines()
with open('cls_num_train.txt', 'r') as f:
    lines_train = f.readlines()
np.random.seed(10101)  # 生成随机数
np.random.shuffle(lines_test)
np.random.shuffle(lines_val)
np.random.shuffle(lines_train)
np.random.seed(None)

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
batch_size: int = 128
weights_train = make_weights_for_balanced_classes(train_data.targets, 4)
weights_val = make_weights_for_balanced_classes(val_data.targets, 4)
weights_test = make_weights_for_balanced_classes(test_data.targets, 4)

weights_train = torch.DoubleTensor(weights_train)
weights_val = torch.DoubleTensor(weights_val)
weights_test = torch.DoubleTensor(weights_test)

sampler_train = WeightedRandomSampler(weights_train, len(weights_train))
sampler_val = WeightedRandomSampler(weights_val, len(weights_val))
sampler_test = WeightedRandomSampler(weights_test, len(weights_test))

gen_train = DataLoader(train_data, batch_size, sampler=sampler_train)  # 训练集batch_size读取小样本，规定每次取多少样本
gen_val = DataLoader(val_data, batch_size, sampler=sampler_val)  # 测试集读取小样本
gen_test = DataLoader(test_data, batch_size, sampler=sampler_test)

# gen_train = DataLoader(train_data, batch_size)  # 训练集batch_size读取小样本，规定每次取多少样本
# gen_val = DataLoader(val_data, batch_size)  # 测试集读取小样本
# gen_test = DataLoader(test_data, batch_size)
# train_data_list = list(DataLoader(train_data, batch_size=64))  # 训练集batch_size读取小样本，规定每次取多少样本
# val_data_list = list(DataLoader(val_data, batch_size=64))  # 测试集读取小样本
# test_data_list = DataLoader(test_data, batch_size=64)
'''构建网络'''
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")  # 电脑主机的选择
net = resnet18(True, progress=True, num_classes=4)  # 定于分类的类别
net.to(device)
'''选择优化器和学习率的调整方法'''
lr = 0.0001  # 定义学习率
weight_decay = 0.01
#optim = torch.optim.Adam(net.parameters(), lr=lr)  # 导入网络和学习率
# 创建优化器，并添加权重衰减参数
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
sculer = torch.optim.lr_scheduler.StepLR(optim, step_size=1)
'''训练'''
epochs = 50  # 读取数据次数，每次读取顺序方式不同

inputs = torch.randn(1, 3, 224, 224).to(device)
writer.add_graph(net, inputs)  # 画图
best_val_loss = float('inf')
best_accuracy = 0
patience = 20  # 定义耐心参数
patience_counter = 0  # 初始化耐心计数器

def load_checkpoint(net, optim, filename=f'checkpoint_radom_{batch_size}.pth'):
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)  #
        start_epoch = checkpoint['epoch']  #
        net.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return net, optim, start_epoch


# Before starting the training
# net, optim, start_epoch = load_checkpoint(net, optim)

for epoch in range(0, epochs):
    total_train = 0  # 定义总损失
    for data in gen_train:
        img, label = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
        optim.zero_grad()
        output = net(img)
        train_loss = nn.CrossEntropyLoss()(output, label).to(device)
        train_loss.backward()  # 反向传播
        optim.step()  # 优化器更新
        total_train += train_loss  # 损失相加

    sculer.step()
    total_val = 0  # 总损失
    total_accuracy = 0  # 总精度
    for data in gen_val:
        img, label = data  # 图片转数据
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            optim.zero_grad()  # 梯度清零
            out = net(img)  # 投入网络
            test_loss = nn.CrossEntropyLoss()(out, label).to(device)
            total_val += test_loss  # 测试损失，无反向传播
            accuracy = ((out.argmax(1) == label).sum()).clone().detach().cpu().numpy()  # 正确预测的总和比测试集的长度，即预测正确的精度
            total_accuracy += accuracy

    # validation_accuracy = total_accuracy / val_len
    writer.add_scalars('Loss', {'Training Loss': total_train, 'Validation Loss': total_val}, epoch)
    writer.add_scalar('Validation Accuracy', total_accuracy / val_len, epoch)
    print("训练集上的损失：{}".format(total_train))
    print("验证集上的损失：{}".format(total_val))
    print("验证集上的精度：{:.1%}".format(total_accuracy / val_len))  # 百分数精度，正确预测的总和比测试集的长度

    if total_val > best_accuracy:
        best_val_loss = total_val
        patience_counter = 0  # 重置耐心计数器
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optim.state_dict(),
        }, f'checkpoint_radom_{batch_size}.pth')
    else:
        patience_counter += 1  # 增加耐心计数器

    if patience_counter >= patience:
        print("早停：在第{}轮停止训练".format(epoch))
        break  # 达到耐心阈值，停止训练
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
        test_loss = nn.CrossEntropyLoss()(out, label).to(device)
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
