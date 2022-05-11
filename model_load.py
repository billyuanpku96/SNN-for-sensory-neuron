import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from spikingjelly import visualizing
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.utils.tensorboard import SummaryWriter
import sys
#prameters

dataset_dir = 'mnist'
batch_size = int(256)
learning_rate = float(0.1)  # 初始学习率
Vth = 1.4
Vreset = 0.85
T = 50
Rs = 4.0e3
Rd = 2.2e3
Cm = 5e-6
# tau = 8e-4  #C = 200nf，R =4k
#train_epoch = 15
dt = 3.5e-6
# 计算RC等效电阻
R = 1 / (1 / Rs + 1 / Rd)
tau = R * Cm

test_net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 14 * 14, bias=False),
    neuron.LIFNode2(tau=tau, v_threshold=Vth, v_reset=Vreset),
    nn.Linear(14 * 14, 10, bias=False),
    neuron.LIFNode2(tau=tau, v_threshold=Vth, v_reset=Vreset)
)

print('Loading Model, please wait......')
test_net.load_state_dict(torch.load('./try22/model_epoch1.pth'))
print('Model loaded successfully!')
print(test_net)


def To_frequency(x):  # transform Force to the value frenquency
    y = (96.6297 * (x ** 3) - 287.16965 * (x ** 2) + 340.35347 * x + 104.85552) * (10 ** 3)
    return y


def To_zhouqi(x):
    y = 1 / x
    return y


def To_fire_pro(x):
    y = dt / x
    return y


def Poisson(x):
    out_spike = torch.rand_like(x).le(x).to(x)
    return out_spike

def frenqucy_spike(x):
    out_spike = Poisson(To_fire_pro(To_zhouqi(To_frequency(x))))
    return out_spike


encoder = encoding.PoissonEncoder()

# 初始化数据加载器
train_dataset = torchvision.datasets.MNIST(
    root=dataset_dir,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(root=dataset_dir, train=False,
                                          transform=torchvision.transforms.ToTensor(), download=True)

train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)
test_data_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False)

list_num_spike = []
for i in range(10):
    list_num_spike.append([0])
    list_num_spike[i].append(torch.zeros(10))
    list_num_spike[i].append(torch.zeros(10))



test_net.eval()
with torch.no_grad():
    for img_test, label_test in test_data_loader:

        spike_num_img_test = torch.zeros(batch_size, 10)
        for t in range(T):
            if t == 0:
                spike_num_img_test = test_net(frenqucy_spike(img_test).float())
                # out_spikes_counter = net(frequence_code_spiking(img).float())
            else:
                spike_num_img_test += test_net(frenqucy_spike(img_test).float())

            # spike_num_img_test  【batch_size,10】 第一行为第一张图，10个输出神经元各自总的发放次数
        pred_label = F.one_hot(spike_num_img_test.max(1)[1], num_classes=10)  # 找到最大的发放率神经元的index，变成One_hot编码
        # 这样就得到了预测的标签(计算出来的标签)
        # pred_label 是(batch_size,10)  对应着batch_size个标签
        for j in range(label_test.size(0)):  # 遍历这个batch里所有的标签
            index = label_test[j]  # 将当前遍历到的标签给index
            list_num_spike[index][0] += 1  # 通过index判断这个标签是哪一类，然后在这类的个数上加一，用于统计每个类的总个数
            list_num_spike[index][1] += pred_label[j]  # 将遍历到的这个标签对应的图片得到的预测标签 放到实际标签对应那类的 第一个
            list_num_spike[index][2] += spike_num_img_test[j]  # 将遍历到的图片的输出，累加到对应类的第二个tensor里，
            # 意思就是  比如第4个类， 的第二个tensor 记录了所有属于这个类的图片的输出的累加值，即输入所有这个类的图，每个输出神经元发放次数总和   (1,10)

        print('ok')
        functional.reset_net(test_net)

with open('./try22/hotmap_epoch1.txt','a') as f2:
    for i in range(len(list_num_spike)):
        s = str(list_num_spike[i][0]) + '\n' + str(list_num_spike[i][1].numpy()).replace('[','').replace(']','') +'\n\n'+str(list_num_spike[i][2].numpy()).replace('[','').replace(']','') +'\n\n\n\n\n'
        f2.write(s)
