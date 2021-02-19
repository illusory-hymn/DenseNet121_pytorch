##   DenseNet网络是在2017的论文 Densely Connected Convolutional Networks
##  (1) 由于密集连接方式，DenseNet提升了梯度的反向传播，使得网络更容易训练 （每层可以直达最后的误差信号）
##  (2) 参数更小且计算更高效 （concatenate来实现特征复用，计算量很小）
##  (3) 由于特征复用，分类器使用到了低级特征

import torch
import torchvision
import torch.nn as nn 
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
from torch.autograd import Variable

##  输入in_channels个特征图，输出in_channels+growth_rate个特征图
##  每次DenseLayer后通道数增加growth_rate个
##  继承了nn.Sequential，所以不用些forward，自动实现将add_module包装在Sequential中
class _DenseLayer(nn.Sequential): 
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(in_channels)) ## self.add_module形式可以实现动态调整，进行替换相同名称的模块，而不用重写forward
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels, bn_size * growth_rate,  ##  1x1卷积
                                            kernel_size=1, bias=False)) 
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate,   ##  3x3卷积
                                            kernel_size=3, padding=1, bias=False))
    ##  重载forward函数                                        
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x) ##  这段的意思是调用所有add_module添加到一个Sequential模块中
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i,
                                        growth_rate, bn_size))

class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                bn_size=4, theta=0.5, num_classes=10):
        super(DenseNet, self).__init__()

        ##  初始的卷积为filter: 2倍的growth_rate
        num_init_filture = 2 * growth_rate

        ##  表示如果是CIFAR-10数据集
        if num_classes == 10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_filture,
                                    kernel_size=3, padding=1, bias=False))
            ]))
        else: 
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_filture,
                                    kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_filture)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))
        num_feature = num_init_filture
        for i,num_layers in enumerate(block_config):
            ##  通道数num_init_filture -> num_init_filture + growth_rate*num_layers(将这个更新为num_feature)
            self.features.add_module('denseblock%d' % (i+1),
                                    _DenseBlock(num_layers, num_feature,
                                                bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            ##   每一个Dense Block解一个Transition模块，通过卷积和池化进行wh压缩
            ##  通过卷积将通道数从  num_feature-> num_feature * theta 进行压缩
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i+1),
                                        _Transition(num_feature,
                                                    int(num_feature * theta)))
                num_feature = int(num_feature * theta)
        
        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1))) ##  表示压缩到（1，1）
        
        self.classifier = nn.Linear(num_feature, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out

def DenseNet121():
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000)

def DenseNet169():
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=1000)

##  导入权重
densenet121 = DenseNet121().cuda()
densenet121.eval() ## 预测一定要加这个，不然结果差异很大
weights_path = 'densenet121-a639ec97.pth'
state_dict = torch.load(weights_path)

##   下面注释掉的内容用于调试检测
'''
##  导入权重使发现名称不匹配，用于检测是否匹配
for v, k in densenet121.state_dict().items():  ## 输出自己模型的层名
    print(v)
for v, k in state_dict.items(): ## 输出导入权重的层名
    print(v)
'''

##  发现权重中layers名称多了个.，接下来用代码进行修改
new_state_dict = OrderedDict()
# 修改 key
for k, v in state_dict.items():
    if 'denseblock' in k:
        param = k.split(".")
        k = ".".join(param[:-3] + [param[-3]+param[-2]] + [param[-1]]) ## '.'.join表示后面的内容通过.相连
    new_state_dict[k] = v
densenet121.load_state_dict(new_state_dict)

##  读入待预测的图片
def img_processing(img_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ##  Normalize参数是因为ImageNet数据集，我们使用的权重是这个数据集训练得到的
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    img = Variable(img).cuda()
    return img

##  预测
img_path = '0.jpg'
img = img_processing(img_path)
output = densenet121(img)
idx = np.argmax(output.cpu().data.numpy())
##  读入label.txt
txt_path = 'label.txt'
labels = []
with open(txt_path, 'r', encoding='utf-8') as f: ## 因为要读入中文，所以要加上encoding='utf-8'
    for lines in f:
        labels.append(lines[:-1])   
print("class:{}".format(idx))
print("name:{}".format(labels[idx]))
