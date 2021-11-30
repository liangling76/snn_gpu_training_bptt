import torch
import torch.nn as nn
import torch.nn.functional as F
from snnlib_func import time_step
import snnlib_op as sop


class CIFAR_BN(nn.Module):
    def __init__(self):
        super(CIFAR_BN, self).__init__()

        self.conv1_1 = sop.Conv(3,   128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_2 = sop.BnFireConv(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_3 = sop.BnFireConv(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) 

        self.conv2_1 = sop.BnFireConv(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv2_2 = sop.BnFireConv(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv2_3 = sop.BnFireConv(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) 

        self.conv3_1 = sop.BnFireConv(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv3_2 = sop.BnFireConv(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv3_3 = sop.BnFireConv(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) 

        self.fc1 = sop.BnFireFc(4*4*512, 1024, dim_bn=(0,1,3,4), cin_bn=512) 
        self.fc2 = sop.BnFireFc(1024, 512, dim_bn=(0,1))
        self.bn  = sop.BnFire(512, dim_bn=(0,1))
        self.fc3 = nn.Linear(512,  10)

        self.bn_layer = [
            self.conv1_2, self.conv1_3, 
            self.conv2_1, self.conv2_2, self.conv2_3,
            self.conv3_1, self.conv3_2, self.conv3_3,
            self.fc1, self.fc2, self.bn
            ]

    def set_bn(self, train_bn):
        for layer in self.bn_layer:
            layer.train_bn = train_bn

    def forward(self, x_):

        x = torch.stack([x_ for _ in range(time_step)], dim=0)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)


        x = self.fc1(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.fc3(x.mean(dim=0))

        return x


class Alexnet_BN_DIST(nn.Module):
    def __init__(self):
        super(Alexnet_BN_DIST, self).__init__()

        self.conv1 = sop.Conv(3,   64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = sop.BnFireConvDist(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) 
        self.conv3 = sop.BnFireConvDist(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) 
        self.conv4 = sop.BnFireConvDist(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) 
        self.conv5 = sop.BnFireConvDist(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv6 = sop.BnFireConvDist(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv7 = sop.BnFireConvDist(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) 

        self.fc1 = sop.BnFireFcDist(7*7*256, 4096, dim_bn=(0,1,3,4), cin_bn=256)
        self.fc2 = sop.BnFireFcDist(4096, 4096, dim_bn=(0,1))
        self.fire = sop.Fire()
        self.fc3 = nn.Linear(4096,  1000)

        self.bn_layer = [self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.fc1, self.fc2]

    def set_bn(self, train_bn):
        for layer in self.bn_layer:
            layer.train_bn = train_bn

    def forward(self, x_):
        x = torch.stack([x_ for _ in range(time_step)], dim=0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)


        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fire(x)
        x = self.fc3(x.mean(dim=0))

        return x



class ResidualDist(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1), downsample=None):
        super(ResidualDist, self).__init__()

        self.conv1 = sop.FireConv(cin, cout, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.conv2 = sop.BnFireConvDist(cout, cout, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = sop.BnDist(cout, dim_bn=(0,1,3,4))
        self.downsample = downsample
        self.bn_layer = [self.conv2, self.bn2]

    def forward(self, input): # input is the result of bn
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            x += self.downsample(input)
        else:
            x += input

        return x



class Resnet_DIST(nn.Module):
    def __init__(self):
        super(Resnet_DIST, self).__init__()

        self.pre_conv = sop.Conv(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.pre_bn = sop.BnDist(64, dim_bn=(0,1,3,4))

        self.bn_layer = [self.pre_bn]

        self.layer1 = self.make_layer(64,   64, 3, stride=(2, 2))
        self.layer2 = self.make_layer(64,  128, 4, stride=(2, 2))
        self.layer3 = self.make_layer(128, 256, 6, stride=(2, 2))
        self.layer4 = self.make_layer(256, 512, 3, stride=(2, 2))

        self.fire = sop.Fire()  
        self.fc = nn.Linear(512, 1000)


    def make_layer(self, cin, cout, block_num, stride=(1, 1)):
        downsample = nn.Sequential(
            sop.FireConv(cin, cout, kernel_size=(1, 1), stride=stride, padding=(0, 0)),
            sop.BnDist(cout, (0,1,3,4))
        )
        self.bn_layer += [downsample[1]]

        layers = []
        layers.append(ResidualDist(cin, cout, stride, downsample))
        for _ in range(1, block_num):
            layers.append(ResidualDist(cout, cout))
        
        for layer in layers:
            self.bn_layer += layer.bn_layer

        return nn.Sequential(*layers)


    def set_bn(self, train_bn):
        for layer in self.bn_layer:
            layer.train_bn = train_bn


    def forward(self, x_):
        x = torch.stack([x_ for _ in range(time_step)], dim=0)

        x = self.pre_conv(x)
        x = self.pre_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fire(x)

        x = x.mean(dim=0)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        return x
