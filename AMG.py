import torch
import torch.nn as nn
import torch.nn.functional as F

# Separable Convolution Block (Depthwise and Pointwise convolution)
class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SepConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Squeeze and Excitation block (optional based on the diagram)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x * out


# Residual Block as shown in the figure
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, out_channels)
        self.dropout1 = nn.Dropout(0.2)
        self.sepconv1 = SepConv(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)
        self.sepconv2 = SepConv(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.sepconv3 = SepConv(in_channels, out_channels)
        # 1x1 convolution to match dimensions if necessary
        # if in_channels != out_channels:
        #     self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        # else:
        #     self.conv1x1 = None

    def forward(self, x):
        identity = self.sepconv3(x)
        out = self.dropout1(self.bn1(x))
        out = self.bn2(self.sepconv1(out))
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.bn3(self.sepconv2(out))
        out = self.se(out)

        # Adjust identity dimensions if they don't match
        # if self.conv1x1 is not None:
        #     identity = self.conv1x1(identity)

        out += identity  # Residual connection
        return out


# Full model architecture based on the figure
class HeartSoundModel(nn.Module):
    def __init__(self, num_classes=1):
        super(HeartSoundModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks with different number of filters
        self.resblock1 = ResBlock(8, 8)
        self.resblock2 = ResBlock(8, 16)
        self.resblock3 = ResBlock(16, 32)
        self.resblock4 = ResBlock(32, 64)

        # Global Average Pooling and Final Fully Connected Layer
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x2 = x[:, :, : 64, :]  # 时频域
        x2 = self.relu(self.bn1(self.conv1(x2)))  # First conv layer
        x2 = self.resblock1(x2)  # Residual blocks
        x2 = self.resblock2(x2)
        x2 = self.resblock3(x2)
        x2 = self.resblock4(x2)
        x2 = self.gap(x2)  # Global Average Pooling
        x2 = torch.flatten(x2, 1)
        x2 = self.dropout(x2)
        x2 = self.fc(x2)  # Fully connected layer
        return x2


'''-------------一、SE模块-----------------------------'''
# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

#残差块
class resblock(nn.Module):  ## that is a part of model
    def __init__(self, inchannel, outchannel, kernel_size, stride, stride_1, stride_2):
        super(resblock, self).__init__()
        ## conv branch
        self.left = nn.Sequential(  ## define a serial of  operation
            # SKConv(inchannel),
            nn.BatchNorm2d(inchannel),
            nn.Dropout(p=0.05, inplace=False),
            self.SepConv(inchannel, outchannel, kernel_size, stride_1, padding=2),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05, inplace=False),
            self.SepConv(outchannel, outchannel, kernel_size, stride_2, padding=1),
            # SKConv(outchannel),
            nn.BatchNorm2d(outchannel)
        )
        self.SE = SE_Block(outchannel, ratio=8)  # 注意力机制
        # self.SK = SKConv(inchannel)
        ## shortcut branch
        self.short_cut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # print("short_cut is open ")
            self.short_cut = nn.Sequential(
                self.SepConv(inchannel, outchannel, kernel_size=1, stride=2)
            )
        # else:
        #     # print("short_cut is not open ")

    def SepConv(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        #         print(kernel_size, stride)
        SepCon = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        )
        return SepCon

    ### get the residual
    def forward(self, x):
        # x= self.SK(x)
        left_out = self.left(x)
        left_out = self.SE(left_out)

        #         left_out=self.SE(left_out)
        # print("left:", left_out.shape)
        short_out = self.short_cut(x)
        # print('short:', short_out.shape)
        return F.relu(left_out+short_out)


class AmgModel(nn.Module):
    def __init__(self, resblock, input_channel, num_class):
        super(AmgModel, self).__init__()
        self.pre = self._pre(input_channel, 8)
        # self.SK = SKConv(8)
        self.layer1 = self._makelayer(resblock, 8, 8, 2, 4, stride=2)
        self.layer2 = self._makelayer(resblock, 8, 16, 2, 4, stride=2)
        self.layer3 = self._makelayer(resblock, 16, 32, 2, 4, stride=2)
        self.layer4 = self._makelayer(resblock, 32, 64, 2, 4, stride=2)
        # self.layer4 = conv_1x1_bn(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化nn.AdaptiveAvgPool2d((1, 1))
        # self.pool = nn.AvgPool2d((2, 16), stride=1)  # 全局平均池化
        # self.pool = nn.AvgPool2d((2, 5), stride=1)  # 全局平均池化
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(64, num_class)
        # self.wide = nn.Linear(12, 20)
        # self.soft = nn.Softmax(dim=1)
        # for m in self.modules():eight, mode='fan_out', nonlinearity='relu')
        #         #     elif isinstance(m, nn.BatchNorm2
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.wd):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def _pre(self, input_channel, outchannel):
        pre = nn.Sequential(
            nn.Conv2d(input_channel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            # nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
        )
        return pre

    def _makelayer(self, resblock, inchannel, outchannel, blockmum, kernel_size, stride=1):
        strides = [stride] + [1] * (blockmum - 1)
        layers = []
        channel = inchannel
        for i in range(blockmum):
            # print(channel, outchannel, kernel_size, strides[i], '1', strides[i])
            layers.append(resblock(channel, outchannel, kernel_size, strides[i], 1, strides[i]))
            channel = outchannel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # 将输入数据的通道数扩展为1
        # x = self.SK(x)
        x1 = self.pre(x)
        # x1 = self.SK(x1)
        #         print(x1.shape)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.pool(x5)
        x6 = self.drop(x6)
        x6 = x6.view(x6.size(0), -1)
        # y = self.wide(y)
        # x6 = torch.cat((x6, y), dim=1)
        x6 = self.fc(x6)
        # x6 = self.soft(x6)
        return x6


if __name__ == '__main__':
    # Instantiate and test the model
    # model = HeartSoundModel(num_classes=3)
    model = AmgModel(resblock, input_channel=1, num_class=3)
    print(model)
    # Example input: (Batch_size, Channels, Height, Width) = (1, 1, 32, 239)
    input_tensor = torch.randn(10, 64, 239)
    output = model(input_tensor)
    print(output.shape)
