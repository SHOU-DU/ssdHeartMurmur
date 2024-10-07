import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from odconv import ODConv2d
from dfm import DF_Module
import numpy as np


class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        # print(x.shape)
        return x


class depthwise_separable_1Dconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_1Dconv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv1d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv1d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        # print(x.shape)
        return x


class SKConv(nn.Module):
    def __init__(self, inchannels,channels, branches=3, reduce=2, stride=1, len=64):
        super(SKConv, self).__init__()
        len = max(int(inchannels // reduce), len)
        self.convs = nn.ModuleList([])
        for i in range(branches):
            self.convs.append(nn.Sequential(
                nn.Conv2d(inchannels, channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i,
                          bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, len, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(len),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv2d(len, channels, kernel_size=1, stride=1,bias=False)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = [conv(x) for conv in self.convs]
        x = torch.stack(x, dim=1)
        attention = torch.sum(x, dim=1)
        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        attention = self.softmax(attention)
        x = torch.sum(x * attention, dim=1)
        return x


class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.pre = self._pre(1, 16)
        self.SK = SKConv(16, 16)
        self.conv1 = nn.Sequential(
            depthwise_separable_conv(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Second Convolution Block
        self.conv2 = nn.Sequential(
            # nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            depthwise_separable_conv(16, 32),
            # nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Third Convolution Block
        self.conv3 = nn.Sequential(
            depthwise_separable_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Fourth Convolution Block
        self.conv4 = nn.Sequential(
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=128, out_features=3)

    def _pre(self, input_channel, outchannel):
        pre = nn.Sequential(nn.ReLU(),
            nn.Conv2d(input_channel, outchannel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
        return pre
    # # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = x.unsqueeze(1)
        x = self.pre(x)
        x = self.SK(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x_all = x.view(x.shape[0], -1)

        x_all = self.lin(x_all)

        # Final output
        return x_all


class AudioClassifierODconv(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    # 2024/09/24 PM 单特征gamma_tone输入ODConv网络
    def __init__(self):
        super().__init__()
        self.pre = self._pre(1, 16)
        self.ODconv = ODConv2d(16, 16, 3, padding=1)
        self.conv1 = nn.Sequential(
            depthwise_separable_conv(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            depthwise_separable_conv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            depthwise_separable_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=128, out_features=3)


    def _pre(self, input_channel, outchannel):
        pre = nn.Sequential(nn.ReLU(),
            nn.Conv2d(input_channel, outchannel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
        return pre
    # # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # 记录每层的输出
        # outputs = {}
        # Run the convolutional blocks
        x = x.unsqueeze(1)  # 为输入特征添加通道，变为(batch_size, 1, height, width)
        # outputs['input'] = x.shape
        xf1 = x[:, :, : 64, :]  # 时频域
        xf2 = x[:, :, 64:, :]  # 时域或GAF或cwt
        xf1 = self.pre(xf1)
        # outputs['pre'] = xf1.shape
        xf1 = self.ODconv(xf1)
        # outputs['ODconv'] = xf1.shape
        xf1 = self.conv1(xf1)
        # outputs['conv1'] = xf1.shape
        xf1 = self.conv2(xf1)
        # outputs['conv2'] = xf1.shape
        xf1 = self.conv3(xf1)
        # outputs['conv3'] = xf1.shape
        xf1 = self.conv4(xf1)
        # outputs['conv4'] = xf1.shape
        # xf2 = self.ap2(xf2)
        # outputs['ap2'] = xf2.shape
        # xf2 = self.conv5(xf2)
        # outputs['conv5'] = xf2.shape

        # Adaptive pool and flatten for input to linear layer
        # x_attn = self.ap(x_attn)
        x_fuse = self.ap(xf1)
        # outputs['ap'] = x_fuse.shape
        x_all = x_fuse.view(x_fuse.shape[0], -1)
        # outputs['flatten'] = x_all.shape

        # Linear layer
        x_all = self.lin(x_all)
        # outputs['output'] = x_all.shape

        # for layer_name, shape in outputs.items():
        #     print(f'{layer_name}: {shape}')

        # Final output
        return x_all


class AudioClassifierFuseODconv(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    # 2024/09/24 PM 14:55 将时域特征xf2重复复制为(1,1,64,239)视作另一个模态，与xf1经过相同网络结构后进行通道层面拼接，最后进入全连接层
    # 通道层面拼接选择128+128;128+64;128+32
    def __init__(self):
        super().__init__()
        self.pre = self._pre(1, 16)
        # self.pre2 = self._pre(1, 16)
        self.pre3 = self._pre(1, 7)
        self.pre2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(1, 7, kernel_size=(1, 3)),
            nn.BatchNorm2d(7),
            nn.ReLU(inplace=True)
        )
        self.ODconv1 = ODConv2d(16, 16, 3, padding=1)
        self.ODconv2 = ODConv2d(16, 16, 3, padding=1)
        self.ODconv3 = ODConv2d(7, 7, 3, padding=1)
        self.dfm = DF_Module(16, 16, reduction=False)
        self.conv1 = nn.Sequential(
            depthwise_separable_conv(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            depthwise_separable_conv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            depthwise_separable_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv21 = nn.Sequential(
            depthwise_separable_conv(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv22 = nn.Sequential(
            depthwise_separable_conv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv23 = nn.Sequential(
            depthwise_separable_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv24 = nn.Sequential(
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv31 = nn.Sequential(
            depthwise_separable_conv(16, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv32 = nn.Sequential(
            depthwise_separable_conv(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv33 = nn.Sequential(
            depthwise_separable_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv34 = nn.Sequential(
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.ap2 = nn.AdaptiveAvgPool2d(output_size=1)
        self.ap3 = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=128, out_features=3)
        self.lin2 = nn.Linear(in_features=5, out_features=3)  # 时域特征判断loud
        self.lin_fuse = nn.Linear(in_features=135, out_features=3)


    def _pre(self, input_channel, outchannel):
        pre = nn.Sequential(nn.ReLU(),
            nn.Conv2d(input_channel, outchannel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )
        return pre
    # # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # 记录每层的输出
        # outputs = {}
        # Run the convolutional blocks
        x = x.unsqueeze(1)  # 为输入特征添加通道，变为(batch_size, 1, height, width)
        # outputs['input'] = x.shape
        xf1 = x[:, :, :64, :]  # 时频域
        # xf2 = x[:, :, 64:, :]  # 时域包络
        xf2 = x[:, :, 64:128, :]  # MFCC
        xf3 = x[:, :, 128:135, :]  # 时域包络+均值方差
        # outputs['xf1 shape'] = xf1.shape
        # outputs['xf2 shape'] = xf2.shape
        xf1 = self.pre(xf1)
        # outputs['pre xf1'] = xf1.shape
        # MM1时频域
        xf1 = self.ODconv1(xf1)  # (32, 120)
        # outputs['ODconv1'] = xf1.shape
        xf1 = self.conv1(xf1)
        xf1 = self.conv2(xf1)
        xf1 = self.conv3(xf1)
        xf1 = self.conv4(xf1)
        xf1 = self.ap(xf1)
        xf1 = xf1.view(xf1.shape[0], -1)
        # xf1_r = self.lin(xf1)
        # outputs['flatten xf1'] = xf1.shape
        # # MM2MFCC
        # xf2 = self.pre2(xf2)
        # xf2 = self.ODconv2(xf2)
        # xf2 = self.conv21(xf2)
        # xf2 = self.conv22(xf2)
        # xf2 = self.conv23(xf2)
        # xf2 = self.conv24(xf2)
        # xf2 = self.ap2(xf2)
        # xf2 = xf2.view(xf2.shape[0], -1)
        # MM3时域包络+均值方差
        xf3 = self.pre2(xf3)
        # xf3 = self.ODconv3(xf3)
        # xf3 = self.conv31(xf3)
        # xf3 = self.conv32(xf3)
        # xf3 = self.conv33(xf3)
        # xf3 = self.conv34(xf3)
        xf3 = self.ap3(xf3)
        xf3 = xf3.view(xf3.shape[0], -1)
        # xf2_r = self.lin2(xf2)
        # x_sum = 0.7*xf1_r + 0.3*xf2_r
        # 融合
        # x_cat = torch.cat((xf1, xf2, xf3), dim=1)
        x_cat = torch.cat((xf1, xf3), dim=1)
        # outputs['x_fuse'] = x_fuse.shape
        x_cat = self.lin_fuse(x_cat)
        # x_fuse = 0.0*x_sum + 1.0*x_cat
        # outputs['x_fuse shape'] = x_fuse.shape

        # for layer_name, shape in outputs.items():
        #     print(f'{layer_name}: {shape}')

        # Final output
        return x_cat


if __name__ == "__main__":
    model = AudioClassifierFuseODconv()
    X = torch.rand(10, 1, 64, 239)
    X2 = torch.rand(128, 160, 239)
    output = model(X2)

    # # 假设 x1 和 x2 的形状都是 [256, 3]
    # x1 = torch.randn(256, 3)
    # x2 = torch.randn(256, 3)
    #
    # # 获取每行的最大值及其对应的索引
    # max_values_x1, indices_x1 = torch.max(x1, dim=1)
    # max_values_x2, indices_x2 = torch.max(x2, dim=1)
    #
    # # 找到索引相同的行
    # matching_indices = torch.eq(indices_x1, indices_x2)
    #
    # # 获取索引相同行的序号
    # matching_rows = torch.where(matching_indices)[0]
    #
    # print("每行最大值及其索引：")
    # print("x1:", max_values_x1, indices_x1)
    # print("x2:", max_values_x2, indices_x2)
    #
    # print("\n索引相同的行序号：", matching_rows)

    # X3 = torch.rand(1, 1, 5, 239)
    # num_copies = 64 // 5 + 1  # 因为原维度大小为 5，需要复制 12 次才能达到 64
    #
    # # 复制 dim=2 的维度
    # X_repeated = X3.repeat(1, 1, num_copies, 1)
    # X_new = X_repeated[:, :, :64, :]
    # print(X_new.shape)
