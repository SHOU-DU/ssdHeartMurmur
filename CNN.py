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


class AudioClassifierFuseODconv(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        self.pre = self._pre(1, 16)
        self.pre2 = self._pre(1, 16)
        self.ODconv1 = ODConv2d(16, 16, 3, padding=1)
        self.ODconv2 = ODConv2d(16, 16, 3, padding=1)
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
        xf1 = x[:, :, :64, :]  # 时频域
        xf2 = x[:, :, 64:, :]  # 时域
        xf2 = F.interpolate(xf2, size=(64, 239), mode='bilinear', align_corners=True)
        # outputs['xf1 shape'] = xf1.shape
        # outputs['xf2 shape'] = xf2.shape
        xf1 = self.pre(xf1)
        xf2 = self.pre2(xf2)
        # outputs['pre xf1'] = xf1.shape
        # outputs['pre xf2'] = xf2.shape
        xf1 = self.ODconv1(xf1)  # (32, 120)
        # outputs['ODconv1'] = xf1.shape
        xf2 = self.ODconv2(xf2)
        # outputs['ODconv2'] = xf2.shape
        # 先融合再送入DSC
        x_fuse = self.dfm(xf1, xf2)
        # outputs['fuse'] = x_fuse.shape
        x_fuse = self.conv1(x_fuse)
        # outputs['conv1'] = x_fuse.shape
        x_fuse = self.conv2(x_fuse)
        # outputs['conv2'] = x_fuse.shape
        x_fuse = self.conv3(x_fuse)
        # outputs['conv3'] = x_fuse.shape
        x_fuse = self.conv4(x_fuse)
        # outputs['conv4'] = x_fuse.shape
        # Adaptive pool and flatten for input to linear layer
        x_fuse = self.ap(x_fuse)
        # outputs['ap'] = x_fuse.shape
        x_all = x_fuse.view(x_fuse.shape[0], -1)
        # outputs['flatten'] = x_all.shape

        # Linear layer
        x_all = self.lin(x_all)
        # outputs['output'] = x_all.shape

        # for layer_name, shape in outputs.items():
        #     print(f'{layer_name}: {shape}')

        # # Final output
        return x_all


class AudioClassifierODconv(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
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
        self.conv5 = nn.Conv2d(1, 128, kernel_size=1, stride=1, padding=0)
        # 注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1, bias=False),
        )

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.ap2 = nn.AdaptiveAvgPool2d(output_size=(2, 7))
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
        xf2 = self.pre(xf2)
        # outputs['pre'] = xf1.shape
        xf2 = self.ODconv(xf2)
        # outputs['ODconv'] = xf1.shape
        xf2 = self.conv1(xf2)
        # outputs['conv1'] = xf1.shape
        xf2 = self.conv2(xf2)
        # outputs['conv2'] = xf1.shape
        xf2 = self.conv3(xf2)
        # outputs['conv3'] = xf1.shape
        xf2 = self.conv4(xf2)
        # outputs['conv4'] = xf1.shape
        # xf2 = self.ap2(xf2)
        # outputs['ap2'] = xf2.shape
        # xf2 = self.conv5(xf2)
        # outputs['conv5'] = xf2.shape

        # Adaptive pool and flatten for input to linear layer
        # x_attn = self.ap(x_attn)
        x_fuse = self.ap(xf2)
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


# 引入两个ODConv模块
class AudioClassifierODconv2(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        self.pre = self._pre(1, 16)
        self.ODconv = ODConv2d(16, 16, 3, padding=1)
        self.ODconv2 = ODConv2d(16, 32, 3, padding=1)
        self.conv1 = nn.Sequential(
            depthwise_separable_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            depthwise_separable_conv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            depthwise_separable_conv(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=512, out_features=3)


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
        xf1 = x[:, :, :, :239]  # 时频域
        xf2 = x[:, :, :, 239:]  # 时域或GAF或cwt
        xf1 = self.pre(xf1)
        # outputs['pre'] = xf1.shape
        xf1 = self.ODconv(xf1)
        # outputs['ODconv'] = xf1.shape
        xf1 = self.ODconv2(xf1)
        # outputs['ODconv2'] = xf1.shape
        xf1 = self.conv1(xf1)
        # outputs['conv1'] = xf1.shape
        xf1 = self.conv2(xf1)
        # outputs['conv2'] = xf1.shape
        xf1 = self.conv3(xf1)
        # outputs['conv3'] = xf1.shape
        xf1 = self.conv4(xf1)
        # outputs['conv4'] = xf1.shape

        # Adaptive pool and flatten for input to linear layer
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



class AudioClassifierConcatFeatureODconv(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
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
        x = self.pre(x)
        # outputs['pre'] = x.shape
        x = self.ODconv(x)
        # outputs['ODconv'] = x.shape
        x = self.conv1(x)
        # outputs['conv1'] = x.shape
        x = self.conv2(x)
        # outputs['conv2'] = x.shape
        x = self.conv3(x)
        # outputs['conv3'] = x.shape
        x_fuse = self.conv4(x)
        # outputs['conv4'] = x_fuse.shape

        # Adaptive pool and flatten for input to linear layer
        x_fuse = self.ap(x_fuse)
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


if __name__ == "__main__":
    input1 = torch.rand(10, 12000)
    input2 = torch.rand(10, 64, 239)
    model = AudioClassifierFuseODconv()
    X = torch.rand(10, 1, 64, 239)
    X2 = torch.rand(10, 69, 239)
    output = model(X2)
