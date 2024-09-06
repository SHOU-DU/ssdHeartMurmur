import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
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
        # conv_layers = []
        # self.bn0 = nn.BatchNorm2d(1)
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.pre = self._pre(1, 16)
        self.SK = SKConv(16, 16)
        # self.conv1 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.conv1 = nn.Sequential(
            # nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            depthwise_separable_conv(16, 16),
            # nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.AvgPool2d(2)
            nn.MaxPool2d(2)
        )
        # self.dp1 = nn.Dropout(p=0.05)
        # init.kaiming_normal_(self.conv1.weight, a=0.1)
        # self.conv1.bias.data.zero_()
        # conv_layers = [self.conv1, self.bn1, self.relu1, self.mp1]

        # Second Convolution Block
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Sequential(
            # nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            depthwise_separable_conv(16, 32),
            # nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.AvgPool2d(2)
            nn.MaxPool2d(2)
        )
        # self.dp2 = nn.Dropout(p=0.05)
        # init.kaiming_normal_(self.conv2.weight, a=0.1)
        # self.conv2.bias.data.zero_()


        # Third Convolution Block
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Sequential(
            # nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            depthwise_separable_conv(32, 64),
            # nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.AvgPool2d(2)
            nn.MaxPool2d(2)
        )
        # self.dp3 = nn.Dropout(p=0.05)
        # init.kaiming_normal_(self.conv3.weight, a=0.1)
        # self.conv3.bias.data.zero_()
        # conv_layers += [self.conv3, self.bn3, self.relu3]

        # Fourth Convolution Block
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Sequential(
            # nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            depthwise_separable_conv(64, 128),
            # nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.AvgPool2d(2)
            nn.MaxPool2d(2)
        )
        # init.kaiming_normal_(self.conv4.weight, a=0.1)
        # self.conv4.bias.data.zero_()
        # conv_layers += [self.conv4, self.bn4, self.relu4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.dp = nn.Dropout(p=0.2, inplace=False)

        # wide features
        # self.wide =nn.Sequential(
        #     nn.Linear(in_features=4, out_features=8),
            # nn.ReLU(),
            # # nn.Linear(in_features=16, out_features=32),
            # # nn.Dropout(0.05)
        # )
        self.lin = nn.Linear(in_features=128, out_features=3)
        # self.lin1 = nn.Linear(in_features=80, out_features=128)
        # Wrap the Convolutional Blocks
        # self.conv = nn.Sequential(*conv_layers)
        # self.dp = nn.Dropout(p=0.2)nn.ReLU(),nn.MaxPool2d((2,1))

    def _pre(self, input_channel, outchannel):
        pre = nn.Sequential(nn.ReLU(),
            nn.Conv2d(input_channel, outchannel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(outchannel),
            # # nn.BatchNorm2d(num_features=8),nn.ReLU(),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        # x = self.dp(x)
        x_all = x.view(x.shape[0], -1)

        # add wide features and concat two layers
        # print(x1.size())
        # y = self.wide(y)
        # x_all = torch.cat((x_all, y), dim=1)

        #
        # Linear layer
        x_all = self.lin(x_all)

        # Final output
        return x_all


class AudioClassifierFuse(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        self.pre = self._pre(1, 16)
        self.SK = SKConv(16, 16)
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
        # Run the convolutional blocks
        x = x.unsqueeze(1)  # 为输入特征添加通道，变为(batch_size, 1, height, width)
        xf1 = x[:, :, : 64, :]  # 时频域
        xf2 = x[:, :, 64:, :]  # 时域
        xf1 = self.pre(xf1)
        xf1 = self.SK(xf1)
        xf1 = self.conv1(xf1)
        xf1 = self.conv2(xf1)
        xf1 = self.conv3(xf1)
        xf1 = self.conv4(xf1)
        xf2 = self.ap2(xf2)
        xf2 = self.conv5(xf2)

        # 特征融合
        x_fuse = xf1 + xf2 + torch.mul(xf1, xf2)

        # 注意力
        # channel_attn = torch.sigmoid(self.channel_att(x_fuse))
        # x_attn = x_fuse * channel_attn

        # Adaptive pool and flatten for input to linear layer
        # x_attn = self.ap(x_attn)
        x_fuse = self.ap(x_fuse)
        x_all = x_fuse.view(x_fuse.shape[0], -1)

        # Linear layer
        x_all = self.lin(x_all)

        # Final output
        return x_all


class AudioClassifier2(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(nn.ReLU(),
                                nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
                                nn.BatchNorm2d(16),
                                # # nn.BatchNorm2d(num_features=8),nn.ReLU(),
                                nn.ReLU(inplace=True),
                                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )
        self.SK = SKConv(16, 16)
        # self.conv1 = nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.conv1 = nn.Sequential(
            # nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2)),
            depthwise_separable_conv(16, 16),
            # nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.AvgPool2d(2)
            nn.MaxPool2d(2)
        )


        # Second Convolution Block
        self.conv2 = nn.Sequential(
            depthwise_separable_conv(16, 32),
            # nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Third Convolution Block
        self.conv3 = nn.Sequential(
            # nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            depthwise_separable_conv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.AvgPool2d(2)
            nn.MaxPool2d(2)
        )

        # Fourth Convolution Block
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Sequential(
            # nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            depthwise_separable_conv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)


        #1Dconv
        self.conv1_1d =nn.Sequential(
            nn.Conv1d(1,16,3,1),
            # depthwise_separable_1Dconv(1,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.MaxPool1d(2)
        )
        self.conv2_1d = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1),
            # depthwise_separable_1Dconv(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.MaxPool1d(2)
        )
        self.conv3_1d = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1),
            # depthwise_separable_1Dconv(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.MaxPool1d(2)
        )
        self.conv4_1d = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1),
            # depthwise_separable_1Dconv(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.MaxPool1d(2)
        )
        #batch_first=True时，LSTM接受的数据维度为input(batch_size,特征数，通道数)
        self.lstm1 = nn.LSTM(128,128,batch_first=True)
        # self.lstm2 = nn.LSTM(256,128,batch_first=True)
        self.eca = ECANet(128)
        self.ap2 =  nn.AdaptiveAvgPool1d(output_size=1)

        self.lin = nn.Linear(in_features=128+128, out_features=3)



    # # ----------------------------
    # x:一维时域信号，y:二维特征
    # ----------------------------
    def forward(self, x,y):
        # Run the convolutional blocks
        x = x.unsqueeze(1)
        x = self.conv1_1d(x)
        x = self.conv2_1d(x)
        x = self.conv3_1d(x)
        x = self.conv4_1d(x)
        x = x.transpose(-1, -2) #(batch,channel,feat) ->(batch,feat,channel)
        x, _ = self.lstm1(x)
        x = x.transpose(-1, -2)
        x = self.eca(x)
        x = self.ap2(x)
        time_feat = x.view(x.shape[0], -1)

        y = y.unsqueeze(1)
        y = self.pre(y)
        y = self.SK(y)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.ap(y)
        fre_feat = y.view(x.shape[0], -1)
        all = torch.cat((time_feat, fre_feat), dim=1)
        out = self.lin(all)

        # Final output
        return out

class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool1d(1)
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x) #(N,C,W) ->(N,C,1)
        output = output.transpose(-1, -2) #(N,C,1) ->(N,1,C)
        output = self.con1(output).transpose(-1, -2) #(N,1,C) ->(N,C,1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output


if __name__ == "__main__":
    input1 = torch.rand(10, 12000)
    input2 = torch.rand(10, 64, 239)

    model = AudioClassifier2()
    # y = torch.rand(128, 8)
    output = model(input1, input2)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fK" % (total/1e3))
    input = torch.rand(128,1,12000)
    model = nn.Conv1d(7,32,3)
    model = nn.LSTM(256,128,batch_first=True,bidirectional=False)
    output,_ = model(input)
    print(output.shape)
    # model = AudioClassifierFuse()



