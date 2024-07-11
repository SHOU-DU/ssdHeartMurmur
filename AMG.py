import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.2):
        super(SepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x += residual
        return x


class SE(nn.Module):
    def __init__(self, in_channels):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), 1, 1)
        x = x * y
        return x


class SE_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(SE_ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            SE(out_channels)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x += residual
        return x


class SE_SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(SE_SepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.se = SE(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.resblock1 = ResBlock(32, 32)
        self.sepconv1 = SepConv(32, 32)
        self.resblock2 = ResBlock(32, 32)
        self.resblock3 = ResBlock(32, 32)
        self.resblock4 = ResBlock(32, 32)
        self.sepconv2 = SepConv(32, 32)
        self.resblock5 = SE_ResBlock(32, 32)
        self.resblock6 = SE_ResBlock(32, 32)
        self.resblock7 = SE_ResBlock(32, 32)
        self.resblock8 = SE_ResBlock(32, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.sepconv1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.sepconv2(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    model = MyModel()
    print(model)
