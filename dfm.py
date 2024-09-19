import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in // 2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


if __name__ == '__main__':
    from thop import profile

    model = DF_Module(16, 1, False)
    x1 = torch.randn(1, 16, 5, 239)
    x1 = F.interpolate(x1, size=(64, 239), mode='bilinear', align_corners=True)
    print(x1.shape)
    x2 = torch.randn(1, 16, 64, 239)
    y = model(x1, x2)
    flops, params = profile(model, inputs=(x1, x2))
    print(f"FLOPs: {flops}, Params: {params}")
    print(y.shape)

    # # 假设原始特征矩阵是5x239
    # original_features = np.random.rand(5, 239)  # 生成一个示例矩阵
    #
    # # 设置重复次数
    # repeat_count = 64 // 5  # 每行重复的次数
    # remainder = 64 % 5  # 计算剩余的行
    #
    # # 先进行重复扩张
    # expanded_features = np.repeat(original_features, repeat_count, axis=0)
    #
    # # 如果有剩余行，随机选择原始特征矩阵中的行来填充
    # if remainder > 0:
    #     additional_features = original_features[np.random.choice(original_features.shape[0], remainder, replace=True)]
    #     expanded_features = np.vstack((expanded_features, additional_features))
    #
    # # 确保最终形状是64x239
    # assert expanded_features.shape == (64, 239)
    #
    # print(expanded_features.shape)  # 输出应该是 (64, 239)

