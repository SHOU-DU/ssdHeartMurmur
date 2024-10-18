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
        x = self.relu(self.bn1(self.conv1(x)))  # First conv layer
        x = self.resblock1(x)  # Residual blocks
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.gap(x)  # Global Average Pooling
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)  # Fully connected layer
        return x


if __name__ == '__main__':
    # Instantiate and test the model
    model = HeartSoundModel(num_classes=3)
    print(model)
    # Example input: (Batch_size, Channels, Height, Width) = (1, 1, 32, 239)
    input_tensor = torch.randn(10, 1, 64, 239)
    output = model(input_tensor)
    print(output.shape)
