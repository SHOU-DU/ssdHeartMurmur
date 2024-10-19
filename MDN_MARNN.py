import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(MultiScaleDenseLayer, self).__init__()
        k = growth_rate

        # Different convolutional kernels
        self.conv1x1 = nn.Conv1d(in_channels, 4 * k, kernel_size=1, stride=1, padding=0)
        self.conv1x3 = nn.Conv1d(4 * k, 4 * k, kernel_size=3, stride=1, padding=1)
        self.conv1x5 = nn.Conv1d(4 * k, 4 * k, kernel_size=5, stride=1, padding=2)

        # 1x1 convolution concatenate 1x3 and 1x5
        self.after_concat1 = nn.Conv1d(8 * k, k, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply different convolutional filters
        out_1x1 = self.conv1x1(x)
        out_1x3 = self.conv1x3(out_1x1)
        out_1x5 = self.conv1x5(out_1x1)

        # Concatenate the results from different convolution filters
        out = torch.cat([out_1x3, out_1x5], dim=1)

        # Apply 1x1 convolution to reduce dimensionality
        out = self.after_concat1(out)
        # concatenate with input
        # out = torch.cat((out, x), dim=1)
        return out


class MultiScaleDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(MultiScaleDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # Create multiple dense layers
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(MultiScaleDenseLayer(layer_in_channels, growth_rate))

    def forward(self, x):
        out1 = self.layers[0](x)
        out2 = self.layers[1](out1)
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


if __name__ == '__main__':
    # Example usage:
    in_channels = 32
    growth_rate = 16
    num_layers = 6

    multi_scale_dense_block = MultiScaleDenseBlock(in_channels, growth_rate, num_layers)
    input_tensor = torch.randn(1, in_channels, 100)  # (batch_size, channels, sequence_length)
    output = multi_scale_dense_block(input_tensor)

    print(output.shape)
