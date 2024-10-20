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
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class BiGRU(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        # 定义双向 GRU 层
        self.bigru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        # 定义全连接层，将 GRU 的输出映射到所需的输出通道数
        self.fc = nn.Linear(hidden_size * 2, out_channels)

    def forward(self, x):
        # x 的形状应为 (sequence_length, batch_size, in_channels)
        # 初始隐藏状态为 None，表示使用零初始化
        gru_out, _ = self.bigru(x, None)
        # gru_out 的形状为 (sequence_length, batch_size, hidden_size * 2)
        # 取最后一个时间步的输出
        # last_time_step_out = gru_out[-1, :, :]
        # 通过全连接层
        # output = self.fc(last_time_step_out)

        return gru_out


class mdn_marnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.msc1 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.msc2 = nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2)
        self.msc3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(96, 32, kernel_size=1, stride=1)
        self.msdb1 = MultiScaleDenseBlock(32, 16, 6)
        self.tl1 = self._tranLayer(128, 64)
        self.msdb2 = MultiScaleDenseBlock(64, 16, 6)
        self.tl2 = self._tranLayer(160, 80)
        self.msdb3 = MultiScaleDenseBlock(80, 16, 6)
        self.tl3 = self._tranLayer(176, 88)
        self.msdb4 = MultiScaleDenseBlock(88, 16, 6)
        self.tl4 = self._tranLayer(184, 184)
        self.bigru1 = BiGRU(184, 64, 128)
        self.mhsa = nn.MultiheadAttention(128, 8)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        # 添加一个线性层，将输出通道数减半
        self.fc_reduce = nn.Linear(128, 64)
        self.bigru2 = BiGRU(64, 32, 64)
        self.fc2 = nn.Linear(64, 3)

    def _tranLayer(self, in_ch, out_ch):

        return nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1),
                             nn.AvgPool1d(kernel_size=2, stride=2),)

    def forward(self, x):
        outputs = {}
        outputs['input'] = x.shape
        out = self.conv1(x)
        outputs['conv1 shape'] = out.shape
        out1 = self.msc1(out)
        out2 = self.msc2(out)
        out3 = self.msc3(out)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv2(out)
        outputs['conv2 shape'] = out.shape
        out = self.msdb1(out)
        outputs['MSDB1 shape'] = out.shape
        out = self.tl1(out)
        outputs['transition Layer1 shape'] = out.shape
        out = self.msdb2(out)
        outputs['MSDB2 shape'] = out.shape
        out = self.tl2(out)
        outputs['transition Layer2 shape'] = out.shape
        out = self.msdb3(out)
        outputs['MSDB3 shape'] = out.shape
        out = self.tl3(out)
        outputs['transition Layer3 shape'] = out.shape
        out = self.msdb4(out)
        outputs['MSDB4 shape'] = out.shape
        out = self.tl4(out)
        outputs['transition Layer4 shape'] = out.shape
        # 调整输入形状为 (sequence_length, batch_size, channels)
        out = out.permute(2, 0, 1)
        out = self.bigru1(out)
        # 调整输出形状为 (batch_size, sequence_length, channels)
        out = out.permute(1, 0, 2)
        outputs['BiGRU Layer shape'] = out.shape
        # out, attn_output_weights = self.mhsa(out, out, out)
        out, _ = self.multihead_attn(out, out, out)
        outputs['MultiHeadSelfAttention shape'] = out.shape
        out = self.fc_reduce(out)
        outputs['FC shape'] = out.shape
        # 调整输入形状为 (sequence_length, batch_size, channels)
        out = out.permute(1, 0, 2)
        out = self.bigru2(out)
        out = out[-1, :, :]  # Use the output from the last time step
        outputs['BiGRU2 Layer shape'] = out.shape
        out = self.fc2(out)
        outputs['FC2 shape'] = out.shape

        for layer_name, shape in outputs.items():
            print(f'{layer_name}: {shape}')

        return out


if __name__ == '__main__':
    # Example usage:
    in_channels = 1
    growth_rate = 16
    num_layers = 6
    input_tensor = torch.randn(128, in_channels, 5000)  # (batch_size, channels, sequence_length)
    model = mdn_marnn()
    model(input_tensor)
    # print(model)
    # multi_scale_dense_block = MultiScaleDenseBlock(in_channels, growth_rate, num_layers)
    # input_tensor = torch.randn(1, in_channels, 100)  # (batch_size, channels, sequence_length)
    # output = multi_scale_dense_block(input_tensor)
    #
    # print(output.shape)
    # 创建模型实例
    # model = BiGRU(in_channels=184, hidden_size=64, out_channels=128)
    #
    # # 打印模型结构
    # print(model)
