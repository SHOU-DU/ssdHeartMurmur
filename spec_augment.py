import torch
import torch.nn.functional as F
import numpy as np


def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15):
    """
    对频谱图应用频谱掩码（SpecAugment）。

    :param spec: 输入的频谱图，形状为 (batch_size, num_channels, num_freq, num_time)
    :param num_mask: 每个频谱图上应用的掩码数量
    :param freq_masking_max_percentage: 频率掩码的最大百分比
    :param time_masking_max_percentage: 时间掩码的最大百分比
    :return: 应用了频谱掩码后的频谱图
    """
    spec = spec.clone()  # 避免修改原始频谱图
    batch_size, num_freq, num_time = spec.shape

    for i in range(batch_size):
        for _ in range(num_mask):
            # 频率掩码
            freq_percentage = np.random.uniform(low=0.0, high=freq_masking_max_percentage)
            num_freq_to_mask = int(freq_percentage * num_freq)
            f0 = np.random.randint(0, num_freq - num_freq_to_mask)
            spec[i, f0:f0 + num_freq_to_mask, :] = 0

            # 时间掩码
            time_percentage = np.random.uniform(low=0.0, high=time_masking_max_percentage)
            num_time_to_mask = int(time_percentage * num_time)
            t0 = np.random.randint(0, num_time - num_time_to_mask)
            spec[i, :, t0:t0 + num_time_to_mask] = 0

    return spec


if __name__ == "__main__":
    # 示例频谱图
    batch_size = 2
    num_channels = 1
    num_freq = 128
    num_time = 100
    spec = torch.randn(batch_size, num_freq, num_time)

    # 应用频谱掩码
    # masked_spec = spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15)
    #
    # print(masked_spec.shape)  # 输出: torch.Size([2, 1, 128, 100])

    train_feature_folder = r'E:\sdmurmur\ssdHeartMurmur\feature_TF_TDF_60Hz_cut_zero\0_fold\feature\train_loggamma.npy'
    train_label_folder = r'E:\sdmurmur\ssdHeartMurmur\feature_TF_TDF_60Hz_cut_zero\0_fold\label\train_label.npy'
    train_feature = np.load(train_feature_folder)
    torch_train_feature = torch.from_numpy(train_feature)  # 将numpy.ndarray转换成torch张量
    train_label = np.load(train_label_folder)
    torch_train_label = torch.from_numpy(train_label)  # 将numpy.ndarray转换成torch张量

    # 找到标签为2的索引
    indices = (torch_train_label == 2).nonzero(as_tuple=True)[0]
    # 对标签为2的特征进行频谱掩码
    masked_spec = spec_augment(torch_train_feature[indices], num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15)
    print(train_feature.shape)
    # 转回numpy数组
    masked_spec = masked_spec.numpy()
    print(masked_spec.shape)
    # print(train_label)
