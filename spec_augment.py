import torch
import torch.nn.functional as F
import numpy as np
import os

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
    kfold = 5
    for i in range(kfold):
        fold = str(i) + '_fold'
        train_feature_folder = os.path.join(r'E:\sdmurmur\ssdHeartMurmur\feature_TF_TDF_60Hz_cut_zero',
                                            fold, r'feature\train_loggamma.npy')
        train_label_folder = os.path.join(r'E:\sdmurmur\ssdHeartMurmur\feature_TF_TDF_60Hz_cut_zero',
                                          fold, r'label\train_label.npy')
        # print(train_feature_folder)
        # print(train_label_folder)
        train_feature = np.load(train_feature_folder)
        torch_train_feature = torch.from_numpy(train_feature)  # 将numpy.ndarray转换成torch张量
        train_label = np.load(train_label_folder)
        torch_train_label = torch.from_numpy(train_label)  # 将numpy.ndarray转换成torch张量
        # 找到标签为2的索引
        indices = (torch_train_label == 2).nonzero(as_tuple=True)[0]
        # 对标签为2的特征进行频谱掩码
        masked_spec = spec_augment(torch_train_feature[indices], num_mask=2, freq_masking_max_percentage=0.15,
                                   time_masking_max_percentage=0.15)
        # 转回numpy数组
        masked_spec = masked_spec.numpy()
        # 拼接掩码后数据与原数据
        mask_train_feature = np.concatenate((train_feature, masked_spec), axis=0)
        # 拼接掩码后标签与原标签
        mask_train_label = np.concatenate((train_label, train_label[indices]), axis=0)
        # print(train_feature_folder[:-19])
        np.save(train_feature_folder[:-19] + r'\mask_train_loggamma.npy', mask_train_feature)
        # print(train_label_folder[:-16])
        np.save(train_label_folder[:-15] + r'\mask_train_label.npy', mask_train_label)
        # print(train_feature_folder)

    # train_feature_folder = r'E:\sdmurmur\ssdHeartMurmur\feature_TF_TDF_60Hz_cut_zero\0_fold\feature\train_loggamma.npy'
    # train_label_folder = r'E:\sdmurmur\ssdHeartMurmur\feature_TF_TDF_60Hz_cut_zero\0_fold\label\train_label.npy'
    # train_feature = np.load(train_feature_folder)
    # torch_train_feature = torch.from_numpy(train_feature)  # 将numpy.ndarray转换成torch张量
    # train_label = np.load(train_label_folder)
    # torch_train_label = torch.from_numpy(train_label)  # 将numpy.ndarray转换成torch张量
    #
    # # 找到标签为2的索引
    # indices = (torch_train_label == 2).nonzero(as_tuple=True)[0]
    # # 对标签为2的特征进行频谱掩码
    # masked_spec = spec_augment(torch_train_feature[indices], num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.15)
    # # 转回numpy数组
    # masked_spec = masked_spec.numpy()
    # # 拼接掩码后数据与原数据
    # mask_train_feature = np.concatenate((train_feature, masked_spec), axis=0)
    # # 拼接掩码后标签与原标签
    # mask_train_label = np.concatenate((train_label, train_label[indices]), axis=0)
    # print(f'origin train feature:{train_feature.shape}')
    # print(f'masked train feature:{masked_spec.shape}')
    # print(f'concatenate train feature:{mask_train_feature.shape}')
    # print(f'masked train label:{mask_train_label.shape}')

