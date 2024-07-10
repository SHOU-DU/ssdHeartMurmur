from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np


class NewDataset(Dataset):

    def __init__(self, wav_label, wav_data, wav_index):

        self.data = torch.from_numpy(wav_data)
        self.label = torch.LongTensor(wav_label)
        self.dex = torch.LongTensor(wav_index)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        data_item = self.data[index]
        label_item = self.label[index]
        dex_item = self.dex[index]

        return data_item.float(), label_item.int(), dex_item.int()

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)


class TrainDataset(Dataset):

    def __init__(self, wav_label, wav_data):

        self.data = torch.from_numpy(wav_data)
        self.label = torch.LongTensor(wav_label)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        data_item = self.data[index]
        label_item = self.label[index]

        return data_item.float(), label_item.int()

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)


class Dataset2(Dataset):

    def __init__(self, wav_label, wav_data,wav_index,demographic,static):

        self.data = torch.from_numpy(wav_data)
        self.label = torch.LongTensor(wav_label)
        self.dex = torch.LongTensor(wav_index)
        self.demo = torch.from_numpy(demographic)
        self.stat = torch.from_numpy(static)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        data_item = self.data[index]
        label_item = self.label[index]
        dex_item = self.dex[index]
        demo_item = self.demo[index]
        stat_item = self.stat[index]

        return data_item.float(), label_item.int(),dex_item.int(),demo_item.float(),stat_item.float()

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)


class MyDataset(Dataset):

    def __init__(self, wav_label, wav_data,wav_index,wav):
        self.data = torch.from_numpy(wav_data)
        self.label = torch.LongTensor(wav_label)
        self.wav = torch.from_numpy(wav)
        self.dex = torch.LongTensor(wav_index)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        data_item = self.data[index]
        wav_item = self.wav[index]
        label_item = self.label[index]

        dex_item = self.dex[index]


        return data_item.float(), label_item.int(), dex_item.int(), wav_item.float()

    def __len__(self):
        # 返回文件数据的数目
        return len(self.data)
####问题： location 和id 的长度不一样 导致后面DataLoader时，每个batch里 拼接不上

if __name__ == "__main__":
    fold_path = 'data_5fold_new2/1_fold'
    # fold_path = 'data'
    data_path = os.path.join(fold_path,'logmel')
    label_path = os.path.join(fold_path,'label')
    stat_feat_path = os.path.join(fold_path,'statistical_feature')

    data_train = np.load(data_path + '/train_feature.npy', allow_pickle=True)
    data_vali = np.load(data_path + '/vali_feature.npy', allow_pickle=True)

    label_train = np.load(label_path + '/train_label.npy', allow_pickle=True)
    label_vali = np.load(label_path + '/vali_label.npy', allow_pickle=True)

    index_train = np.load(label_path + '/train_index.npy', allow_pickle=True)
    index_vali = np.load(label_path + '/vali_index.npy', allow_pickle=True)

    demo_train = np.load(stat_feat_path + '/train_demographic.npy', allow_pickle=True)
    demo_vali = np.load(stat_feat_path + '/vali_demographic.npy', allow_pickle=True)

    stat_train = np.load(stat_feat_path + '/train_static.npy', allow_pickle=True)
    stat_vali = np.load(stat_feat_path + '/vali_static.npy', allow_pickle=True)
    vali_set = Dataset2(wav_label=label_vali, wav_data=data_vali, wav_index=index_vali,demographic=demo_vali,static =stat_vali)
    a = 1

