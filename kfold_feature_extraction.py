import os
import math
import shutil

import torch

from helper_code import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from python_speech_features import logfbank
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import wave
import librosa.display
import librosa
import soundfile
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
from spafe.utils.converters import erb2hz
from spafe.utils.vis import show_spectrogram
from spafe.utils.preprocessing import SlidingWindow
from dataset2kfold import *
from pyts.image import GramianAngularField


def save_kfold_feature(kfold_folder, tdf_feature, feature_folder, kfold=int):
    for i in range(kfold):
        kfold_feature_out = os.path.join(feature_folder, str(i) + "_fold")  #
        kfold_folder_train = os.path.join(kfold_folder, str(i) + "_fold" + r"\train_data")
        kfold_folder_vali = os.path.join(kfold_folder, str(i) + "_fold" + r"\vali_data")  # 五折交叉验证，最后通过测试集测试
        tdf_train_folder = os.path.join(tdf_feature, str(i) + "_fold" + r"\train_data")  # 时域特征
        tdf_vali_folder = os.path.join(tdf_feature, str(i) + "_fold" + r"\vali_data")
        label_dir = os.path.join(kfold_feature_out, "label")
        feature_dir = os.path.join(kfold_feature_out, "feature")
        # 制作存储特征和标签的文件夹
        if not os.path.exists(kfold_feature_out):
            os.makedirs(kfold_feature_out)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        train_feature = Log_GF_GAF(kfold_folder_train)
        # train_feature = Log_GF_TDF(kfold_folder_train, tdf_train_folder)

        train_label, train_location, train_id = get_label(kfold_folder_train)  # 获取各个3s片段label和听诊区位置和个体ID
        train_index = get_index(kfold_folder_train)

        vali_feature = Log_GF_GAF(kfold_folder_vali)
        # vali_feature = Log_GF_TDF(kfold_folder_vali, tdf_vali_folder)

        vali_label, vali_location, vali_id = get_label(kfold_folder_vali)
        vali_index = get_index(kfold_folder_vali)
        # 存储训练集特征和标签
        np.save(feature_dir + r'\train_loggamma.npy', train_feature)
        np.save(label_dir + r'\train_label.npy', train_label)
        np.save(label_dir + r'\train_location.npy', train_location)
        np.save(label_dir + r'\train_id.npy', train_id)
        np.save(label_dir + r'\train_index.npy', train_index)
        # 存储验证集特征标签等信息
        np.save(feature_dir + r'\vali_loggamma.npy', vali_feature)
        np.save(label_dir + r'\vali_label.npy', vali_label)
        np.save(label_dir + r'\vali_location.npy', vali_location)
        np.save(label_dir + r'\vali_id.npy', vali_id)
        np.save(label_dir + r'\vali_index.npy', vali_index)
        print("train_feature shape:", train_feature.shape)  # train_feature shape: (样本数：14649, 滤波器数：64, 3s段数据帧数：239)
        print("train_label shape:", train_label.shape)
        print("vali_feature shape:", vali_feature.shape)  # train_feature shape: (样本数：14649, 滤波器数：64, 3s段数据帧数：239)
        print("vali_label shape:", vali_label.shape)
        print(f"第{i}折特征提取完毕")


def Log_GF(data_directory):
    loggamma = list()
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Log_GF feature:'):  # 加tqdm可视化特征提取过程
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)
            x = x / np.max(np.abs(x))
            # gfreqs为经过gammatone滤波器后得到的傅里叶变换矩阵
            gSpec, gfreqs = erb_spectrogram(x,
                                            fs=fs,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.025, 0.0125, "hamming"),
                                            nfilts=64,
                                            nfft=512,
                                            low_freq=25,
                                            high_freq=2000)
            fbank_feat = gSpec.T
            fbank_feat = np.log(fbank_feat)
            fbank_feat = feature_norm(fbank_feat)

            # fbank_feat = feature_norm(fbank_feat)
            # fbank_feat = delt_feature(fbank_feat)
            loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


def Log_GF_TDF(data_directory, TDF_directory):  # 提取时频域和时域特征
    loggamma = list()
    tdfnum = 0
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Log_GF and TDF feature:'):  # 加tqdm可视化特征提取过程
        tdfnum = tdfnum + 1
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)
            x = x / np.max(np.abs(x))
            # gfreqs为经过gammatone滤波器后得到的傅里叶变换矩阵
            gSpec, gfreqs = erb_spectrogram(x,
                                            fs=fs,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.025, 0.0125, "hamming"),
                                            nfilts=64,
                                            nfft=512,
                                            low_freq=25,
                                            high_freq=2000)
            fbank_feat = gSpec.T
            fbank_feat = np.log(fbank_feat)
            fbank_feat = feature_norm(fbank_feat)
            # 读取对应的.csv文件
            csv_file = os.path.join(TDF_directory, root + '.csv')
            if os.path.exists(csv_file):
                # csv_data = np.genfromtxt(csv_file, delimiter=',', dtype=None, encoding='utf-8')
                csv_data = np.loadtxt(csv_file, delimiter=',')
                num_cols_to_add = fbank_feat.shape[1] - csv_data.shape[1]
                csv_data_pad = np.pad(csv_data, ((0, 0), (0, num_cols_to_add)), mode='constant', constant_values=0)
                # 拼接.wav文件特征和.csv文件数据
                combined_feat = np.concatenate((fbank_feat, csv_data_pad), axis=0)
                loggamma.append(combined_feat)
                csv_data = None
                csv_data_pad = None
                combined_feat = None

            else:
                print(f"CSV file not found for {f}")

            # loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


def Log_GF_GAF(data_directory):  # 提取时频域和GAF特征
    loggamma = list()
    tdfnum = 0
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Log_GF and GAF feature:'):  # 加tqdm可视化特征提取过程
        tdfnum = tdfnum + 1
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)  # 心音信号已归一化为[-1, 1]

            # GAF特征参数设置
            image_size = 239  # 图像（数组）大小与提取gammatone特征后的帧数匹配
            gasf = GramianAngularField(image_size=image_size, method='summation')
            x_data = np.array([x])  # 将x转化成1x12000的二维形式
            x_gasf = gasf.fit_transform(x_data)  # 得到[sample_num, img_size, img_size]形式GAF特征

            x = x - np.mean(x)
            x = x / np.max(np.abs(x))
            # gfreqs为经过gammatone滤波器后得到的傅里叶变换矩阵
            gSpec, gfreqs = erb_spectrogram(x,
                                            fs=fs,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.025, 0.0125, "hamming"),
                                            nfilts=64,
                                            nfft=512,
                                            low_freq=25,
                                            high_freq=2000)
            fbank_feat = gSpec.T
            fbank_feat = np.log(fbank_feat)
            fbank_feat = feature_norm(fbank_feat)

            combined_feat = np.concatenate((fbank_feat, x_gasf[0]), axis=0)
            loggamma.append(combined_feat)

        else:
            continue
    return np.array(loggamma)


def GF(data_directory):
    loggamma = list()
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' GF feature:'):  # 加tqdm可视化特征提取过程
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)
            x = x / np.max(np.abs(x))
            # gfreqs为经过gammatone滤波器后得到的傅里叶变换矩阵
            gSpec, gfreqs = erb_spectrogram(x,
                                            fs=fs,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.025, 0.0125, "hamming"),
                                            nfilts=64,
                                            nfft=512,
                                            low_freq=25,
                                            high_freq=2000)
            fbank_feat = gSpec.T
            # fbank_feat = np.log(fbank_feat)
            fbank_feat = feature_norm(fbank_feat)

            # fbank_feat = feature_norm(fbank_feat)
            # fbank_feat = delt_feature(fbank_feat)
            loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


def log10_GF(data_directory):
    loggamma = list()
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Log10_GF feature:'):  # 加tqdm可视化特征提取过程
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)
            x = x / np.max(np.abs(x))
            # gfreqs为经过gammatone滤波器后得到的傅里叶变换矩阵
            gSpec, gfreqs = erb_spectrogram(x,
                                            fs=fs,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.025, 0.0125, "hamming"),
                                            nfilts=64,
                                            nfft=512,
                                            low_freq=25,
                                            high_freq=2000)
            fbank_feat = gSpec.T
            fbank_feat = np.log10(fbank_feat)
            fbank_feat = feature_norm(fbank_feat)

            # fbank_feat = feature_norm(fbank_feat)
            # fbank_feat = delt_feature(fbank_feat)
            loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


def Aweight_Log_GF(data_directory):
    loggamma = list()
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + 'Aweight_Log_GF feature:'):  # 加tqdm可视化特征提取过程
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)  # 将音信信号置为0均值
            # x = x / np.max(np.abs(x))
            # gfreqs为经过gammatone滤波器后得到的傅里叶变换矩阵
            gSpec, gfreqs = erb_spectrogram(x,
                                            fs=fs,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.025, 0.0125, "hamming"),
                                            nfilts=64,
                                            nfft=512,
                                            low_freq=25,
                                            high_freq=2000)
            gamma_fbanks_mat, gcenfreqs = gammatone_filter_banks(nfilts=64,
                                                                   nfft=512,
                                                                   fs=fs,
                                                                   low_freq=25,
                                                                   high_freq=2000)
            center_freqs = [erb2hz(freq) for freq in gcenfreqs]
            # print(center_freqs)
            gSpecLog = 20 * (np.log10(gSpec))  # 计算后的gammatone频谱取对数得到结果类似于声压级
            Aweight_gSpec = A_weight(gSpecLog, center_freqs)  # 进行类似A计算操作
            fbank_feat = Aweight_gSpec.T
            # fbank_feat = np.log(fbank_feat)
            fbank_feat = feature_norm(fbank_feat)

            # fbank_feat = feature_norm(fbank_feat)
            # fbank_feat = delt_feature(fbank_feat)
            loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


def Cweight_Log_GF(data_directory):
    loggamma = list()
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Cweight_Log_GF feature:'):  # 加tqdm可视化特征提取过程
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)  # 将音信信号置为0均值
            # x = x / np.max(np.abs(x))
            # gfreqs为经过gammatone滤波器后得到的傅里叶变换矩阵
            gSpec, gfreqs = erb_spectrogram(x,
                                            fs=fs,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.025, 0.0125, "hamming"),
                                            nfilts=64,
                                            nfft=512,
                                            low_freq=25,
                                            high_freq=2000)
            gamma_fbanks_mat, gcenfreqs = gammatone_filter_banks(nfilts=64,
                                                                   nfft=512,
                                                                   fs=fs,
                                                                   low_freq=25,
                                                                   high_freq=2000)
            center_freqs = [erb2hz(freq) for freq in gcenfreqs]
            # print(center_freqs)
            gSpecLog = 20 * (np.log10(gSpec))  # 计算后的gammatone频谱取对数得到结果类似于声压级
            Cweight_gSpec = C_weight(gSpecLog, center_freqs)  # 进行类似A计算操作
            fbank_feat = Cweight_gSpec.T
            # fbank_feat = np.log(fbank_feat)
            fbank_feat = feature_norm(fbank_feat)

            # fbank_feat = feature_norm(fbank_feat)
            # fbank_feat = delt_feature(fbank_feat)
            loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


def A_weight(data, freq):
    freq = np.array(freq)
    data = np.array(data)
    A1000 = 2.0
    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12200.0
    Aweighted = ((f4 ** 2) * (freq ** 4)) / ((freq ** 2 + f1 ** 2) * np.sqrt(freq ** 2 + f2 ** 2) *
                                             np.sqrt(freq ** 2 + f3 ** 2) * (freq ** 2 + f4 ** 2))
    Aweighted = 20 * (np.log10(Aweighted)) + A1000
    for i in range(data.shape[0]):
        data[i] += Aweighted
    return data


def C_weight(data, freq):
    freq = np.array(freq)
    data = np.array(data)
    C1000 = 0.062
    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12200.0
    Cweighted = ((f4 ** 2) * (freq ** 2)) / ((freq ** 2 + f1 ** 2) * (freq ** 2 + f4 ** 2))
    Cweighted = 20 * (np.log10(Cweighted)) + C1000
    for i in range(data.shape[0]):
        data[i] += Cweighted
    return data


# 对得到的特征进行归一化
def feature_norm(feat):
    normalized_feat = (feat - feat.min()) / (feat.max() - feat.min())
    # mean = np.mean(data)
    # std = np.std(data)
    # # 使用z-score方法进行归一化
    # normalized_data = (data - mean) / std
    return normalized_feat


if __name__ == '__main__':
    # 特征提取
    kfold_festure_in = "data_kfold_cut_zero"  # 切割好的数据，对于present个体，只复制murmur存在的.wav文件
    kfold_feature_folder = "feature_TF_GAF_cut_zero"  # 存储每折特征文件夹
    kfold_Aweight_feature_location_folder = "data_kfold_Aweight_feature_location"
    kfold_feature_location_folder = "data_kfold_Cweight_feature_location"
    tdf_feature_folder = r"E:\sdmurmur\EnvelopeandSE\data_kfold_cut_zero"  # 时域特征存储文件夹
    save_kfold_feature(kfold_festure_in, tdf_feature_folder, kfold_feature_folder, kfold=5)
    print('this is feature extraction file')

    # 特征图拼接测试, 可尝试分帧时补零，
    # a = np.random.rand(12, 64, 150)
    # b = np.random.rand(12, 5, 150)
    # c = np.concatenate((a, b), axis=1)
    # print(c.shape)
    # arr = np.array([[1, 2], [3, 4]])
    # a1 = np.random.rand(64, 239)
    # print(a1.shape)
    # a2 = np.random.rand(5, 150)
    # print(a2.shape)
    # # 使用常数值0填充数组，第一轴（行）前后各填充1个元素，第二轴（列）前后各填充2个元素
    # padded_arr = np.pad(a2, ((0, 0), (0, 89)), mode='constant', constant_values=0)
    # a3 = np.concatenate((a1, padded_arr), axis=0)
    # print(a3.shape)

    # csv文件转换为ndarray测试
    # TDF_folder = r"E:\sdmurmur\EnvelopeandSE\data_kfold_cut_zero\0_fold\train_data"
    # data = np.genfromtxt(TDF_folder+r'\2530_AV_Absent_0.csv', delimiter=',', dtype=None, encoding='utf-8')
    # # 检查读取的数据形状是否为5x150
    # if data.shape == (5, 150):
    #     print("数据已成功读取为5x150的ndarray")
    # else:
    #     print("数据形状不符合预期，请检查CSV文件内容")
    # padded_arr = np.pad(data, ((0, 0), (0, 89)), mode='constant', constant_values=0)
    # print(padded_arr)
