import os
import math
import shutil
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


def save_kfold_feature(kfold_folder, feature_folder, kfold=int):
    for i in range(kfold):
        kfold_feature_out = os.path.join(feature_folder, str(i) + "_fold")  #
        kfold_folder_train = os.path.join(kfold_folder, str(i) + "_fold" + r"\train_data")
        kfold_folder_test = os.path.join(kfold_folder, str(i) + "_fold" + r"\vali_data")  # 五折交叉验证，最后通过测试集测试
        label_dir = os.path.join(kfold_feature_out, "label")
        feature_dir = os.path.join(kfold_feature_out, "feature")
        # 制作存储特征和标签的文件夹
        if not os.path.exists(kfold_feature_out):
            os.makedirs(kfold_feature_out)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        train_feature = Log_GF(kfold_folder_train)  # 提取第i折训练集特征
        # train_feature = Aweight_Log_GF(kfold_folder_train)
        # train_feature = Cweight_Log_GF(kfold_folder_train)
        train_label, train_location, train_id = get_label(kfold_folder_train)  # 获取各个3s片段label和听诊区位置和个体ID
        train_index = get_index(kfold_folder_train)

        test_feature = Log_GF(kfold_folder_test)  # 提取第i折验证集特征
        # test_feature = Aweight_Log_GF(kfold_folder_test)  # 采用类似A计权
        # test_feature = Cweight_Log_GF(kfold_folder_test)  # 采用类似A计权
        test_label, test_location, test_id = get_label(kfold_folder_test)
        test_index = get_index(kfold_folder_test)
        # 存储训练集特征和标签
        np.save(feature_dir + r'\train_loggamma.npy', train_feature)
        np.save(label_dir + r'\train_label.npy', train_label)
        np.save(label_dir + r'\train_location.npy', train_location)
        np.save(label_dir + r'\train_id.npy', train_id)
        np.save(label_dir + r'\train_index.npy', train_index)
        # 存储验证集特征标签等信息
        np.save(feature_dir + r'\vali_loggamma.npy', test_feature)
        np.save(label_dir + r'\vali_label.npy', test_label)
        np.save(label_dir + r'\vali_location.npy', test_location)
        np.save(label_dir + r'\vali_id.npy', test_id)
        np.save(label_dir + r'\vali_index.npy', test_index)
        print("train_feature shape:", train_feature.shape)  # train_feature shape: (样本数：14649, 滤波器数：64, 3s段数据帧数：239)
        print("train_label shape:", train_label.shape)
        print("vali_feature shape:", test_feature.shape)  # train_feature shape: (样本数：14649, 滤波器数：64, 3s段数据帧数：239)
        print("vali_label shape:", test_label.shape)
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


def RPF(data_directory):
    loggamma = []
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Cweight_Log_GF feature:'):  # 加tqdm可视化特征提取过程
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)  # 将音信信号置为0均值
            fbank_feat = feature_norm(fbank_feat)
            loggamma.append(fbank_feat)
        else:
            continue
    return np.array(loggamma)


if __name__ == '__main__':
    kfold_festure_in = "data_kfold_double_s1s2"  # 切割好的数据，对于present个体，只复制murmur存在的.wav文件
    kfold_feature_folder = "feature_double_s1s2_calibrated"  # 存储每折特征文件夹
    kfold_Aweight_feature_location_folder = "data_kfold_Aweight_feature_location"
    kfold_feature_location_folder = "data_kfold_Cweight_feature_location"
    save_kfold_feature(kfold_festure_in, kfold_feature_folder, kfold=5)
    print('this is feature extraction file')
