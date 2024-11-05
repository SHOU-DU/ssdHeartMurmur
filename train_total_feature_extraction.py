# import os
# import math
# import shutil
#
# import torch
# import torch.nn as nn
# from helper_code import *
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from python_speech_features import logfbank
# from sklearn.model_selection import StratifiedKFold
# from tqdm import tqdm
# import wave
import librosa.display
# import librosa
# import soundfile
# from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
# from spafe.utils.converters import erb2hz
# from spafe.utils.vis import show_spectrogram
# from spafe.utils.preprocessing import SlidingWindow
from dataset2kfold import *
# from pyts.image import GramianAngularField
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler


def save_test_feature(train_folder, train_tdf_folder, train_feature_folder):
    train_feature_out = train_feature_folder
    label_dir = os.path.join(train_feature_out, "label")
    feature_dir = os.path.join(train_feature_out, "feature")
    # 制作存储特征和标签的文件夹
    if not os.path.exists(train_feature_out):
        os.makedirs(train_feature_out)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    # train_feature = Log_GF_GAF(kfold_folder_train)
    # train_feature = Log_GF_CWT_PCA(kfold_folder_train, test_tdf_folder)
    train_feature = Log_GF_TDF_MV_CST(train_folder, train_tdf_folder)
    # train_feature = Log_mel_32(train_folder)

    train_label, train_location, train_id = get_label(train_folder)  # 获取各个3s片段label和听诊区位置和个体ID
    train_index = get_index(train_folder)

    # 存储总训练集特征和标签
    np.save(feature_dir + r'\train_loggamma.npy', train_feature)
    np.save(label_dir + r'\train_label.npy', train_label)
    np.save(label_dir + r'\train_location.npy', train_location)
    np.save(label_dir + r'\train_id.npy', train_id)
    np.save(label_dir + r'\train_index.npy', train_index)
    print("train_feature shape:", train_feature.shape)  # train_feature shape: (样本数：14649, 滤波器数：64, 3s段数据帧数：239)
    print("train_label shape:", train_label.shape)
    print(f"测试集特征提取完毕")


def Log_GF_TDF_MV_CST(data_directory, TDF_directory):  # 提取时频域和时域特征
    loggamma = list()
    # 加tqdm可视化特征提取过程
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Log_GF, TDF, MV, CST feature 60Hz:'):
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)
            x = x / np.max(np.abs(x))  # 归一化为[-1, 1]
            # 对音频数据进行分帧
            frame_length = int(0.025 * fs)  # 帧长
            hop_length = int(0.0125 * fs)  # 帧移
            frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
            # 计算每一帧的均值和方差
            frame_means = np.mean(frames, axis=0)
            frame_variances = np.var(frames, axis=0)
            # 将均值和方差转换成1x帧数的二维数组
            frame_means_2d = frame_means.reshape(1, -1)
            frame_variances_2d = frame_variances.reshape(1, -1)

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
            # chroma Feature
            chromagram = librosa.feature.chroma_stft(y=x, sr=fs, hop_length=50, win_length=100)
            chromagram = chromagram[:, 0:-2]  # 取前239帧
            # spectral contrast Feature
            spectral = np.abs(librosa.stft(x, hop_length=50, win_length=100))
            contrast = librosa.feature.spectral_contrast(S=spectral, sr=fs, hop_length=50, win_length=100, fmin=20)
            contrast = contrast[:, 0:-2]
            # tonnetz Feature
            y = librosa.effects.harmonic(y=x)  # 提取谐波分量
            tonnetz = librosa.feature.tonnetz(y=y, sr=fs, hop_length=50, chroma=chromagram)

            # 读取对应的.csv文件
            csv_file = os.path.join(TDF_directory, root + '.csv')
            if os.path.exists(csv_file):
                # csv_data = np.genfromtxt(csv_file, delimiter=',', dtype=None, encoding='utf-8')
                csv_data = np.loadtxt(csv_file, delimiter=',')
                # num_cols_to_add = fbank_feat.shape[1] - csv_data.shape[1]
                # csv_data_pad = np.pad(csv_data, ((0, 0), (0, num_cols_to_add)), mode='constant', constant_values=0)
                csv_data_cut = csv_data[:, :-1]
                # 拼接.wav文件特征和.csv文件数据
                combined_feat = np.concatenate((fbank_feat, csv_data_cut, frame_means_2d, frame_variances_2d,
                                                chromagram, contrast, tonnetz), axis=0)
                loggamma.append(combined_feat)

            else:
                print(f"CSV file not found for {f}")

            # loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


def Log_mel_32(data_directory):
    loggamma = list()
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Log_GF feature:'):  # 加tqdm可视化特征提取过程
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)
            x = x / np.max(np.abs(x))
            frame_length = int(0.025 * fs)  # 帧长
            hop_length = int(0.0125 * fs)  # 帧移
            # gfreqs为经过gammatone滤波器后得到的傅里叶变换矩阵
            gSpec = librosa.feature.melspectrogram(y=x, sr=fs, n_fft=512, hop_length=hop_length, win_length=frame_length,
                                                   n_mels=32, window='hamming', fmax=800, power=2.0)
            fbank_feat = gSpec[:, :-2]
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
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + ' Log_GF and TDF feature 60Hz:'):  # 加tqdm可视化特征提取过程
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
                # num_cols_to_add = fbank_feat.shape[1] - csv_data.shape[1]
                # csv_data_pad = np.pad(csv_data, ((0, 0), (0, num_cols_to_add)), mode='constant', constant_values=0)
                csv_data_cut = csv_data[:, :-1]
                # 拼接.wav文件特征和.csv文件数据
                combined_feat = np.concatenate((fbank_feat, csv_data_cut), axis=0)
                loggamma.append(combined_feat)

            else:
                print(f"CSV file not found for {f}")

            # loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


def Log_GF_TDF_CST_MV_MFCC(data_directory, TDF_directory):  # 提取时频域和时域特征
    loggamma = list()
    # 加tqdm可视化特征提取过程
    for f in tqdm(sorted(os.listdir(data_directory)), desc=str(data_directory) + 'train set Log_GF_MFCCTDFCST_MV feat 60Hz:'):
        root, extension = os.path.splitext(f)
        if extension == '.wav':
            x, fs = librosa.load(os.path.join(data_directory, f), sr=4000)
            x = x - np.mean(x)
            x = x / np.max(np.abs(x))  # 归一化为[-1, 1]
            # 对音频数据进行分帧
            frame_length = int(0.025 * fs)  # 帧长
            hop_length = int(0.0125 * fs)  # 帧移
            frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
            # 计算每一帧的均值和方差
            frame_means = np.mean(frames, axis=0)
            frame_variances = np.var(frames, axis=0)
            # 将均值和方差转换成1x帧数的二维数组
            frame_means_2d = frame_means.reshape(1, -1)
            frame_variances_2d = frame_variances.reshape(1, -1)

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
            # chroma Feature
            chromagram = librosa.feature.chroma_stft(y=x, sr=fs, hop_length=50, win_length=100)
            chromagram = chromagram[:, 0:-2]  # 取前239帧
            # spectral contrast Feature
            spectral = np.abs(librosa.stft(x, hop_length=50, win_length=100))
            contrast = librosa.feature.spectral_contrast(S=spectral, sr=fs, hop_length=50, win_length=100, fmin=20)
            contrast = contrast[:, 0:-2]
            # tonnetz Feature
            y = librosa.effects.harmonic(y=x)  # 提取谐波分量
            tonnetz = librosa.feature.tonnetz(y=y, sr=fs, hop_length=50, chroma=chromagram)
            # MFCC特征
            mfcc_f = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=64, hop_length=50, win_length=100)
            mfcc_f = mfcc_f[:, 0:-2]  # 去掉最后两帧使得各个特征图形状保持一致
            # 读取对应的.csv文件，提取时域包络特征
            csv_file = os.path.join(TDF_directory, root + '.csv')
            if os.path.exists(csv_file):
                # csv_data = np.genfromtxt(csv_file, delimiter=',', dtype=None, encoding='utf-8')
                csv_data = np.loadtxt(csv_file, delimiter=',')
                # num_cols_to_add = fbank_feat.shape[1] - csv_data.shape[1]
                # csv_data_pad = np.pad(csv_data, ((0, 0), (0, num_cols_to_add)), mode='constant', constant_values=0)
                csv_data_cut = csv_data[:, :-1]
                # 拼接.wav文件特征和.csv文件数据
                combined_feat = np.concatenate((fbank_feat, mfcc_f, csv_data_cut, frame_means_2d, frame_variances_2d,
                                                chromagram, contrast, tonnetz), axis=0)
                loggamma.append(combined_feat)

            else:
                print(f"CSV file not found for {f}")

            # loggamma.append(fbank_feat)

        else:
            continue
    return np.array(loggamma)


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
    kfold_festure_in = r"E:\sdmurmur\ssdHeartMurmurFiles\S1S2Experiment\train_vali_mixed_scale\train_vali_mask_s2"  # test set切割好的数据，对于present个体，只复制murmur存在的.wav文件
    kfold_feature_folder = (r"E:\sdmurmur\ssdHeartMurmurFiles\S1S2Experiment\train_vali_mixed_scale"
                            r"\train_vali_mask_s2_feature")  # 特征输出文件夹
    tdf_feature_folder = (r"E:\sdmurmur\ssdHeartMurmurFiles\S1S2Experiment\train_vali_mixed_scale"
                          r"\train_vali_mask_s2_EnvelopeandSE60Hz")  # 时域特征存储文件夹
    save_test_feature(kfold_festure_in, tdf_feature_folder, kfold_feature_folder)
    print('this is feature extraction file')
