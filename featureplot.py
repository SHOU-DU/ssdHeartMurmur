# 画特征图比较各个特征图
import numpy as np
from kfold_feature_extraction import *
import matplotlib.pyplot as plt
from matplotlib import image
import os
from spafe.utils.converters import erb2hz
# from pyts.image import RecurrencePlot
import pywt
from pyts.image import GramianAngularField, RecurrencePlot
import librosa
import torch


if __name__ == '__main__':
    # fig_path = 'feature_fig'
    # if not os.path.exists(fig_path):
    #     os.makedirs(fig_path)
    # kfold_festure_in = "data_kfold_out_grade_location"  # 分好折的3s数据
    # kfold_folder_vali = os.path.join(kfold_festure_in, "0_fold" + r"\vali_data")  # 五折交叉验证，最后通过测试集测试
    #
    # # 选择特征
    # # vali_feature = Log_GF(kfold_folder_vali)
    # # vali_feature = GF(kfold_folder_vali)
    # vali_feature = Cweight_Log_GF(kfold_folder_vali)
    # # vali_feature = log10_GF(kfold_folder_vali)
    # # np.savetxt(r'gfreqs.txt', gfreqs, newline='\n')
    # # 选择样本
    # sample_100 = vali_feature[99]  # 第100个片段的特征图
    # # sample_100 = vali_feature[100]  # 第101个片段的特征图
    # plt.imshow(sample_100)
    # plt.title('100_Cweight_Log_GF')
    # plt.savefig(fig_path + r'\100_Cweight_Log_GF.png')
    # plt.show()

    # wavefile = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\2530_AV_Absent_0.wav"
    # wavefile2 = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\14241_AV_Soft_0.wav"
    # wavefile3 = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\9979_AV_Loud_0.wav"
    #
    # wave_data, fs = librosa.load(wavefile, sr=4000)
    # wave_data = np.array([wave_data])
    # wave_data2, fs2 = librosa.load(wavefile2, sr=4000)
    # wave_data2 = np.array([wave_data2])
    # wave_data3, fs3 = librosa.load(wavefile3, sr=4000)
    # wave_data3 = np.array([wave_data3])
    # image_size = 239

    # GAF特征
    # gasf = GramianAngularField(image_size=image_size, method='summation')
    # wave_gasf = gasf.fit_transform(wave_data)
    # wave_gasf2 = gasf.fit_transform(wave_data2)
    # wave_gasf3 = gasf.fit_transform(wave_data3)
    # gadf = GramianAngularField(image_size=image_size, method='difference')
    # wave_gadf = gadf.fit_transform(wave_data)
    # wave_gadf2 = gadf.fit_transform(wave_data2)
    # wave_gadf3 = gadf.fit_transform(wave_data3)
    #
    # print(wave_gasf.shape)
    #
    # imges = [wave_gasf[0], wave_gadf[0]]
    # titles = ['Summation', 'Difference']
    # # 两种方法的可视化差异对比
    # # Comparison of two different methods
    # fig, axs = plt.subplots(1, 2, constrained_layout=True)
    # for img, title, ax in zip(imges, titles, axs):
    #     ax.imshow(img)
    #     ax.set_title(title)
    # fig.suptitle('GramianAngularField', y=0.94, fontsize=16)
    # plt.margins(0, 0)
    # # plt.savefig("./GramianAngularField.pdf", pad_inches=0)
    # # plt.show()
    #
    # imges2 = [wave_gasf2[0], wave_gadf2[0]]
    # titles = ['Summation', 'Difference']
    # # 两种方法的可视化差异对比
    # # Comparison of two different methods
    # fig, axs = plt.subplots(1, 2, constrained_layout=True)
    # for img, title, ax in zip(imges2, titles, axs):
    #     ax.imshow(img)
    #     ax.set_title(title)
    # fig.suptitle('GramianAngularField', y=0.94, fontsize=16)
    # plt.margins(0, 0)
    # # plt.savefig("./GramianAngularField.pdf", pad_inches=0)
    # # plt.show()
    #
    # imges3 = [wave_gasf3[0], wave_gadf3[0]]
    # titles = ['Summation', 'Difference']
    # # 两种方法的可视化差异对比
    # # Comparison of two different methods
    # fig, axs = plt.subplots(1, 2, constrained_layout=True)
    # for img, title, ax in zip(imges3, titles, axs):
    #     ax.imshow(img)
    #     ax.set_title(title)
    # fig.suptitle('GramianAngularField', y=0.94, fontsize=16)
    # plt.margins(0, 0)
    # # plt.savefig("./GramianAngularField.pdf", pad_inches=0)
    # plt.show()

    cwt_file = r"E:\sdmurmur\murmurmatlab\E\sdmurmur\wavelets\2530_AV_Absent_0.csv"
    csv_data = np.loadtxt(cwt_file, delimiter=',')
    print(csv_data.shape)

    plt.imshow(csv_data, aspect='auto')
    plt.colorbar()
    plt.show()

