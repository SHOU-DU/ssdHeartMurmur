# 画特征图比较各个特征图
import numpy as np
from kfold_feature_extraction import *
import matplotlib.pyplot as plt
import os
from spafe.utils.converters import erb2hz
from pyts.image import RecurrencePlot


def extract_recurrence_plot_features(audio_file):
    # 加载音频文件
    audio, fs = librosa.load(audio_file, sr=None)
    audio = np.array([audio])
    audio = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))
    # 计算递归图
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.transform(audio)[0]

    # 计算递归图特征：确定率（DET）、平均对角线长度（L）和熵（ENT）
    # det = rp.recurrence_rate_[0]
    # l = rp.mean_diagonal_[0]
    # ent = rp.recurrence_entropy_[0]

    return X_rp


def myRPF(audio_file):
    audio, fs = librosa.load(audio_file, sr=None)
    audio = np.array([audio])
    audio = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))
    N = len(audio)
    S = np.column_stack((audio[:-1], audio[1:]))

    R = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        for j in range(N - 1):
            R[i, j] = np.sum((S[i, :] - S[j, :]) ** 2)
    R = (R - np.min(np.min(R))) / (np.max(np.max(R)) - np.min(np.min(R))) * 4
    plt.figure()
    plt.imshow(R)
    plt.title('imaging time series of RP')



if __name__ == '__main__':
    fig_path = 'feature_fig'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
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
    # sd RPF
    audio_file = r'D:\shoudu\pycharmssdHeartMurmur\data_kfold_out_grade_location\0_fold\train_data\14241_AV_Soft_0.wav'
    X_rp = extract_recurrence_plot_features(audio_file)
    # X_rp = np.log(abs(X_rp) + 1)
    X_rp = np.resize(X_rp, [256, 256])
    print(X_rp.shape)
    # 可视化递归图
    plt.imshow(X_rp, cmap='binary', origin='lower')
    plt.title('RP_point_14241')
    # plt.savefig(fig_path + r'\RP_None_14241.png')
    plt.show()
    # myRPF(audio_file)

