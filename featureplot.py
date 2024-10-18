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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from torchvision import transforms

if __name__ == '__main__':
    # MFCC Feature
    wavefile = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\9979_AV_Loud_0.wav"
    wave_data, fs = librosa.load(wavefile, sr=4000)
    print(fs)
    # 检查音频时长
    audio_duration = len(wave_data) / fs
    print(f"音频时长: {audio_duration} 秒")

    frame_length = int(0.025 * fs)  # 帧长
    hop_length = int(0.0125 * fs)  # 帧移
    gSpec = librosa.feature.melspectrogram(y=wave_data, sr=fs, n_fft=512, hop_length=hop_length,
                                           win_length=frame_length, n_mels=32, window='hamming', fmax=800, power=2.0)
    log_gSpec = librosa.power_to_db(gSpec, ref=np.max)
    print(log_gSpec.shape)
    librosa.display.specshow(log_gSpec, x_axis='time', y_axis='linear', sr=fs*10, fmax=800)
    log_gSpecT = log_gSpec[:, :-2]
    print(log_gSpecT.shape)
    # plt.ylim(0, 800)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    # tonnetz Feature
    # wavefile = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\9979_AV_Loud_0.wav"
    # wave_data, fs = librosa.load(wavefile, sr=4000)
    # chromagram = librosa.feature.chroma_stft(y=wave_data, sr=fs, hop_length=50, win_length=100)
    # chromagram = chromagram[:, 0:-2]
    # frame_length = int(0.025*fs)  # 帧长
    # hop_length = int(0.0125 * fs)  # 帧移
    # # 对音频数据进行分帧
    # frames = librosa.util.frame(wave_data, frame_length=frame_length, hop_length=hop_length)
    # # 计算每一帧的均值和方差
    # frame_means = np.mean(frames, axis=0)
    # frame_variances = np.var(frames, axis=0)
    # print(frame_means.shape)
    # print(frame_variances.shape)
    # # 将均值和方差转换成1x帧数的二维数组
    # frame_means_2d = frame_means.reshape(1, -1)
    # frame_variances_2d = frame_variances.reshape(1, -1)
    # print(frame_means_2d.shape)
    # print(frame_variances_2d.shape)
    # combined_feat = np.concatenate((chromagram, frame_means_2d, frame_variances_2d), axis=0)
    # print(combined_feat.shape)
    #
    # # 绘制均值和方差的图像
    # plt.figure(figsize=(12, 6))
    #
    # # 绘制均值图像
    # plt.subplot(2, 1, 1)
    # plt.plot(frame_means_2d.flatten(), label='Frame Means')
    # plt.title(wavefile[65:]+'Frame Means')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Mean Value')
    # plt.legend()
    #
    # # 绘制方差图像
    # plt.subplot(2, 1, 2)
    # plt.plot(frame_variances_2d.flatten(), label='Frame Variances', color='orange')
    # plt.title(wavefile[65:]+'Frame Variances')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Variance Value')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # y = librosa.effects.harmonic(y=wave_data)
    # chromagram = librosa.feature.chroma_stft(y=wave_data, sr=fs, hop_length=50, win_length=100)
    # chromagram = chromagram[:, 0:-2]
    # tonnetz = librosa.feature.tonnetz(y=y, sr=fs, hop_length=50, chroma=chromagram)
    # # tonnetz = tonnetz[:, 0:-2]
    # librosa.display.specshow(tonnetz, x_axis='s', sr=4000, hop_length=50, win_length=100)
    # print(tonnetz.shape)
    # plt.colorbar()
    # plt.xlabel('time')
    # plt.title(wavefile[27:])
    # plt.tight_layout()
    # plt.show()

    # # spectral contrast
    # wavefile = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\14241_MV_Soft_0.wav"
    # wave_data, fs = librosa.load(wavefile, sr=4000)
    # S = np.abs(librosa.stft(wave_data, hop_length=50, win_length=100))
    # contrast = librosa.feature.spectral_contrast(S=S, sr=fs, hop_length=50, win_length=100, fmin=20)
    # librosa.display.specshow(contrast, x_axis='s', sr=4000, hop_length=50, win_length=100)
    # print(contrast.shape)
    # plt.colorbar()
    # plt.xlabel('time')
    # plt.title(wavefile[27:])
    # plt.tight_layout()
    # plt.show()

    # # chroma 特征
    # wavefile = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\14241_MV_Soft_0.wav"
    # wave_data, fs = librosa.load(wavefile, sr=4000)
    # chromagram = librosa.feature.chroma_stft(y=wave_data, sr=fs, hop_length=50, win_length=100)
    # librosa.display.specshow(chromagram, x_axis='s', y_axis='chroma', sr=4000, hop_length=50, win_length=100)
    # print(chromagram.shape)
    # plt.colorbar()
    # plt.xlabel('time')
    # plt.title('Chroma Features')
    # plt.tight_layout()
    # plt.show()

    # # CQT特征
    # wavefile = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\9979_AV_Loud_0.wav"
    # wave_data, fs = librosa.load(wavefile, sr=4000)
    # CQT_feature = np.abs(librosa.cqt(wave_data, n_bins=60, fmin=25))
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(librosa.amplitude_to_db(CQT_feature, ref=np.max),
    #                                sr=fs, x_axis='time', y_axis='cqt_note', ax=ax)
    # ax.set_title('Constant-Q power spectrum')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    # plt.show()

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

    # 小波变换特征
    # cwt_file = r"E:\sdmurmur\murmurmatlab\E\sdmurmur\wavelets\2530_AV_Absent_0.csv"
    # # output_file = r"E:\sdmurmur\murmurmatlab\E\sdmurmur\wavelets\2530_AV_Absent_0.png"
    # csv_data = np.loadtxt(cwt_file, delimiter=',')
    # print(csv_data.shape)
    # expdata = np.power(10, csv_data)
    # # plt.savefig(output_file, dpi=300)  # 设置dpi为300，适合打印输出或高质量显示
    # plt.imshow(csv_data, aspect='auto')
    # plt.colorbar()
    # plt.show()

    # # RPF特征
    # wavefile = r"E:\sdmurmur\ssdHeartMurmur\data_kfold_cut_zero\0_fold\train_data\2530_AV_Absent_0.wav"
    # wave_data, fs = librosa.load(wavefile, sr=4000)

    # # 生成一个示例时间序列
    # time_series = np.sin(np.linspace(0, 20, 50)) + np.random.normal(0, 0.1, 50)
    #
    # 计算距离矩阵
    # dist_matrix = np.abs(wave_data[:, None] - wave_data[None, :])
    #
    # 设置阈值
    # epsilon = 0.1
    # recurrence_matrix = (dist_matrix <= epsilon).astype(int)
    # print(recurrence_matrix.shape)
    # print(type(recurrence_matrix))
    #
    # # 绘制Recurrence Plot
    # plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
    # plt.colorbar(label='Recurrence')
    # plt.title('Recurrence Plot')
    # plt.show()

    # cwt_file = r"E:\sdmurmur\murmurmatlab\E\sdmurmur\wavelets\2530_AV_Absent_0.csv"
    # csv_data = np.loadtxt(cwt_file, delimiter=',')
    # new_csv_data = np.transpose(csv_data)
    # print(new_csv_data.shape)
    # # 数据标准化
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(csv_data)
    #
    # # 应用 PCA 将特征维度从 12000 降到 239
    # n_components = 64
    # pca = PCA(n_components=n_components)
    # data_pca = np.transpose(pca.fit_transform(data_scaled))
    #
    # # 检查结果
    # print("降维后的数据形状:", data_pca.shape)
