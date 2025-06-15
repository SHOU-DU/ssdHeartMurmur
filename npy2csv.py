import numpy as np
from spafe.features.gfcc import erb_spectrogram
import librosa
from kfold_feature_extraction import *

if __name__ == '__main__':
    # feature_path = (r"E:\sdmurmur\ssdHeartMurmurFiles\calibrated_train_vali_new_feature"
    #                 r"\TF_TDF_MV_CST_feature\0_fold\feature\train_loggamma.npy")
    # id_path = (r'E:\sdmurmur\ssdHeartMurmurFiles\calibrated_train_vali_new_feature\TF_TDF_MV_CST_feature'
    #            r'\0_fold\label\train_id.npy')
    # label_path = (r'E:\sdmurmur\ssdHeartMurmurFiles\calibrated_train_vali_new_feature\TF_TDF_MV_CST_feature'
    #               r'\0_fold\label\train_label.npy')
    # location_path = (r'E:\sdmurmur\ssdHeartMurmurFiles\calibrated_train_vali_new_feature\TF_TDF_MV_CST_feature'
    #                  r'\0_fold\label\train_location.npy')
    # # 读取 .npy 文件
    # feature = np.load(feature_path)
    # id = np.load(id_path)
    # label = np.load(label_path)
    # location = np.load(location_path)
    # counter = 0
    # index_2530 = 0
    # for item in id:
    #     counter = counter + 1
    #     if item == '2530':
    #         index_2530 = counter
    #         break
    #
    # index_2530_TV_0 = counter + 9  # 2530_TV: 44,45,46
    # print(index_2530_TV_0)
    # # for i in range(3):
    # #     print(location[index_2530_TV_0])
    # #     print(id[index_2530_TV_0])
    # #     index_2530_TV_0 = index_2530_TV_0 + 1
    #
    # # 保存为 .csv 文件
    # feature_csv_path = r'E:\sdmurmur\ssdHeartMurmurFiles\murmurMatlabPlot'
    # np.savetxt(feature_csv_path+r'\2530_TV_0.csv', feature[index_2530_TV_0], delimiter=',')
    # print(feature[index_2530_TV_0].shape)

    wavefile = r"E:\sdmurmur\ssdHeartMurmurFiles\S1S2Experiment\vali_scale\vali_mask_s1\0_fold\train_data\9979_MV_Loud_0.wav"
    wave_data, fs = librosa.load(wavefile, sr=4000)
    nfilts = 8
    nfft = 512
    low_freq = 0
    high_freq = 2000
    # compute erb spectrogram
    gSpec, gfreqs = erb_spectrogram(wave_data,
                                    fs=fs,
                                    pre_emph=0,
                                    pre_emph_coeff=0.97,
                                    window=SlidingWindow(0.025, 0.0125, "hamming"),
                                    nfilts=64,
                                    nfft=512,
                                    low_freq=25,
                                    high_freq=fs / 2)
    myspectram = gSpec.T
    # 保存为 .csv 文件
    feature_csv_path = r'E:\sdmurmur\ssdHeartMurmurFiles\murmurMatlabPlot'
    np.savetxt(feature_csv_path+r'\gamma_tone_9979_MV_Loud_0.csv', myspectram, delimiter=',')
    print(myspectram.shape)
    # visualize spectrogram
    show_spectrogram(gSpec.T,
                     fs=fs,
                     xmin=0,
                     xmax=len(wave_data) / fs,
                     ymin=0,
                     ymax=(fs / 2) / 1000,
                     dbf=80.0,
                     xlabel="Time (s)",
                     ylabel="Frequency (kHz)",
                     title="Erb spectrogram (dB)",
                     cmap="jet")
