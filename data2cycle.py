import os
import shutil
from helper_code import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from python_speech_features import logfbank
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# import wave
import librosa.display
import librosa
import soundfile
from spafe.features.gfcc import erb_spectrogram
from spafe.utils.vis import show_spectrogram
from spafe.utils.preprocessing import SlidingWindow
import time
import csv
import soundfile as sf


# 2025/06/07 此文件用于将心音按周期分割后提特征并加入相关周期特征

def get_split_cycles(dataset_folder, output_folder):
    patient_files = find_patient_files(dataset_folder)  # 获取升序排列后.txt文件路径列表
    num_patient_files = len(patient_files)
    # 检查是否读取到了原始数据
    if num_patient_files == 0:
        raise Exception('No data was provided.')
    # classes = ['Present', 'Unknown', 'Absent']  # 杂音类别
    grades = ['Soft', 'Loud', 'Absent']
    pAbsentID = []
    # pPresentID = []
    pUnknowID = []
    pSoftID = []
    pLoudID = []
    pIDs = []
    plabel = []  # 个体标签
    for i in range(num_patient_files):
        current_patient_data = load_patient_data(patient_files[i])  # 加载对应个体.txt文件
        # label = get_murmur(current_patient_data)  # 'Present', 'Unknown', 'Absent'
        pID = get_patient_id(current_patient_data)  # 个体ID
        grade = get_grade(current_patient_data)  # 'soft', 'loud', 'absent'

        if grade == grades[0]:
            pSoftID.append(pID)
            pIDs.append(pID)
            plabel.append(grades.index(grade))  # 按照ID读取顺序存储标签
        elif grade == grades[1]:
            pLoudID.append(pID)
            pIDs.append(pID)
            plabel.append(grades.index(grade))  # 按照ID读取顺序存储标签
        elif grade == grades[2]:
            pAbsentID.append(pID)
            pIDs.append(pID)
            plabel.append(grades.index(grade))  # 按照ID读取顺序存储标签
            # print('Absent ID is:', pID)
        else:
            pUnknowID.append(pID)
    print('Total patientID num(without Unknown):', len(pIDs))
    print('SoftID num:', len(pSoftID))
    print('LoudID num:', len(pLoudID))
    print('AbsentID num:', len(pAbsentID))
    print('UnknownID num:', len(pUnknowID))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for ID in tqdm(pIDs, desc='dataset split by cycles:'):
        split_audio_by_cycles(
            dataset_folder,
            ID,
            output_folder
        )


def split_audio_by_cycles(data_directory: str, patient_id: str, out_directory: str):
    sample_rate = 4000
    files = os.listdir(data_directory)
    cycles = []
    # max_cycle_duration = 0.0
    max_cycle_samples = 7500  # 最长心音周期点数为7009
    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)

            elif extension == '.tsv':
                # cycles = []
                file_path = os.path.join(data_directory, f)
                with open(file_path, mode='r', encoding='utf-8') as tsv_f:
                    lines = tsv_f.readlines()
                    # 寻找连续的1->2->3->4周期
                    for i in range(len(lines) - 3):
                        parts0 = lines[i].split()
                        parts1 = lines[i + 1].split()
                        parts2 = lines[i + 2].split()
                        parts3 = lines[i + 3].split()

                        if (len(parts0) == 3 and len(parts1) == 3 and
                                len(parts2) == 3 and len(parts3) == 3):  # 检查是否都有三列
                            try:
                                labels = [int(parts0[2]), int(parts1[2]), int(parts2[2]), int(parts3[2])]
                                if labels == [1, 2, 3, 4]:
                                    start_time = float(parts0[0])
                                    end_time = float(parts3[1])
                                    # 转换为样本数
                                    start_sample = int(start_time * sample_rate)
                                    end_sample = int(end_time * sample_rate)
                                    actual_duration = end_sample - start_sample
                                    cycles.append((start_sample, end_sample, actual_duration))
                            except ValueError:
                                continue

                # 计算最大周期时长（秒）
                # max_cycle_samples = max(cycle[2] for cycle in cycles)
            elif extension == '.wav':
                # 获取当前wav文件的ID 听诊区 等级
                with open(os.path.join(data_directory, patient_id+'.txt'), 'r') as txt_f:
                    txt_data = txt_f.read()
                    murmur = get_murmur(txt_data)
                    murmur_locations = (get_murmur_locations(txt_data)).split("+")  # 获取murmur存在的locations
                    patient_ID = txt_data.split('\n')[0].split()[0]  # 获取病人ID
                    grade = get_grade(txt_data)
                    location = root.split('_')[1]

                if murmur == 'Absent':  # Absent所有.wav文件
                    recording, fs = librosa.load(os.path.join(data_directory, f), sr=4000)  #
                    total_samples = len(recording)
                    # 处理每个周期
                    for idx, (start_sample, end_sample, actual_samples) in enumerate(cycles):

                        # 防止越界
                        start_sample = max(0, min(start_sample, total_samples - 1))
                        end_sample = max(start_sample + 1, min(end_sample, total_samples))
                        actual_samples = end_sample - start_sample

                        # 提取音频片段
                        segment = recording[start_sample:end_sample]

                        # 补零到统一长度
                        padding_length = max_cycle_samples - actual_samples
                        if padding_length < 0:
                            # 安全处理：这理论上不应该发生
                            padding_length = 0
                            print(f"警告: 周期 {idx + 1} 长度({actual_samples})超过最大长度{max_cycle_samples}")

                        padding = np.zeros(padding_length)
                        padded_segment = np.concatenate((segment, padding))

                        # 保存片段
                        output_path = os.path.join(out_directory, patient_ID + '_' +
                                                   str(location) + '_' + str(grade) + '_' + str(idx) + '.wav')
                        sf.write(output_path, padded_segment, fs)

                elif location in murmur_locations:  # 有杂音的Patient只获取有杂音的.wav文件
                    recording, fs = librosa.load(os.path.join(data_directory, f), sr=4000)  # 分割（3s不重叠）
                    total_samples = len(recording)
                    # 处理每个周期
                    for idx, (start_sample, end_sample, actual_samples) in enumerate(cycles):

                        # 防止越界
                        start_sample = max(0, min(start_sample, total_samples - 1))
                        end_sample = max(start_sample + 1, min(end_sample, total_samples))
                        actual_samples = end_sample - start_sample

                        # 提取音频片段
                        segment = recording[start_sample:end_sample]

                        # 补零到统一长度
                        padding_length = max_cycle_samples - actual_samples
                        if padding_length < 0:
                            # 安全处理：这理论上不应该发生
                            padding_length = 0
                            print(f"警告: 周期 {idx + 1} 长度({actual_samples})超过最大长度{max_cycle_samples}")

                        padding = np.zeros(padding_length)
                        padded_segment = np.concatenate((segment, padding))

                        # 保存片段
                        output_path = os.path.join(out_directory, patient_ID + '_' +
                                                   str(location) + '_' + str(grade) + '_' + str(idx) + '.wav')
                        sf.write(output_path, padded_segment, fs)

                cycles.clear()  # 清除cycles
                # return len(cycles)


def get_max_cycle_samples(data_folder):
    sample_rate = 4000
    max_samples = 0
    cycles = []
    files = os.listdir(data_folder)
    for f in tqdm(files):
        root, extension = os.path.splitext(f)
        if extension == '.tsv':
            file_path = os.path.join(data_folder, f)
            with open(file_path, mode='r', encoding='utf-8') as tsv_f:
                lines = tsv_f.readlines()
                # 寻找连续的1->2->3->4周期
                for i in range(len(lines) - 3):
                    parts0 = lines[i].split()
                    parts1 = lines[i + 1].split()
                    parts2 = lines[i + 2].split()
                    parts3 = lines[i + 3].split()

                    if (len(parts0) == 3 and len(parts1) == 3 and
                            len(parts2) == 3 and len(parts3) == 3):  # 检查是否都有三列
                        try:
                            labels = [int(parts0[2]), int(parts1[2]), int(parts2[2]), int(parts3[2])]
                            if labels == [1, 2, 3, 4]:
                                start_time = float(parts0[0])
                                end_time = float(parts3[1])
                                # 转换为样本数
                                start_sample = int(start_time * sample_rate)
                                end_sample = int(end_time * sample_rate)
                                actual_duration = end_sample - start_sample
                                cycles.append((start_sample, end_sample, actual_duration))
                        except ValueError:
                            continue
        if cycles:
            max_samples = max(cycle[2] for cycle in cycles)

    return max_samples


# 使用示例
if __name__ == "__main__":
    train_vali_data_fold = r"D:\sdmurmur\calibrateddataset2022\calibrated_train_vali_new"
    cali_train_vali_cycle_data = r"D:\sdmurmur\sdMurmurFiles\cali_train_vali_cycle_data"
    get_split_cycles(train_vali_data_fold, cali_train_vali_cycle_data)
    # split_audio_by_cycles(train_vali_data_fold, '36327', cali_train_vali_cycle_data)  # 测试函数用
    # print(f"成功切分并补零处理 {num_cycles} 个完整心音周期")
    # max_cycle_samples = get_max_cycle_samples(train_vali_data_fold)
    # print('max_cycle_samples: ', max_cycle_samples)
