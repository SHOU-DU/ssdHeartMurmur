import os
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
from spafe.features.gfcc import erb_spectrogram
from spafe.utils.vis import show_spectrogram
from spafe.utils.preprocessing import SlidingWindow
import time


'''
目标：输入原始数据集路径datafolder，折数num_fold，切割后的数据时长duration,将原始数据集分成对应折，存在相应文件夹下。
每折包含相近的absent,present个体，记录present的murmur locations(AV\PV\TV\MV)，并为对应.wav文件打标签，present:0,unknown:1,absent:2

结果：数据被分成了五折，每一折包含全部数据，分为了训练集和测试集，一次失败的分折尝试^……^
'''
def dataset_split_kfold(data_folder, kfold_folder, kfold=int):

    patient_files = find_patient_files(data_folder)  # 获取升序排列后.txt文件路径列表
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
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=2024)  # random_state=2024随机数种子，使得结果可复现
    for i, (train_idx, val_idx) in enumerate(kf.split(pIDs, plabel)):  # train_idx, val_idx为每一折中pIDs,plabel的下标列表
        print(f'this is {i}th fold')
        print('train_patient_num:', len(train_idx))
        print('val_patient_num:', len(val_idx))
        # 查看absent/present分布是否均匀
        temp_train_absent = 0
        temp_train_present = 0
        temp_train_loud = 0
        for index in train_idx:
            # print(id_label_zip)
            if plabel[index] == 2:
                temp_train_absent += 1
            elif plabel[index] == 1:
                temp_train_loud += 1
            else:
                temp_train_present += 1
        print(f"{i}th fold train_absent_id_num:{temp_train_absent}")
        print(f"{i}th fold train_present_id_num:{temp_train_present}")
        print(f"{i}th fold train_loud_id_num:{temp_train_loud}")
        temp_val_absent = 0
        temp_val_present = 0
        temp_val_loud = 0
        for index in val_idx:
            # print(id_label_zip)
            if plabel[index] == 2:
                temp_val_absent += 1
            elif plabel[index] == 1:
                temp_val_loud += 1
            else:
                temp_val_present += 1
        print(f"{i}th fold val_absent_id_num:{temp_val_absent}")
        print(f"{i}th fold val_present_id_num:{temp_val_present}")
        print(f"{i}th fold val_loud_id_num:{temp_val_loud}")
        kfold_out_dir = os.path.join(kfold_folder, str(i) + "_fold")
        if not os.path.exists(kfold_out_dir):
            os.makedirs(kfold_out_dir)
        # 第i折训练集
        if not os.path.exists(os.path.join(kfold_out_dir, "train_data")):
            os.makedirs(os.path.join(kfold_out_dir, "train_data"))
            for index in tqdm(train_idx):
                f = pIDs[index]  # 获取patientID
                my_cut_copy_files(
                    data_folder,
                    f,
                    os.path.join(kfold_out_dir, "train_data/"),
                )

        # 第i折测试集
        if not os.path.exists(os.path.join(kfold_out_dir, "vali_data")):
            os.makedirs(os.path.join(kfold_out_dir, "vali_data"))
            for index in tqdm(val_idx):
                f = pIDs[index]  # 获取patientID
                my_cut_copy_files(
                    data_folder,
                    f,
                    os.path.join(kfold_out_dir, "vali_data/"),
                )


def cut_copy_files(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)
            elif extension == '.wav':
                # 获取当前wav文件的ID 听诊区 等级
                with open(os.path.join(data_directory, patient_id+'.txt'), 'r') as txt_f:
                    txt_data = txt_f.read()
                    patient_ID = txt_data.split('\n')[0].split()[0]  # 获取病人ID
                    grade = get_grade(txt_data)
                    location = root.split('_')[1]
                recording, fs = librosa.load(os.path.join(data_directory, f), sr=4000)  # 分割（3s不重叠）
                num_cut = len(recording) / (3 * 4000)  # 每个记录的片段数量
                # time = len(recording)/fs
                if num_cut >= 2:
                    recording = recording[2*fs:len(recording)-fs]
                # recording = (recording- np.mean(recording))/ np.max(np.abs(recording)) #幅值归一化
                # recording = schmidt_spike_removal(recording) #去尖峰
                start = 0
                end = start+3*fs
                cut = list()
                num_cut = len(recording) / (3 * 4000)
                for num in range(int(num_cut)):  # 将每个片段写入对应的听诊区文件夹,int()小数部分被截断
                    small = recording[start:end]
                    cut.append(small)
                    soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' + str(num) + '.wav', cut[num], fs)
                    start += 3 * fs
                    end = start + 3 * fs


# 对于present个体，只复制murmur存在的.wav文件
def my_cut_copy_files(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)
            elif extension == '.wav':
                # 获取当前wav文件的ID 听诊区 等级
                with open(os.path.join(data_directory, patient_id+'.txt'), 'r') as txt_f:
                    txt_data = txt_f.read()
                    patient_ID = txt_data.split('\n')[0].split()[0]  # 获取病人ID
                    murmur = get_murmur(txt_data)
                    murmur_locations = (get_murmur_locations(txt_data)).split("+")  # 获取murmur存在的locations
                    grade = get_grade(txt_data)
                    location = root.split('_')[1]
                if murmur == 'Absent':  # Absent所有.wav文件均切片3s
                    recording, fs = librosa.load(os.path.join(data_directory, f), sr=4000)  # 分割（3s不重叠）
                    num_cut = len(recording) / (3 * 4000)  # 每个记录的片段数量
                    # time = len(recording)/fs
                    if num_cut >= 2:
                        recording = recording[2 * fs:len(recording) - fs]
                    # recording = (recording- np.mean(recording))/ np.max(np.abs(recording)) #幅值归一化
                    # recording = schmidt_spike_removal(recording) #去尖峰
                    start = 0
                    end = start + 3 * fs
                    cut = list()
                    num_cut = len(recording) / (3 * 4000)
                    for num in range(int(num_cut)):  # 将每个片段写入对应的听诊区文件夹,int()小数部分被截断
                        small = recording[start:end]
                        cut.append(small)
                        soundfile.write(
                            out_directory + '/' + patient_ID + '_' + str(location) + '_' + str(grade) + '_' + str(
                                num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs
                elif location in murmur_locations:
                    recording, fs = librosa.load(os.path.join(data_directory, f), sr=4000)  # 分割（3s不重叠）
                    num_cut = len(recording) / (3 * 4000)  # 每个记录的片段数量
                    # time = len(recording)/fs
                    if num_cut >= 2:
                        recording = recording[2*fs:len(recording)-fs]
                    # recording = (recording- np.mean(recording))/ np.max(np.abs(recording)) #幅值归一化
                    # recording = schmidt_spike_removal(recording) #去尖峰
                    start = 0
                    end = start+3*fs
                    cut = list()
                    num_cut = len(recording) / (3 * 4000)
                    for num in range(int(num_cut)):  # 将每个片段写入对应的听诊区文件夹,int()小数部分被截断
                        small = recording[start:end]
                        cut.append(small)
                        soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' + str(num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs


if __name__ == '__main__':
    # tqdm_ex()
    original_dataset_folder = r"E:\sdmurmur\thecircordataset\train_vali_data"
    kfold_out = "data_kfold_out_grade_location"  # 对于present个体，只复制murmur存在的.wav文件
    dataset_split_kfold(original_dataset_folder, kfold_out, kfold=5)
    # for i in tqdm(range(num_patient_files)):
    #     print(patient_files[i])
