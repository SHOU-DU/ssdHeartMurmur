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
import csv

'''
目标：输入原始数据集路径datafolder，折数num_fold，切割后的数据时长duration,将原始数据集分成对应折，存在相应文件夹下。
每折包含相近的absent,present个体，记录present的murmur locations(AV\PV\TV\MV)，并为对应.wav文件打标签，present:0,unknown:1,absent:2

结果：数据被分成了五折，每一折包含全部数据，分为了训练集和测试集
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
            for index in tqdm(train_idx, desc='calibrated train set cut zero:'):
                f = pIDs[index]  # 获取patientID
                cut_copy_files_zero(
                    data_folder,
                    f,
                    os.path.join(kfold_out_dir, "train_data/"),
                )

        # 第i折测试集
        if not os.path.exists(os.path.join(kfold_out_dir, "vali_data")):
            os.makedirs(os.path.join(kfold_out_dir, "vali_data"))
            for index in tqdm(val_idx, desc='calibrated vali set cut zero:'):
                f = pIDs[index]  # 获取patientID
                cut_copy_files_zero(
                    data_folder,
                    f,
                    os.path.join(kfold_out_dir, "vali_data/"),
                )


# 为测试集做s1,s2幅值伸缩
def test_dataset_scale(test_data_folder, scaled_test_folder):
    patient_files = find_patient_files(test_data_folder)  # 获取升序排列后.txt文件路径列表
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

    if not os.path.exists(scaled_test_folder):
        os.makedirs(scaled_test_folder)

    for ID in tqdm(pIDs, desc='test set double s2:'):
        # print(ID)  打印ID检查
        cut_copy_files_double_s2(
            test_data_folder,
            ID,
            scaled_test_folder,
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


# 切掉为0的部分和s1, s2部分
def cut_copy_files_s1_s2(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    s1_start = []
    s1_end = []
    s2_start = []
    s2_end = []
    zero_start = []
    zero_end = []
    total_start = []
    total_end = []
    new_recording = []

    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)

            elif extension == '.tsv':  # 获取S1，S2位置
                file_path = os.path.join(data_directory, f)
                with open(file_path, mode='r', encoding='utf-8') as tsv_file:
                    tsv_reader = csv.reader(tsv_file, delimiter='\t')
                    for row in tsv_reader:
                        if row[2] == '1':
                            s1_start.append(float(row[0]))  # 为string类型，需要类型转换
                            s1_end.append(float(row[1]))
                        elif row[2] == '3':
                            s2_start.append(float(row[0]))
                            s2_end.append(float(row[1]))
                        elif row[2] == '0':
                            zero_start.append(float(row[0]))
                            zero_end.append(float(row[1]))
                total_start = s1_start + s2_start  # 存储起始位置
                total_end = s1_end + s2_end  # 存储结束位置
                zero_start = zero_start[1:]  # 移除第一个未标注起始点
                zero_end = zero_end[:-1]  # 移除最后一个未标注终点

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
                    for start_pos, end_pos in zip(total_start, total_end):
                        start_pos_int = int(start_pos*fs)
                        end_pos_int = int(end_pos*fs)
                        recording[start_pos_int: end_pos_int] = 0.0
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s*fs)
                        zero_e_int = int(zero_e*fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording

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
                    for start_pos, end_pos in zip(total_start, total_end):
                        recording[int(start_pos*fs): int(end_pos*fs)] = 0.0
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s * fs)
                        zero_e_int = int(zero_e * fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording
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
                        soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' +
                                        str(num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs
                s1_start.clear()  # 清空先前存储的S1,S2起止点
                s1_end.clear()
                s2_start.clear()
                s2_end.clear()
                total_start.clear()
                total_end.clear()
                zero_start.clear()
                zero_end.clear()
                new_recording.clear()


# 切掉为0的部分和s1部分
def cut_copy_files_s1(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    s1_start = []
    s1_end = []
    zero_start = []
    zero_end = []
    new_recording = []

    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)

            elif extension == '.tsv':  # 获取S1，S2位置
                file_path = os.path.join(data_directory, f)
                with open(file_path, mode='r', encoding='utf-8') as tsv_file:
                    tsv_reader = csv.reader(tsv_file, delimiter='\t')
                    for row in tsv_reader:
                        if row[2] == '1':
                            s1_start.append(float(row[0]))  # 为string类型，需要类型转换
                            s1_end.append(float(row[1]))
                        elif row[2] == '0':
                            zero_start.append(float(row[0]))
                            zero_end.append(float(row[1]))
                zero_start = zero_start[1:]  # 移除第一个未标注起始点
                zero_end = zero_end[:-1]  # 移除最后一个未标注终点

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
                    for start_pos, end_pos in zip(s1_start, s1_end):
                        start_pos_int = int(start_pos*fs)
                        end_pos_int = int(end_pos*fs)
                        recording[start_pos_int: end_pos_int] = 0.0
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s*fs)
                        zero_e_int = int(zero_e*fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording

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
                    for start_pos, end_pos in zip(s1_start, s1_end):
                        recording[int(start_pos*fs): int(end_pos*fs)] = 0.0
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s * fs)
                        zero_e_int = int(zero_e * fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording
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
                        soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' +
                                        str(num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs
                s1_start.clear()  # 清空先前存储的S1,S2起止点
                s1_end.clear()
                zero_start.clear()
                zero_end.clear()
                new_recording.clear()


# 切掉为0的部分和s2部分
def cut_copy_files_s2(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    s2_start = []
    s2_end = []
    zero_start = []
    zero_end = []
    new_recording = []

    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)

            elif extension == '.tsv':  # 获取S1，S2位置
                file_path = os.path.join(data_directory, f)
                with open(file_path, mode='r', encoding='utf-8') as tsv_file:
                    tsv_reader = csv.reader(tsv_file, delimiter='\t')
                    for row in tsv_reader:
                        if row[2] == '3':
                            s2_start.append(float(row[0]))  # 为string类型，需要类型转换
                            s2_end.append(float(row[1]))
                        elif row[2] == '0':
                            zero_start.append(float(row[0]))
                            zero_end.append(float(row[1]))
                zero_start = zero_start[1:]  # 移除第一个未标注起始点
                zero_end = zero_end[:-1]  # 移除最后一个未标注终点

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
                    for start_pos, end_pos in zip(s2_start, s2_end):
                        start_pos_int = int(start_pos*fs)
                        end_pos_int = int(end_pos*fs)
                        recording[start_pos_int: end_pos_int] = 0.0
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s*fs)
                        zero_e_int = int(zero_e*fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording

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
                    for start_pos, end_pos in zip(s2_start, s2_end):
                        recording[int(start_pos*fs): int(end_pos*fs)] = 0.0
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s * fs)
                        zero_e_int = int(zero_e * fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording
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
                        soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' +
                                        str(num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs
                s2_start.clear()  # 清空先前存储的S1,S2起止点
                s2_end.clear()
                zero_start.clear()
                zero_end.clear()
                new_recording.clear()


# 切掉为0的部分
def cut_copy_files_zero(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    zero_start = []
    zero_end = []
    new_recording = []

    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)

            elif extension == '.tsv':  # 获取S1，S2位置
                file_path = os.path.join(data_directory, f)
                with open(file_path, mode='r', encoding='utf-8') as tsv_file:
                    tsv_reader = csv.reader(tsv_file, delimiter='\t')
                    for row in tsv_reader:
                        if row[2] == '0':
                            zero_start.append(float(row[0]))
                            zero_end.append(float(row[1]))
                zero_start = zero_start[1:]  # 移除第一个未标注起始点
                zero_end = zero_end[:-1]  # 移除最后一个未标注终点

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

                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s*fs)
                        zero_e_int = int(zero_e*fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording

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

                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s * fs)
                        zero_e_int = int(zero_e * fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording
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
                        soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' +
                                        str(num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs

                zero_start.clear()
                zero_end.clear()
                new_recording.clear()


# 切掉为0的部分,将s2幅值加倍
def cut_copy_files_double_s2(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    s2_start = []
    s2_end = []
    zero_start = []
    zero_end = []
    new_recording = []

    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)

            elif extension == '.tsv':  # 获取S1，S2位置
                file_path = os.path.join(data_directory, f)
                with open(file_path, mode='r', encoding='utf-8') as tsv_file:
                    tsv_reader = csv.reader(tsv_file, delimiter='\t')
                    for row in tsv_reader:
                        if row[2] == '3':
                            s2_start.append(float(row[0]))  # 为string类型，需要类型转换
                            s2_end.append(float(row[1]))
                        elif row[2] == '0':
                            zero_start.append(float(row[0]))
                            zero_end.append(float(row[1]))
                zero_start = zero_start[1:]  # 移除第一个未标注起始点
                zero_end = zero_end[:-1]  # 移除最后一个未标注终点

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
                    for start_pos, end_pos in zip(s2_start, s2_end):
                        start_pos_int = int(start_pos*fs)
                        end_pos_int = int(end_pos*fs)
                        recording[start_pos_int: end_pos_int] = 2 * recording[start_pos_int: end_pos_int]
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s*fs)
                        zero_e_int = int(zero_e*fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording

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
                    for start_pos, end_pos in zip(s2_start, s2_end):
                        start_pos_int = int(start_pos * fs)
                        end_pos_int = int(end_pos * fs)
                        recording[start_pos_int: end_pos_int] = 2 * recording[start_pos_int: end_pos_int]
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s * fs)
                        zero_e_int = int(zero_e * fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording
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
                        soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' +
                                        str(num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs
                s2_start.clear()  # 清空先前存储的S1,S2起止点
                s2_end.clear()
                zero_start.clear()
                zero_end.clear()
                new_recording.clear()


# 切掉为0的部分,将s1幅值加倍
def cut_copy_files_double_s1(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    s1_start = []
    s1_end = []
    zero_start = []
    zero_end = []
    new_recording = []

    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)

            elif extension == '.tsv':  # 获取S1，S2位置
                file_path = os.path.join(data_directory, f)
                with open(file_path, mode='r', encoding='utf-8') as tsv_file:
                    tsv_reader = csv.reader(tsv_file, delimiter='\t')
                    for row in tsv_reader:
                        if row[2] == '1':
                            s1_start.append(float(row[0]))  # 为string类型，需要类型转换
                            s1_end.append(float(row[1]))
                        elif row[2] == '0':
                            zero_start.append(float(row[0]))
                            zero_end.append(float(row[1]))
                zero_start = zero_start[1:]  # 移除第一个未标注起始点
                zero_end = zero_end[:-1]  # 移除最后一个未标注终点

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
                    for start_pos, end_pos in zip(s1_start, s1_end):
                        start_pos_int = int(start_pos*fs)
                        end_pos_int = int(end_pos*fs)
                        recording[start_pos_int: end_pos_int] = 2 * recording[start_pos_int: end_pos_int]
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s*fs)
                        zero_e_int = int(zero_e*fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording

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
                    for start_pos, end_pos in zip(s1_start, s1_end):
                        start_pos_int = int(start_pos * fs)
                        end_pos_int = int(end_pos * fs)
                        recording[start_pos_int: end_pos_int] = 2 * recording[start_pos_int: end_pos_int]
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s * fs)
                        zero_e_int = int(zero_e * fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording
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
                        soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' +
                                        str(num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs
                s1_start.clear()  # 清空先前存储的S1,S2起止点
                s1_end.clear()
                zero_start.clear()
                zero_end.clear()
                new_recording.clear()


# 切掉为0的部分,将s1, s2部分幅值加倍
def cut_copy_files_double_s1s2(data_directory: str, patient_id: str, out_directory: str) -> None:
    files = os.listdir(data_directory)
    s1_start = []
    s1_end = []
    s2_start = []
    s2_end = []
    zero_start = []
    zero_end = []
    total_start = []
    total_end = []
    new_recording = []

    for f in files:
        root, extension = os.path.splitext(f)
        if f.startswith(patient_id):
            if extension == '.txt':
                _ = shutil.copy(os.path.join(data_directory, f), out_directory)

            elif extension == '.tsv':  # 获取S1，S2位置
                file_path = os.path.join(data_directory, f)
                with open(file_path, mode='r', encoding='utf-8') as tsv_file:
                    tsv_reader = csv.reader(tsv_file, delimiter='\t')
                    for row in tsv_reader:
                        if row[2] == '1':
                            s1_start.append(float(row[0]))  # 为string类型，需要类型转换
                            s1_end.append(float(row[1]))
                        elif row[2] == '3':
                            s2_start.append(float(row[0]))
                            s2_end.append(float(row[1]))
                        elif row[2] == '0':
                            zero_start.append(float(row[0]))
                            zero_end.append(float(row[1]))
                total_start = s1_start + s2_start  # 存储起始位置
                total_end = s1_end + s2_end  # 存储结束位置
                zero_start = zero_start[1:]  # 移除第一个未标注起始点
                zero_end = zero_end[:-1]  # 移除最后一个未标注终点

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
                    for start_pos, end_pos in zip(total_start, total_end):
                        start_pos_int = int(start_pos*fs)
                        end_pos_int = int(end_pos*fs)
                        recording[start_pos_int: end_pos_int] = 2 * recording[start_pos_int: end_pos_int]
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s*fs)
                        zero_e_int = int(zero_e*fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording

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
                    for start_pos, end_pos in zip(total_start, total_end):
                        start_pos_int = int(start_pos * fs)
                        end_pos_int = int(end_pos * fs)
                        recording[start_pos_int: end_pos_int] = 2 * recording[start_pos_int: end_pos_int]
                    for zero_s, zero_e in zip(zero_start, zero_end):
                        zero_s_int = int(zero_s * fs)
                        zero_e_int = int(zero_e * fs)
                        new_recording.extend(recording[zero_e_int:zero_s_int])  # 拼接标注非0的recording
                    recording = new_recording
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
                        soundfile.write(out_directory + '/' + patient_ID + '_'+str(location)+'_' + str(grade) + '_' +
                                        str(num) + '.wav', cut[num], fs)
                        start += 3 * fs
                        end = start + 3 * fs
                s1_start.clear()  # 清空先前存储的S1,S2起止点
                s1_end.clear()
                s2_start.clear()
                s2_end.clear()
                total_start.clear()
                total_end.clear()
                zero_start.clear()
                zero_end.clear()
                new_recording.clear()


# 检查tsv文件标注是否有问题
def check_tsv(data_directory: str):
    files = os.listdir(data_directory)
    wrong_list = []
    for f in files:
        root, extension = os.path.splitext(f)
        if extension == '.tsv':
            file_path = os.path.join(data_directory, f)
            with open(file_path, mode='r', encoding='utf-8') as tsv_file:
                tsv_reader = csv.reader(tsv_file, delimiter='\t')
                last_row = '0'  # 当前行的上一行
                # next_row = '0'  # 当前行的下一行
                # 逐行读取文件内容
                for row in tsv_reader:
                    # float_row = [float(x) for x in row]
                    # print(float_row)
                    if row[2] == '0':
                        last_row = '0'  # 中间可能出现标注为0的情况
                    elif row[2] == '1' and (last_row == '4' or last_row == '0'):
                        last_row = row[2]
                    elif row[2] == '2' and (last_row == '1' or last_row == '0'):
                        last_row = row[2]
                    elif row[2] == '3' and (last_row == '2' or last_row == '0'):
                        last_row = row[2]
                    elif row[2] == '4' and (last_row == '3' or last_row == '0'):
                        last_row = row[2]
                    else:
                        patient_id = root
                        wrong_list.append(patient_id)
                        print(f'patient {patient_id} heart beats order are wrong')
                        last_row = '0'
                    # last_row = row[2]
    return wrong_list


if __name__ == '__main__':

    # 进行数据分折
    original_dataset_folder = r"E:\sdmurmur\calibrated_train_vali_dataset"
    kfold_out = "data_kfold_cut_zero"  # grade:soft和loud均匀分折。location:对于present个体，只复制murmur存在的.wav文件
    dataset_split_kfold(original_dataset_folder, kfold_out, kfold=5)

    # # 对测试集进行切分和s1,s1幅值缩放操作
    # test_data_folder = r"D:\shoudu\calibrated_test_data"  # 校正过的测试集路径
    # scaled_test_folder = "test_data_double_s2"  # 指定幅值缩放后的路径
    # test_dataset_scale(test_data_folder, scaled_test_folder)

    # # 检查tsv文件是否有标记错误
    # original_dataset_folder = r"D:\shoudu\the-circor-digiscope-phonocardiogram-dataset-1.0.3\test_data"
    # wrong = check_tsv(original_dataset_folder)
    # new_wrong_list = r"D:\shoudu\original_test_wrong_list.txt"
    # with open(new_wrong_list, 'w') as file:
    #     for item in wrong:
    #         file.write(item + '\n')

    # 检查cut_copy_files_s1_s2函数是否正常工作
    # patient_id = '14241'
    # out_directory = r'D:\shoudu\checkout14241'
    # cut_copy_files_s1_s2(original_dataset_folder, patient_id, out_directory)

