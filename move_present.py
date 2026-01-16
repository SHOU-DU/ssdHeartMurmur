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
from pathlib import Path


def balance_dataset(up: str, bp: str, max_absent=2000) ->None:
    if not os.path.exists(bp):
        os.makedirs(bp)
    source_path = Path(up)
    label_files = list(source_path.glob("*.txt"))

    print(f"找到 {len(label_files)} 个标签文件")
    print("开始筛选数据...")

    soft_count = 0
    loud_count = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            content = f.read()

        # 提取信息
        lines = content.strip().split('\n')  # 提取每行标签信息

        # 提取ID（第一行的第一个数字）
        first_line = lines[0].strip().split()
        if not first_line:
            continue

        patient_id = first_line[0]

        # 提取杂音等级
        murmur_grading = None

        for line in lines:
            if line.startswith("#Murmur grading:"):
                murmur_grading = line.replace("#Murmur grading:", "").strip()

        if murmur_grading == "Soft":
            soft_count += 1
            print(f"Soft [{soft_count}]: {patient_id}")
            shutil.copy2(label_file, bp)
            # 在源目录中搜索.wav文件
            wav_files_found = []
            for wav_file in source_path.glob("*.wav"):
                if patient_id in wav_file.name:
                    wav_files_found.append(wav_file)

            for wav_file in wav_files_found:
                shutil.copy2(wav_file, bp)

            if wav_files_found:
                print(f"  找到 {len(wav_files_found)} 个.wav文件")
            else:
                print(f"  警告: 未找到{patient_id}的.wav文件")

        elif murmur_grading == "Loud":
            loud_count += 1
            print(f"Loud [{loud_count}]: {patient_id}")
            shutil.copy2(label_file, bp)
            # 在源目录中搜索.wav文件
            wav_files_found = []
            for wav_file in source_path.glob("*.wav"):
                if patient_id in wav_file.name:
                    wav_files_found.append(wav_file)

            for wav_file in wav_files_found:
                shutil.copy2(wav_file, bp)

            if wav_files_found:
                print(f"  找到 {len(wav_files_found)} 个.wav文件")
            else:
                print(f"  警告: 未找到{patient_id}的.wav文件")

    print("数据筛选完成!")
    print(f"总共处理了 {len(label_files)} 个标签文件")
    print(f"筛选结果:")
    print(f"  Soft: {soft_count} 条")
    print(f"  Loud: {loud_count} 条")


if __name__ == '__main__':
    with_absent_data_path = r"D:\sdmurmur\Qwen2Audio\bp_calibrated_train_vali_16kHz"
    without_absent_data_path = r"D:\sdmurmur\Qwen2Audio\bp_calibrated_train_vali_16kHz_soft_loud"
    balance_dataset(with_absent_data_path, without_absent_data_path)

